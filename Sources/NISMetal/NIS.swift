import Foundation
import Metal

/// NVIDIA Image Scaling (NIS) 1.0.3 on Metal: a spatial upscaler with edge-adaptive sharpening, ported from NVIDIA's
/// MIT-licensed source (github.com/NVIDIAGameWorks/NVIDIAImageScaling, NIS_Scaler.h and NIS_Config.h; notice below).
///
/// The algorithm is NVIDIA's, step for step: brightness of the source block plus a 3-pixel border into threadgroup
/// memory, a 4-direction edge map of it, then per output pixel a 6x6 polyphase scaler (64 phases) blended with
/// directional filters by the edge map, each with unsharp masking; the change in brightness is added to a bilinear
/// sample of the colour. For Apple GPUs:
/// - the brightness tile, edge map and coefficients are half precision in threadgroup memory, as in NVIDIA's fp16
///   build (twice the occupancy of float), with float maths,
/// - the source is read with texture gathers (three per 2x2 instead of four filtered samples),
/// - the coefficient tables are compiled into the shader instead of read from textures,
/// - block size and HDR mode are compile-time constants.
///
/// NVIDIA rates it for 1x–2x; larger steps work too (the source block only gets smaller), downscaling doesn't.
public final class NISUpscaler: @unchecked Sendable {
    public enum HDR: Int, Sendable { case none = 0, linear = 1, pq = 2 }

    public let inputSize: MTLSize
    public let outputSize: MTLSize
    public let sharpness: Float
    public let hdr: HDR

    private let pipeline: MTLComputePipelineState
    private let block: (w: Int, h: Int)
    private let threads: Int

    /// `sharpness` 0 (none) … 1, as NVIDIA's slider (0.5 is its default).
    public init(device: MTLDevice, inputSize: MTLSize, outputSize: MTLSize, sharpness: Float, hdr: HDR = .none,
                options: Options = Options()) throws {
        guard inputSize.width > 0, inputSize.height > 0, outputSize.width >= inputSize.width,
              outputSize.height >= inputSize.height else {
            throw NISError("NIS upscales only (input \(inputSize.width)x\(inputSize.height), output \(outputSize.width)x\(outputSize.height))")
        }
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.sharpness = sharpness
        self.hdr = hdr
        block = (options.blockWidth, options.blockHeight)
        threads = options.threads
        pipeline = try Self.pipeline(device: device, options: options, hdr: hdr)
    }

    /// Compile-time choices (for tuning; the defaults are the fastest measured on Apple silicon).
    public struct Options: Hashable, Sendable {
        public var blockWidth = 32, blockHeight = 24, threads = 256
        /// Precision: half maths implies half storage. The default is NVIDIA's fp16 build: half storage (half the
        /// threadgroup memory, so twice the occupancy: 0.31 against 0.52 ms at 1728x1117 -> 3456x2234 on an M4 Max)
        /// with float maths. Half maths saves another ~10% but flips edge decisions (outliers up to 38/255) and
        /// loses precision in linear HDR.
        public var halfStorage = true
        public var halfMath = false
        public init() {}
    }

    /// Pipelines are shared by every upscaler on a device with the same options (compiling one takes ~100 ms).
    nonisolated(unsafe) private static var cache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pipeline(device: MTLDevice, options o: Options, hdr: HDR) throws -> MTLComputePipelineState {
        let key = "\(device.registryID) \(o) \(hdr)"
        lock.lock(); defer { lock.unlock() }
        if let p = cache[key] { return p }
        let opts = MTLCompileOptions()
        opts.preprocessorMacros = [
            "NIS_BLOCK_WIDTH": NSNumber(value: o.blockWidth), "NIS_BLOCK_HEIGHT": NSNumber(value: o.blockHeight),
            "NIS_THREAD_GROUP_SIZE": NSNumber(value: o.threads), "NIS_HALF_MATH": NSNumber(value: o.halfMath ? 1 : 0),
            "NIS_HALF_STORAGE": NSNumber(value: o.halfStorage || o.halfMath ? 1 : 0),
            "NIS_HDR_MODE": NSNumber(value: hdr.rawValue),
        ]
        opts.mathMode = .fast
        let library = try device.makeLibrary(source: source, options: opts)
        let p = try device.makeComputePipelineState(function: library.makeFunction(name: "nis_scaler")!)
        guard p.maxTotalThreadsPerThreadgroup >= o.threads else {
            throw NISError("NIS: \(o.threads) threads per group, the device allows \(p.maxTotalThreadsPerThreadgroup)")
        }
        cache[key] = p
        return p
    }

    /// NVIDIA's NISConfig (NIS_Config.h), 112 bytes.
    struct Config {
        var detectRatio: Float = 0, detectThres: Float = 0, minContrastRatio: Float = 0, ratioNorm: Float = 0
        var contrastBoost: Float = 0, eps: Float = 0, sharpStartY: Float = 0, sharpScaleY: Float = 0
        var sharpStrengthMin: Float = 0, sharpStrengthScale: Float = 0, sharpLimitMin: Float = 0, sharpLimitScale: Float = 0
        var scaleX: Float = 0, scaleY: Float = 0, dstNormX: Float = 0, dstNormY: Float = 0
        var srcNormX: Float = 0, srcNormY: Float = 0
        var inOriginX: UInt32 = 0, inOriginY: UInt32 = 0, inWidth: UInt32 = 0, inHeight: UInt32 = 0
        var outOriginX: UInt32 = 0, outOriginY: UInt32 = 0, outWidth: UInt32 = 0, outHeight: UInt32 = 0
        var reserved0: Float = 0, reserved1: Float = 0
    }

    /// NVScalerUpdateConfig, as NVIDIA wrote it.
    static func config(sharpness: Float, hdr: HDR, input: (x: Int, y: Int, w: Int, h: Int), inputTexture: (w: Int, h: Int),
                       output: (x: Int, y: Int, w: Int, h: Int), outputTexture: (w: Int, h: Int)) -> Config {
        let slider = max(0, min(1, sharpness)) - 0.5   // 0 … 1 -> -0.5 … +0.5
        let maxScale: Float = slider >= 0 ? 1.25 : 1.75
        let minScale: Float = slider >= 0 ? 1.25 : 1.0
        let limitScale: Float = slider >= 0 ? 1.25 : 1.0

        var detectThres: Float = 64.0 / 1024.0
        var minContrastRatio: Float = 2.0, maxContrastRatio: Float = 10.0
        var sharpStartY: Float = 0.45, sharpEndY: Float = 0.9
        var strengthMin = max(0, 0.4 + slider * minScale * 1.2)
        var strengthMax = 1.6 + slider * maxScale * 1.8
        var limitMin = max(0.1, 0.14 + slider * limitScale * 0.32)
        var limitMax = 0.5 + slider * limitScale * 0.6
        if hdr != .none {
            detectThres = 32.0 / 1024.0
            minContrastRatio = 1.5; maxContrastRatio = 5.0
            strengthMin = max(0, 0.4 + slider * minScale * 1.1)
            strengthMax = 2.2 + slider * maxScale * 1.8
            limitMin = max(0.06, 0.10 + slider * limitScale * 0.28)
            limitMax = 0.6 + slider * limitScale * 0.6
            if hdr == .pq { sharpStartY = 0.35; sharpEndY = 0.55 } else { sharpStartY = 0.3; sharpEndY = 0.5 }
        }
        var c = Config()
        c.detectRatio = 2 * 1127.0 / 1024.0
        c.detectThres = detectThres
        c.minContrastRatio = minContrastRatio
        c.ratioNorm = 1 / (maxContrastRatio - minContrastRatio)
        c.contrastBoost = 1
        c.eps = 1.0 / 255.0
        c.sharpStartY = sharpStartY
        c.sharpScaleY = 1 / (sharpEndY - sharpStartY)
        c.sharpStrengthMin = strengthMin
        c.sharpStrengthScale = strengthMax - strengthMin
        c.sharpLimitMin = limitMin
        c.sharpLimitScale = limitMax - limitMin
        c.inOriginX = UInt32(input.x); c.inOriginY = UInt32(input.y)
        c.inWidth = UInt32(input.w); c.inHeight = UInt32(input.h)
        c.outOriginX = UInt32(output.x); c.outOriginY = UInt32(output.y)
        c.outWidth = UInt32(output.w); c.outHeight = UInt32(output.h)
        c.srcNormX = 1 / Float(inputTexture.w); c.srcNormY = 1 / Float(inputTexture.h)
        c.dstNormX = 1 / Float(outputTexture.w); c.dstNormY = 1 / Float(outputTexture.h)
        c.scaleX = Float(input.w) / Float(output.w); c.scaleY = Float(input.h) / Float(output.h)
        return c
    }

    /// NVIDIA's NISConfig bytes for a whole-texture upscale, for running other builds of NIS (`lss-metal nis-check`).
    public static func configBytes(sharpness: Float, hdr: HDR, input: (w: Int, h: Int), output: (w: Int, h: Int)) -> [UInt8] {
        var c = config(sharpness: sharpness, hdr: hdr, input: (0, 0, input.w, input.h), inputTexture: input,
                       output: (0, 0, output.w, output.h), outputTexture: output)
        return withUnsafeBytes(of: &c) { Array($0) }
    }

    /// Encodes the upscale of `input`'s region at `inputOrigin` into `output` at `outputOrigin`. `output` needs
    /// shader-write usage.
    public func encode(commandBuffer cb: MTLCommandBuffer, input: MTLTexture, inputOrigin: MTLOrigin = MTLOrigin(),
                       output: MTLTexture, outputOrigin: MTLOrigin = MTLOrigin()) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "NIS"
        encode(encoder: enc, input: input, inputOrigin: inputOrigin, output: output, outputOrigin: outputOrigin)
        enc.endEncoding()
    }

    public func encode(encoder enc: MTLComputeCommandEncoder, input: MTLTexture, inputOrigin: MTLOrigin = MTLOrigin(),
                       output: MTLTexture, outputOrigin: MTLOrigin = MTLOrigin()) {
        var c = Self.config(sharpness: sharpness, hdr: hdr,
                            input: (inputOrigin.x, inputOrigin.y, inputSize.width, inputSize.height),
                            inputTexture: (input.width, input.height),
                            output: (outputOrigin.x, outputOrigin.y, outputSize.width, outputSize.height),
                            outputTexture: (output.width, output.height))
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&c, length: MemoryLayout<Config>.stride, index: 0)
        enc.setTexture(input, index: 0)
        enc.setTexture(output, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: (outputSize.width + block.w - 1) / block.w,
                                         height: (outputSize.height + block.h - 1) / block.h, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    // MARK: - Shader

    /// NVIDIA's coef_scale and coef_usm (NIS_Config.h): 64 phases x 6 taps (their last two of 8 are always 0).
    public static let coefScale: [[Float]] = [
        [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.0029, -0.0127, 1.0000, 0.0132, -0.0034, 0.0000],
        [0.0063, -0.0249, 0.9985, 0.0269, -0.0068, 0.0000],
        [0.0088, -0.0361, 0.9956, 0.0415, -0.0103, 0.0005],
        [0.0117, -0.0474, 0.9932, 0.0562, -0.0142, 0.0005],
        [0.0142, -0.0576, 0.9897, 0.0713, -0.0181, 0.0005],
        [0.0166, -0.0674, 0.9844, 0.0874, -0.0220, 0.0010],
        [0.0186, -0.0762, 0.9785, 0.1040, -0.0264, 0.0015],
        [0.0205, -0.0850, 0.9727, 0.1206, -0.0308, 0.0020],
        [0.0225, -0.0928, 0.9648, 0.1382, -0.0352, 0.0024],
        [0.0239, -0.1006, 0.9575, 0.1558, -0.0396, 0.0029],
        [0.0254, -0.1074, 0.9487, 0.1738, -0.0439, 0.0034],
        [0.0264, -0.1138, 0.9390, 0.1929, -0.0488, 0.0044],
        [0.0278, -0.1191, 0.9282, 0.2119, -0.0537, 0.0049],
        [0.0288, -0.1245, 0.9170, 0.2310, -0.0581, 0.0059],
        [0.0293, -0.1294, 0.9058, 0.2510, -0.0630, 0.0063],
        [0.0303, -0.1333, 0.8926, 0.2710, -0.0679, 0.0073],
        [0.0308, -0.1367, 0.8789, 0.2915, -0.0728, 0.0083],
        [0.0308, -0.1401, 0.8657, 0.3120, -0.0776, 0.0093],
        [0.0313, -0.1426, 0.8506, 0.3330, -0.0825, 0.0103],
        [0.0313, -0.1445, 0.8354, 0.3540, -0.0874, 0.0112],
        [0.0313, -0.1460, 0.8193, 0.3755, -0.0923, 0.0122],
        [0.0313, -0.1470, 0.8022, 0.3965, -0.0967, 0.0137],
        [0.0308, -0.1479, 0.7856, 0.4185, -0.1016, 0.0146],
        [0.0303, -0.1479, 0.7681, 0.4399, -0.1060, 0.0156],
        [0.0298, -0.1479, 0.7505, 0.4614, -0.1104, 0.0166],
        [0.0293, -0.1470, 0.7314, 0.4829, -0.1147, 0.0181],
        [0.0288, -0.1460, 0.7119, 0.5049, -0.1187, 0.0190],
        [0.0278, -0.1445, 0.6929, 0.5264, -0.1226, 0.0200],
        [0.0273, -0.1431, 0.6724, 0.5479, -0.1260, 0.0215],
        [0.0264, -0.1411, 0.6528, 0.5693, -0.1299, 0.0225],
        [0.0254, -0.1387, 0.6323, 0.5903, -0.1328, 0.0234],
        [0.0244, -0.1357, 0.6113, 0.6113, -0.1357, 0.0244],
        [0.0234, -0.1328, 0.5903, 0.6323, -0.1387, 0.0254],
        [0.0225, -0.1299, 0.5693, 0.6528, -0.1411, 0.0264],
        [0.0215, -0.1260, 0.5479, 0.6724, -0.1431, 0.0273],
        [0.0200, -0.1226, 0.5264, 0.6929, -0.1445, 0.0278],
        [0.0190, -0.1187, 0.5049, 0.7119, -0.1460, 0.0288],
        [0.0181, -0.1147, 0.4829, 0.7314, -0.1470, 0.0293],
        [0.0166, -0.1104, 0.4614, 0.7505, -0.1479, 0.0298],
        [0.0156, -0.1060, 0.4399, 0.7681, -0.1479, 0.0303],
        [0.0146, -0.1016, 0.4185, 0.7856, -0.1479, 0.0308],
        [0.0137, -0.0967, 0.3965, 0.8022, -0.1470, 0.0313],
        [0.0122, -0.0923, 0.3755, 0.8193, -0.1460, 0.0313],
        [0.0112, -0.0874, 0.3540, 0.8354, -0.1445, 0.0313],
        [0.0103, -0.0825, 0.3330, 0.8506, -0.1426, 0.0313],
        [0.0093, -0.0776, 0.3120, 0.8657, -0.1401, 0.0308],
        [0.0083, -0.0728, 0.2915, 0.8789, -0.1367, 0.0308],
        [0.0073, -0.0679, 0.2710, 0.8926, -0.1333, 0.0303],
        [0.0063, -0.0630, 0.2510, 0.9058, -0.1294, 0.0293],
        [0.0059, -0.0581, 0.2310, 0.9170, -0.1245, 0.0288],
        [0.0049, -0.0537, 0.2119, 0.9282, -0.1191, 0.0278],
        [0.0044, -0.0488, 0.1929, 0.9390, -0.1138, 0.0264],
        [0.0034, -0.0439, 0.1738, 0.9487, -0.1074, 0.0254],
        [0.0029, -0.0396, 0.1558, 0.9575, -0.1006, 0.0239],
        [0.0024, -0.0352, 0.1382, 0.9648, -0.0928, 0.0225],
        [0.0020, -0.0308, 0.1206, 0.9727, -0.0850, 0.0205],
        [0.0015, -0.0264, 0.1040, 0.9785, -0.0762, 0.0186],
        [0.0010, -0.0220, 0.0874, 0.9844, -0.0674, 0.0166],
        [0.0005, -0.0181, 0.0713, 0.9897, -0.0576, 0.0142],
        [0.0005, -0.0142, 0.0562, 0.9932, -0.0474, 0.0117],
        [0.0005, -0.0103, 0.0415, 0.9956, -0.0361, 0.0088],
        [0.0000, -0.0068, 0.0269, 0.9985, -0.0249, 0.0063],
        [0.0000, -0.0034, 0.0132, 1.0000, -0.0127, 0.0029],
    ]
    public static let coefUSM: [[Float]] = [
        [0.0000, -0.6001, 1.2002, -0.6001, 0.0000, 0.0000],
        [0.0029, -0.6084, 1.1987, -0.5903, -0.0029, 0.0000],
        [0.0049, -0.6147, 1.1958, -0.5791, -0.0068, 0.0005],
        [0.0073, -0.6196, 1.1890, -0.5659, -0.0103, 0.0000],
        [0.0093, -0.6235, 1.1802, -0.5513, -0.0151, 0.0000],
        [0.0112, -0.6265, 1.1699, -0.5352, -0.0195, 0.0005],
        [0.0122, -0.6270, 1.1582, -0.5181, -0.0259, 0.0005],
        [0.0142, -0.6284, 1.1455, -0.5005, -0.0317, 0.0005],
        [0.0156, -0.6265, 1.1274, -0.4790, -0.0386, 0.0005],
        [0.0166, -0.6235, 1.1089, -0.4570, -0.0454, 0.0010],
        [0.0176, -0.6187, 1.0879, -0.4346, -0.0532, 0.0010],
        [0.0181, -0.6138, 1.0659, -0.4102, -0.0615, 0.0015],
        [0.0190, -0.6069, 1.0405, -0.3843, -0.0698, 0.0015],
        [0.0195, -0.6006, 1.0161, -0.3574, -0.0796, 0.0020],
        [0.0200, -0.5928, 0.9893, -0.3286, -0.0898, 0.0024],
        [0.0200, -0.5820, 0.9580, -0.2988, -0.1001, 0.0029],
        [0.0200, -0.5728, 0.9292, -0.2690, -0.1104, 0.0034],
        [0.0200, -0.5620, 0.8975, -0.2368, -0.1226, 0.0039],
        [0.0205, -0.5498, 0.8643, -0.2046, -0.1343, 0.0044],
        [0.0200, -0.5371, 0.8301, -0.1709, -0.1465, 0.0049],
        [0.0195, -0.5239, 0.7944, -0.1367, -0.1587, 0.0054],
        [0.0195, -0.5107, 0.7598, -0.1021, -0.1724, 0.0059],
        [0.0190, -0.4966, 0.7231, -0.0649, -0.1865, 0.0063],
        [0.0186, -0.4819, 0.6846, -0.0288, -0.1997, 0.0068],
        [0.0186, -0.4668, 0.6460, 0.0093, -0.2144, 0.0073],
        [0.0176, -0.4507, 0.6055, 0.0479, -0.2290, 0.0083],
        [0.0171, -0.4370, 0.5693, 0.0859, -0.2446, 0.0088],
        [0.0161, -0.4199, 0.5283, 0.1255, -0.2598, 0.0098],
        [0.0161, -0.4048, 0.4883, 0.1655, -0.2754, 0.0103],
        [0.0151, -0.3887, 0.4497, 0.2041, -0.2910, 0.0107],
        [0.0142, -0.3711, 0.4072, 0.2446, -0.3066, 0.0117],
        [0.0137, -0.3555, 0.3672, 0.2852, -0.3228, 0.0122],
        [0.0132, -0.3394, 0.3262, 0.3262, -0.3394, 0.0132],
        [0.0122, -0.3228, 0.2852, 0.3672, -0.3555, 0.0137],
        [0.0117, -0.3066, 0.2446, 0.4072, -0.3711, 0.0142],
        [0.0107, -0.2910, 0.2041, 0.4497, -0.3887, 0.0151],
        [0.0103, -0.2754, 0.1655, 0.4883, -0.4048, 0.0161],
        [0.0098, -0.2598, 0.1255, 0.5283, -0.4199, 0.0161],
        [0.0088, -0.2446, 0.0859, 0.5693, -0.4370, 0.0171],
        [0.0083, -0.2290, 0.0479, 0.6055, -0.4507, 0.0176],
        [0.0073, -0.2144, 0.0093, 0.6460, -0.4668, 0.0186],
        [0.0068, -0.1997, -0.0288, 0.6846, -0.4819, 0.0186],
        [0.0063, -0.1865, -0.0649, 0.7231, -0.4966, 0.0190],
        [0.0059, -0.1724, -0.1021, 0.7598, -0.5107, 0.0195],
        [0.0054, -0.1587, -0.1367, 0.7944, -0.5239, 0.0195],
        [0.0049, -0.1465, -0.1709, 0.8301, -0.5371, 0.0200],
        [0.0044, -0.1343, -0.2046, 0.8643, -0.5498, 0.0205],
        [0.0039, -0.1226, -0.2368, 0.8975, -0.5620, 0.0200],
        [0.0034, -0.1104, -0.2690, 0.9292, -0.5728, 0.0200],
        [0.0029, -0.1001, -0.2988, 0.9580, -0.5820, 0.0200],
        [0.0024, -0.0898, -0.3286, 0.9893, -0.5928, 0.0200],
        [0.0020, -0.0796, -0.3574, 1.0161, -0.6006, 0.0195],
        [0.0015, -0.0698, -0.3843, 1.0405, -0.6069, 0.0190],
        [0.0015, -0.0615, -0.4102, 1.0659, -0.6138, 0.0181],
        [0.0010, -0.0532, -0.4346, 1.0879, -0.6187, 0.0176],
        [0.0010, -0.0454, -0.4570, 1.1089, -0.6235, 0.0166],
        [0.0005, -0.0386, -0.4790, 1.1274, -0.6265, 0.0156],
        [0.0005, -0.0317, -0.5005, 1.1455, -0.6284, 0.0142],
        [0.0005, -0.0259, -0.5181, 1.1582, -0.6270, 0.0122],
        [0.0005, -0.0195, -0.5352, 1.1699, -0.6265, 0.0112],
        [0.0000, -0.0151, -0.5513, 1.1802, -0.6235, 0.0093],
        [0.0000, -0.0103, -0.5659, 1.1890, -0.6196, 0.0073],
        [0.0005, -0.0068, -0.5791, 1.1958, -0.6147, 0.0049],
        [0.0000, -0.0029, -0.5903, 1.1987, -0.6084, 0.0029],
    ]

    private static func table(_ t: [[Float]]) -> String {
        t.map { "{" + $0.map { String(format: "%.4f", $0) }.joined(separator: ", ") + "}" }.joined(separator: ",\n    ")
    }

    static let source = """
    // NVIDIA Image Scaling, ported to Metal from NVIDIA's NIS_Scaler.h (v1.0.3). Original notice:
    //
    // The MIT License(MIT)
    //
    // Copyright(c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    //
    // Permission is hereby granted, free of charge, to any person obtaining a copy of
    // this software and associated documentation files(the "Software"), to deal in
    // the Software without restriction, including without limitation the rights to
    // use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
    // the Software, and to permit persons to whom the Software is furnished to do so,
    // subject to the following conditions :
    //
    // The above copyright notice and this permission notice shall be included in all
    // copies or substantial portions of the Software.
    //
    // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    // FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
    // COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    // IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    // CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    #include <metal_stdlib>
    using namespace metal;

    // F: maths; S: the brightness tile and edge map in threadgroup memory (half in NVIDIA's fp16 build, float in its
    // default one)
    #if NIS_HALF_MATH
    typedef half F; typedef half3 F3; typedef half4 F4;
    #else
    typedef float F; typedef float3 F3; typedef float4 F4;
    #endif
    #if NIS_HALF_STORAGE
    typedef half S; typedef half4 S4; typedef half C;
    #else
    typedef float S; typedef float4 S4; typedef float C;
    #endif

    #define kPhaseCount   64
    #define kSupportSize  6
    #define kPadSize      kSupportSize
    #define kTilePitch    (NIS_BLOCK_WIDTH + kPadSize)
    #define kTileSize     (kTilePitch * (NIS_BLOCK_HEIGHT + kPadSize))
    #define kEdgeMapPitch (NIS_BLOCK_WIDTH + 2)
    #define kEdgeMapSize  (kEdgeMapPitch * (NIS_BLOCK_HEIGHT + 2))
    #define kHDRCompressionFactor 0.282842712f

    struct NISConfig {
        float kDetectRatio, kDetectThres, kMinContrastRatio, kRatioNorm;
        float kContrastBoost, kEps, kSharpStartY, kSharpScaleY;
        float kSharpStrengthMin, kSharpStrengthScale, kSharpLimitMin, kSharpLimitScale;
        float kScaleX, kScaleY, kDstNormX, kDstNormY;
        float kSrcNormX, kSrcNormY;
        uint kInputViewportOriginX, kInputViewportOriginY, kInputViewportWidth, kInputViewportHeight;
        uint kOutputViewportOriginX, kOutputViewportOriginY, kOutputViewportWidth, kOutputViewportHeight;
        float reserved0, reserved1;
    };

    // NVIDIA's coef_scale and coef_usm (NIS_Config.h): 64 phases x 6 taps.
    constant C kCoefScale[kPhaseCount][6] = {
    \(table(coefScale))
    };
    constant C kCoefUSM[kPhaseCount][6] = {
    \(table(coefUSM))
    };

    static inline F getY(F3 c) {
    #if NIS_HDR_MODE == 2
        return F(0.262f) * c.x + F(0.678f) * c.y + F(0.0593f) * c.z;
    #elif NIS_HDR_MODE == 1
        return sqrt(F(0.2126f) * c.x + F(0.7152f) * c.y + F(0.0722f) * c.z) * F(kHDRCompressionFactor);
    #else
        return F(0.2126f) * c.x + F(0.7152f) * c.y + F(0.0722f) * c.z;
    #endif
    }

    static inline F getYLinear(F3 c) { return F(0.2126f) * c.x + F(0.7152f) * c.y + F(0.0722f) * c.z; }

    static inline F4 getEdgeMap(thread const F (&p)[4][4], int i, int j, constant NISConfig& c) {
        const F g_0 = abs(p[0 + i][0 + j] + p[0 + i][1 + j] + p[0 + i][2 + j] - p[2 + i][0 + j] - p[2 + i][1 + j] - p[2 + i][2 + j]);
        const F g_45 = abs(p[1 + i][0 + j] + p[0 + i][0 + j] + p[0 + i][1 + j] - p[2 + i][1 + j] - p[2 + i][2 + j] - p[1 + i][2 + j]);
        const F g_90 = abs(p[0 + i][0 + j] + p[1 + i][0 + j] + p[2 + i][0 + j] - p[0 + i][2 + j] - p[1 + i][2 + j] - p[2 + i][2 + j]);
        const F g_135 = abs(p[1 + i][0 + j] + p[2 + i][0 + j] + p[2 + i][1 + j] - p[0 + i][1 + j] - p[0 + i][2 + j] - p[1 + i][2 + j]);

        const F g_0_90_max = max(g_0, g_90), g_0_90_min = min(g_0, g_90);
        const F g_45_135_max = max(g_45, g_135), g_45_135_min = min(g_45, g_135);
        if (g_0_90_max + g_45_135_max == F(0)) return F4(0);

        const F e_0_90 = min(g_0_90_max / (g_0_90_max + g_45_135_max), F(1));
        const F e_45_135 = F(1) - e_0_90;

        const bool c_0_90 = (g_0_90_max > g_0_90_min * F(c.kDetectRatio)) && (g_0_90_max > F(c.kDetectThres)) && (g_0_90_max > g_45_135_min);
        const bool c_45_135 = (g_45_135_max > g_45_135_min * F(c.kDetectRatio)) && (g_45_135_max > F(c.kDetectThres)) && (g_45_135_max > g_0_90_min);
        const bool c_g_0_90 = g_0_90_max == g_0;
        const bool c_g_45_135 = g_45_135_max == g_45;

        const F f_e_0_90 = (c_0_90 && c_45_135) ? e_0_90 : F(1);
        const F f_e_45_135 = (c_0_90 && c_45_135) ? e_45_135 : F(1);
        return F4((c_0_90 && c_g_0_90) ? f_e_0_90 : F(0),
                  (c_0_90 && !c_g_0_90) ? f_e_0_90 : F(0),
                  (c_45_135 && c_g_45_135) ? f_e_45_135 : F(0),
                  (c_45_135 && !c_g_45_135) ? f_e_45_135 : F(0));
    }

    static inline F calcLTI(F p0, F p1, F p2, F p3, F p4, F p5, int phase, constant NISConfig& c) {
        const bool selector = phase <= kPhaseCount / 2;
        F sel = selector ? p0 : p3;
        const F a_min = min(min(p1, p2), sel), a_max = max(max(p1, p2), sel);
        sel = selector ? p2 : p5;
        const F b_min = min(min(p3, p4), sel), b_max = max(max(p3, p4), sel);
        const F a_cont = a_max - a_min, b_cont = b_max - b_min;
        const F cont_ratio = max(a_cont, b_cont) / (min(a_cont, b_cont) + F(c.kEps));
        return (F(1) - saturate((cont_ratio - F(c.kMinContrastRatio)) * F(c.kRatioNorm))) * F(c.kContrastBoost);
    }

    static inline F evalPoly6(thread const F (&px)[6], int phase, threadgroup const C (&cs)[kPhaseCount][6],
                              threadgroup const C (&cu)[kPhaseCount][6], constant NISConfig& c) {
        F y = 0, y_usm = 0;
        for (int i = 0; i < 6; ++i) { y += F(cs[phase][i]) * px[i]; y_usm += F(cu[phase][i]) * px[i]; }
        // a piece-wise ramp on brightness: sharpen, and limit the sharpening, as a function of it
        const F y_scale = F(1) - saturate((y - F(c.kSharpStartY)) * F(c.kSharpScaleY));
        const F y_sharpness = y_scale * F(c.kSharpStrengthScale) + F(c.kSharpStrengthMin);
        y_usm *= y_sharpness;
        const F y_sharpness_limit = (y_scale * F(c.kSharpLimitScale) + F(c.kSharpLimitMin)) * y;
        y_usm = min(y_sharpness_limit, max(-y_sharpness_limit, y_usm));
        y_usm *= calcLTI(px[0], px[1], px[2], px[3], px[4], px[5], phase, c);   // reduce ringing
        return y + y_usm;
    }

    kernel void nis_scaler(constant NISConfig& c [[buffer(0)]],
                           texture2d<half> in_texture [[texture(0)]],
                           texture2d<half, access::write> out_texture [[texture(1)]],
                           uint2 blockIdx [[threadgroup_position_in_grid]],
                           uint threadIdx [[thread_index_in_threadgroup]]) {
        constexpr sampler linearClamp(filter::linear, address::clamp_to_edge, coord::normalized);
        threadgroup S shPixelsY[kTileSize];
        threadgroup C shCoefScaler[kPhaseCount][6];
        threadgroup C shCoefUSM[kPhaseCount][6];
        threadgroup S4 shEdgeMap[kEdgeMapSize];

        // the source pixels this block needs
        const int dstBlockX = NIS_BLOCK_WIDTH * int(blockIdx.x);
        const int dstBlockY = NIS_BLOCK_HEIGHT * int(blockIdx.y);
        const int srcBlockStartX = int(floor((dstBlockX + 0.5f) * c.kScaleX - 0.5f));
        const int srcBlockStartY = int(floor((dstBlockY + 0.5f) * c.kScaleY - 0.5f));
        const int srcBlockEndX = int(ceil((dstBlockX + NIS_BLOCK_WIDTH + 0.5f) * c.kScaleX - 0.5f));
        const int srcBlockEndY = int(ceil((dstBlockY + NIS_BLOCK_HEIGHT + 0.5f) * c.kScaleY - 0.5f));
        int numTilePixelsX = srcBlockEndX - srcBlockStartX + kSupportSize - 1;
        int numTilePixelsY = srcBlockEndY - srcBlockStartY + kSupportSize - 1;
        numTilePixelsX += numTilePixelsX & 1;   // even: loaded in 2x2s
        numTilePixelsY += numTilePixelsY & 1;
        const int numTilePixels = numTilePixelsX * numTilePixelsY;
        const int numEdgeMapPixelsX = numTilePixelsX - kSupportSize + 2;
        const int numEdgeMapPixelsY = numTilePixelsY - kSupportSize + 2;
        const int numEdgeMapPixels = numEdgeMapPixelsX * numEdgeMapPixelsY;

        // brightness tile, 2x2 at a time: three gathers at the corner between the four pixels
        for (uint i = threadIdx * 2; i < uint(numTilePixels) >> 1; i += NIS_THREAD_GROUP_SIZE * 2) {
            const uint py = (i / numTilePixelsX) * 2;
            const uint px = i % numTilePixelsX;
            // the 2x2's top-left source pixel, shifted by the filter's support ((6 - 1) / 2 = 2.5 -> 2 whole pixels)
            const float2 corner = float2(int(c.kInputViewportOriginX) + srcBlockStartX + int(px) - 2 + 1,
                                         int(c.kInputViewportOriginY) + srcBlockStartY + int(py) - 2 + 1);
            const float2 t = corner * float2(c.kSrcNormX, c.kSrcNormY);
            const half4 sr = in_texture.gather(linearClamp, t, int2(0), component::x);
            const half4 sg = in_texture.gather(linearClamp, t, int2(0), component::y);
            const half4 sb = in_texture.gather(linearClamp, t, int2(0), component::z);
            // gather order: (x, y+1), (x+1, y+1), (x+1, y), (x, y)
            const uint idx = py * kTilePitch + px;
            shPixelsY[idx]                  = S(getY(F3(sr.w, sg.w, sb.w)));
            shPixelsY[idx + 1]              = S(getY(F3(sr.z, sg.z, sb.z)));
            shPixelsY[idx + kTilePitch]     = S(getY(F3(sr.x, sg.x, sb.x)));
            shPixelsY[idx + kTilePitch + 1] = S(getY(F3(sr.y, sg.y, sb.y)));
        }
        // filter banks
        for (uint i = threadIdx; i < kPhaseCount * 6; i += NIS_THREAD_GROUP_SIZE) {
            shCoefScaler[i / 6][i % 6] = kCoefScale[i / 6][i % 6];
            shCoefUSM[i / 6][i % 6] = kCoefUSM[i / 6][i % 6];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // edge map, 2x2 at a time
        for (uint i = threadIdx * 2; i < uint(numEdgeMapPixels) >> 1; i += NIS_THREAD_GROUP_SIZE * 2) {
            const uint py = (i / numEdgeMapPixelsX) * 2;
            const uint px = i % numEdgeMapPixelsX;
            const uint edgeMapIdx = py * kEdgeMapPitch + px;
            const uint tileCornerIdx = (py + 1) * kTilePitch + px + 1;
            F p[4][4];
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++) p[j][k] = F(shPixelsY[tileCornerIdx + j * kTilePitch + k]);
            shEdgeMap[edgeMapIdx]                     = S4(getEdgeMap(p, 0, 0, c));
            shEdgeMap[edgeMapIdx + 1]                 = S4(getEdgeMap(p, 0, 1, c));
            shEdgeMap[edgeMapIdx + kEdgeMapPitch]     = S4(getEdgeMap(p, 1, 0, c));
            shEdgeMap[edgeMapIdx + kEdgeMapPitch + 1] = S4(getEdgeMap(p, 1, 1, c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int2 pos = int2(threadIdx % NIS_BLOCK_WIDTH, threadIdx / NIS_BLOCK_WIDTH);
        const int dstX = dstBlockX + pos.x;
        if (uint(dstX) >= c.kOutputViewportWidth) return;
        const float srcX = (0.5f + dstX) * c.kScaleX - 0.5f;
        const int px = int(floor(srcX) - srcBlockStartX);
        const float fxf = srcX - floor(srcX);
        const F fx = F(fxf);
        const int fx_int = int(fxf * kPhaseCount);

        for (int k = 0; k < NIS_BLOCK_WIDTH * NIS_BLOCK_HEIGHT / NIS_THREAD_GROUP_SIZE; ++k) {
            const int dstY = dstBlockY + pos.y + k * (NIS_THREAD_GROUP_SIZE / NIS_BLOCK_WIDTH);
            if (uint(dstY) >= c.kOutputViewportHeight) break;
            const float srcY = (0.5f + dstY) * c.kScaleY - 0.5f;
            const int py = int(floor(srcY) - srcBlockStartY);
            const float fyf = srcY - floor(srcY);
            const F fy = F(fyf);
            const int fy_int = int(fyf * kPhaseCount);

            // weights of the directional filters: the edge map, interpolated
            const int e = py * kEdgeMapPitch + px;
            const F4 h0 = mix(F4(shEdgeMap[e]), F4(shEdgeMap[e + 1]), fx);
            const F4 h1 = mix(F4(shEdgeMap[e + kEdgeMapPitch]), F4(shEdgeMap[e + kEdgeMapPitch + 1]), fx);
            const F4 w = mix(h0, h1, fy);

            // 6x6 support
            const int t = py * kTilePitch + px;
            F p[6][6];
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) p[i][j] = F(shPixelsY[t + i * kTilePitch + j]);

            // the plain polyphase scaler, weighted by what the directional filters leave
            F cy[6], cx[6];
            for (int i = 0; i < 6; ++i) { cy[i] = F(shCoefScaler[fy_int][i]); cx[i] = F(shCoefScaler[fx_int][i]); }
            F h_acc = 0;
            for (int j = 0; j < 6; ++j) {
                F v_acc = 0;
                for (int i = 0; i < 6; ++i) v_acc += p[i][j] * cy[i];
                h_acc += v_acc * cx[j];
            }
            F opY = h_acc * (F(1) - w.x - w.y - w.z - w.w);

            if (w.x > F(0)) {   // 0 degrees
                F v[6];
                for (int i = 0; i < 6; ++i) v[i] = mix(p[i][2], p[i][3], fx);
                opY += evalPoly6(v, fy_int, shCoefScaler, shCoefUSM, c) * w.x;
            }
            if (w.y > F(0)) {   // 90 degrees
                F v[6];
                for (int i = 0; i < 6; ++i) v[i] = mix(p[2][i], p[3][i], fy);
                opY += evalPoly6(v, fx_int, shCoefScaler, shCoefUSM, c) * w.y;
            }
            if (w.z > F(0)) {   // 45 degrees
                F b = F(0.5f) + F(0.5f) * (fx - fy);
                F tmp[7];
                tmp[1] = mix(p[2][1], p[1][2], b);
                tmp[3] = mix(p[3][2], p[2][3], b);
                tmp[5] = mix(p[4][3], p[3][4], b);
                b = b - F(0.5f);
                const bool pos45 = b >= F(0);
                const F ab = abs(b);
                tmp[0] = mix(p[1][1], pos45 ? p[0][2] : p[2][0], ab);
                tmp[2] = mix(p[2][2], pos45 ? p[1][3] : p[3][1], ab);
                tmp[4] = mix(p[3][3], pos45 ? p[2][4] : p[4][2], ab);
                tmp[6] = mix(p[4][4], pos45 ? p[3][5] : p[5][3], ab);
                F v[6];
                float ph = fxf + fyf;
                const int o = ph >= 1 ? 1 : 0;
                ph -= float(o);
                for (int i = 0; i < 6; ++i) v[i] = tmp[i + o];
                opY += evalPoly6(v, int(ph * 64), shCoefScaler, shCoefUSM, c) * w.z;
            }
            if (w.w > F(0)) {   // 135 degrees
                F b = F(0.5f) * (fx + fy);
                F tmp[7];
                tmp[1] = mix(p[3][1], p[4][2], b);
                tmp[3] = mix(p[2][2], p[3][3], b);
                tmp[5] = mix(p[1][3], p[2][4], b);
                b = b - F(0.5f);
                const bool pos135 = b >= F(0);
                const F ab = abs(b);
                tmp[0] = mix(p[4][1], pos135 ? p[5][2] : p[3][0], ab);
                tmp[2] = mix(p[3][2], pos135 ? p[4][3] : p[2][1], ab);
                tmp[4] = mix(p[2][3], pos135 ? p[3][4] : p[1][2], ab);
                tmp[6] = mix(p[1][4], pos135 ? p[2][5] : p[0][3], ab);
                F v[6];
                float ph = 1 + (fxf - fyf);
                const int o = ph >= 1 ? 1 : 0;
                ph -= float(o);
                for (int i = 0; i < 6; ++i) v[i] = tmp[i + o];
                opY += evalPoly6(v, int(ph * 64), shCoefScaler, shCoefUSM, c) * w.w;
            }

            // colour: a bilinear sample, with the brightness change added
            const float2 coord = float2((srcX + c.kInputViewportOriginX + 0.5f) * c.kSrcNormX,
                                        (srcY + c.kInputViewportOriginY + 0.5f) * c.kSrcNormY);
            half4 op = in_texture.sample(linearClamp, coord, level(0));
    #if NIS_HDR_MODE == 1
            const float kNorm = 1.0f / kHDRCompressionFactor;
            const float opYN = max(float(opY), 0.0f) * kNorm;
            const float corr = (opYN * opYN + 1e-4f) / (max(float(getYLinear(F3(op.xyz))), 0.0f) + 1e-4f);
            op.xyz = half3(float3(op.xyz) * corr);
    #else
            const F corr = opY - getY(F3(op.xyz));
            op.xyz = half3(F3(op.xyz) + corr);
    #endif
            out_texture.write(op, uint2(dstX + int(c.kOutputViewportOriginX), dstY + int(c.kOutputViewportOriginY)));
        }
    }
    """
}

public struct NISError: Error, CustomStringConvertible {
    public let description: String
    init(_ description: String) { self.description = description }
}
