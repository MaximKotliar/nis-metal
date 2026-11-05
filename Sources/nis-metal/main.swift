import CoreGraphics
import Foundation
import ImageIO
import Metal
import NISMetal
import UniformTypeIdentifiers

let usage = """
usage: nis-metal <command>
  upscale IN.png -o OUT.png [--sharpness S] [--scale F | --size WxH] [--runs N]
      S: 0 (none) … 1 (0.5: NVIDIA's default); F: default 2
  bench IN.png [--scale F]
      shrink IN by F, upscale it back with every block size and precision: GPU time, PSNR vs the float build
  check DIR IN.png [--scale F]
      compare with NVIDIA's own compiled NIS, translated to Metal: DIR/243.metal (SDR) and DIR/244.metal (linear HDR),
      e.g. from Lossless Scaling's Lossless.dll (which ships it as DXBC shaders 243-246) via dxbc-spirv and spirv-cross
"""

func fail(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(1) }

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else { fail(usage) }
args.removeFirst()
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fail("no Metal device") }

// MARK: - Images

/// 8-bit RGBA, sRGB values, rows top to bottom.
struct Image { var width: Int, height: Int, rgba: [UInt8] }

func load(_ path: String) -> Image {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { fail("can't read \(path)") }
    return draw(img, width: img.width, height: img.height)
}

func draw(_ img: CGImage, width: Int, height: Int) -> Image {
    var px = [UInt8](repeating: 0, count: width * height * 4)
    let ctx = CGContext(data: &px, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
    return Image(width: width, height: height, rgba: px)
}

func cgImage(_ i: Image) -> CGImage {
    let provider = CGDataProvider(data: Data(i.rgba) as CFData)!
    return CGImage(width: i.width, height: i.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: i.width * 4,
                   space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                   provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
}

func save(_ i: Image, _ path: String) {
    let d = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(d, cgImage(i), nil)
    guard CGImageDestinationFinalize(d) else { fail("can't write \(path)") }
}

/// High-quality resize (for making a small input from a sharp original).
func resized(_ i: Image, width: Int, height: Int) -> Image { draw(cgImage(i), width: width, height: height) }

func psnr(_ a: [Float], _ b: [Float], skip: (Int) -> Bool = { _ in false }) -> Double {
    var se = 0.0, n = 0.0
    for i in 0..<a.count where i % 4 != 3 && !skip(i / 4) {
        let d = Double(a[i] - b[i]); se += d * d; n += 1
    }
    return se == 0 ? 99 : 10 * log10(1 / (se / n))
}

// MARK: - Textures (RGBA16F, so comparisons see more than 8 bits)

/// Float to fp16 bits, round to nearest even (Float16 doesn't exist on x86_64).
func halfBits(_ f: Float) -> UInt16 {
    let u = f.bitPattern, sign = UInt16((u >> 16) & 0x8000)
    let e = Int((u >> 23) & 0xFF) - 127 + 15
    var m = u & 0x7F_FFFF
    if e >= 31 { return sign | 0x7C00 }
    if e <= 0 {
        if e < -10 { return sign }
        m |= 0x80_0000
        let shift = UInt32(14 - e)
        var h = m >> shift
        if (m >> (shift - 1)) & 1 != 0 && ((m & ((1 << (shift - 1)) - 1)) != 0 || h & 1 != 0) { h += 1 }
        return sign | UInt16(h)
    }
    var h = UInt32(e) << 10 | (m >> 13)
    if m & 0x1000 != 0 && (m & 0xFFF != 0 || h & 1 != 0) { h += 1 }
    return sign | UInt16(h)
}

func halfToFloat(_ h: UInt16) -> Float {
    let sign: Float = h & 0x8000 != 0 ? -1 : 1
    let e = Int((h >> 10) & 0x1F), m = Float(h & 0x3FF)
    if e == 0 { return sign * m * pow(2, -24) }
    if e == 31 { return m == 0 ? sign * .infinity : .nan }
    return sign * (1 + m / 1024) * pow(2, Float(e - 15))
}

func texture(_ w: Int, _ h: Int) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
    d.usage = [.shaderRead, .shaderWrite]
    d.storageMode = .shared
    return device.makeTexture(descriptor: d)!
}

func upload(_ i: Image) -> MTLTexture {
    let t = texture(i.width, i.height)
    let bits = i.rgba.map { halfBits(Float($0) / 255) }
    t.replace(region: MTLRegionMake2D(0, 0, i.width, i.height), mipmapLevel: 0, withBytes: bits, bytesPerRow: i.width * 8)
    return t
}

func values(_ t: MTLTexture) -> [Float] {
    var bits = [UInt16](repeating: 0, count: t.width * t.height * 4)
    t.getBytes(&bits, bytesPerRow: t.width * 8, from: MTLRegionMake2D(0, 0, t.width, t.height), mipmapLevel: 0)
    return bits.map(halfToFloat)
}

func image(_ t: MTLTexture) -> Image {
    Image(width: t.width, height: t.height, rgba: values(t).map { UInt8((min(max($0, 0), 1) * 255).rounded()) })
}

/// Runs `encode` until clocks are steady, then `batches` command buffers of `perBatch`; the median per run, in ms.
func time(batches: Int = 15, perBatch: Int = 20, _ encode: (MTLCommandBuffer) -> Void) -> Double {
    var times: [Double] = []
    for i in 0..<(batches + 8) {
        let cb = queue.makeCommandBuffer()!
        for _ in 0..<perBatch { encode(cb) }
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { fail("GPU error: \(e)") }
        if i >= 8 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3 / Double(perBatch)) }
    }
    return times.sorted()[times.count / 2]
}

func size(_ s: Int) -> MTLSize { MTLSize(width: s, height: 0, depth: 1) }

// MARK: - Commands

switch command {
case "upscale":
    guard args.count >= 1, let out = option("-o") else { fail(usage) }
    let img = load(args[0])
    var w = img.width * 2, h = img.height * 2
    if let s = option("--scale"), let f = Double(s) { w = Int((Double(img.width) * f).rounded()); h = Int((Double(img.height) * f).rounded()) }
    if let s = option("--size") { let p = s.split(separator: "x").compactMap { Int($0) }; if p.count == 2 { w = p[0]; h = p[1] } }
    let sharpness = Float(option("--sharpness") ?? "") ?? 0.5
    let src = upload(img), dst = texture(w, h)
    let up: NISUpscaler
    do {
        up = try NISUpscaler(device: device, inputSize: MTLSize(width: img.width, height: img.height, depth: 1),
                             outputSize: MTLSize(width: w, height: h, depth: 1), sharpness: sharpness)
    } catch { fail("\(error)") }
    let runs = Int(option("--runs") ?? "") ?? 1
    let ms = runs > 1 ? time(batches: runs) { up.encode(commandBuffer: $0, input: src, output: dst) }
                      : time(batches: 1, perBatch: 1) { up.encode(commandBuffer: $0, input: src, output: dst) }
    save(image(dst), out)
    print("NIS sharpness \(sharpness), \(img.width)x\(img.height) -> \(w)x\(h): GPU \(String(format: "%.3f", ms)) ms, wrote \(out)")

case "bench":
    guard args.count >= 1 else { fail(usage) }
    let truth = load(args[0])
    let f = Double(option("--scale") ?? "") ?? 2
    let small = resized(truth, width: Int(Double(truth.width) / f), height: Int(Double(truth.height) / f))
    let src = upload(small), dst = texture(truth.width, truth.height)
    func run(_ o: NISUpscaler.Options) throws -> ([Float], Double) {
        let up = try NISUpscaler(device: device, inputSize: MTLSize(width: small.width, height: small.height, depth: 1),
                                 outputSize: MTLSize(width: truth.width, height: truth.height, depth: 1), sharpness: 0.5, options: o)
        let ms = time { up.encode(commandBuffer: $0, input: src, output: dst) }
        return (values(dst), ms)
    }
    var ref = NISUpscaler.Options(); ref.halfMath = false; ref.halfStorage = false
    guard let (reference, _) = try? run(ref) else { fail("can't build NIS") }
    print("\(small.width)x\(small.height) -> \(truth.width)x\(truth.height), sharpness 0.5; PSNR vs the all-float 32x24 build (NVIDIA's default):")
    for (bw, bh, t) in [(32, 24, 256), (32, 32, 256), (32, 16, 256), (32, 24, 128), (32, 32, 128), (64, 16, 256), (16, 16, 256), (32, 48, 512)] {
        for (storage, half) in [(false, false), (true, false), (true, true)] {
            guard bw * bh % t == 0, t % bw == 0 else { continue }
            var o = NISUpscaler.Options(); o.blockWidth = bw; o.blockHeight = bh; o.threads = t; o.halfMath = half; o.halfStorage = storage
            let label = String(format: "  block %2dx%-2d %3d threads, %@ storage, %@ maths", bw, bh, t,
                               (storage ? "half " : "float") as NSString, (half ? "half " : "float") as NSString)
            guard let (v, ms) = try? run(o) else { print(label + ": not supported here"); continue }
            print(label + String(format: "   GPU %.3f ms   PSNR %6.2f dB", ms, psnr(v, reference)))
        }
    }

case "check":
    guard args.count >= 2 else { fail(usage) }
    let dir = URL(fileURLWithPath: args[0])
    let truth = load(args[1])
    let f = Double(option("--scale") ?? "") ?? 2
    let small = resized(truth, width: Int(Double(truth.width) / f), height: Int(Double(truth.height) / f))
    let (iw, ih, ow, oh) = (small.width, small.height, truth.width, truth.height)
    let src = upload(small)
    // coefficient textures as NVIDIA makes them: RGBA32F, 2 x 64 (8 taps per phase, the last two 0)
    func coefTexture(_ t: [[Float]]) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 2, height: 64, mipmapped: false)
        d.usage = .shaderRead
        let tex = device.makeTexture(descriptor: d)!
        let v: [Float] = t.flatMap { row -> [Float] in row + [0, 0] }
        tex.replace(region: MTLRegionMake2D(0, 0, 2, 64), mipmapLevel: 0, withBytes: v, bytesPerRow: 2 * 16)
        return tex
    }
    let coefS = coefTexture(NISUpscaler.coefScale), coefU = coefTexture(NISUpscaler.coefUSM)
    let sd = MTLSamplerDescriptor(); sd.minFilter = .linear; sd.magFilter = .linear
    sd.sAddressMode = .clampToEdge; sd.tAddressMode = .clampToEdge
    let sampler = device.makeSamplerState(descriptor: sd)!
    // NVIDIA's bounds check converts the first column's source x (-0.25) to an unsigned integer: 0 in Direct3D,
    // undefined after translation to Metal, so the reference is unreliable within a few pixels of the border.
    let border = { (p: Int) -> Bool in let x = p % ow, y = p / ow; return x < 4 || y < 4 || x >= ow - 4 || y >= oh - 4 }
    func compare(_ a: [Float], _ b: [Float]) -> String {
        var worst: Float = 0, over = 0, n = 0
        for i in 0..<a.count where i % 4 != 3 && !border(i / 4) {
            let d = abs(a[i] - b[i]); worst = max(worst, d); n += 1
            if d > 1.0 / 255 { over += 1 }
        }
        return String(format: "interior PSNR %6.2f dB, largest difference %.1f/255, %.3f%% of values off by more than 1/255",
                      psnr(a, b, skip: border), worst * 255, 100 * Double(over) / Double(n))
    }
    for (id, hdr) in [(243, NISUpscaler.HDR.none), (244, .linear)] {
        let file = dir.appending(path: "\(id).metal")
        guard var code = try? String(contentsOf: file, encoding: .utf8) else { print("\(id): no \(file.path)"); continue }
        // dxbc-spirv gives the group size as LocalSizeId, which spirv-cross leaves undefined
        code = code.replacingOccurrences(of: "kernel void main0", with: "constant uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);\nkernel void main0")
        do {
            let lib = try device.makeLibrary(source: code, options: nil)
            var refl: MTLComputePipelineReflection?
            let ref = try device.makeComputePipelineState(function: lib.makeFunction(name: "main0")!, options: [.bindingInfo], reflection: &refl)
            var idx: [String: Int] = [:]
            for b in refl?.bindings ?? [] { idx[b.name] = b.index }
            let refOut = texture(ow, oh)
            var cfg = NISUpscaler.configBytes(sharpness: 0.5, hdr: hdr, input: (iw, ih), output: (ow, oh))
            let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(ref)
            enc.setBytes(&cfg, length: cfg.count, index: idx["cb0"] ?? 0)
            enc.setTexture(src, index: idx["t0"]!); enc.setTexture(coefS, index: idx["t1"]!); enc.setTexture(coefU, index: idx["t2"]!)
            enc.setTexture(refOut, index: idx["u0"]!); enc.setSamplerState(sampler, index: idx["s0"]!)
            enc.dispatchThreadgroups(MTLSize(width: (ow + 31) / 32, height: (oh + 23) / 24, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            if let e = cb.error { throw e }
            let reference = values(refOut)
            print("\(id) (\(hdr)), \(iw)x\(ih) -> \(ow)x\(oh), sharpness 0.5, against NVIDIA's compiled shader:")
            for (storage, half) in [(false, false), (true, false), (true, true)] {
                var o = NISUpscaler.Options(); o.halfMath = half; o.halfStorage = storage
                let ours = texture(ow, oh)
                let up = try NISUpscaler(device: device, inputSize: MTLSize(width: iw, height: ih, depth: 1),
                                         outputSize: MTLSize(width: ow, height: oh, depth: 1), sharpness: 0.5, hdr: hdr, options: o)
                let cb = queue.makeCommandBuffer()!
                up.encode(commandBuffer: cb, input: src, output: ours)
                cb.commit(); cb.waitUntilCompleted()
                if let e = cb.error { throw e }
                print("  \(storage ? "half " : "float") storage, \(half ? "half " : "float") maths\(storage && !half ? " (default)" : "          "): " + compare(values(ours), reference))
            }
        } catch { fail("\(id): \(error)") }
    }

default:
    fail(usage)
}
