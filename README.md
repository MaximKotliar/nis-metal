# NISMetal

[NVIDIA Image Scaling](https://github.com/NVIDIAGameWorks/NVIDIAImageScaling) (NIS) 1.0.3 on Metal, for developers:
a spatial upscaler with edge-adaptive sharpening, as a Swift library and a command-line tool. It upscales a finished
frame on its own (no depth, motion or history), so it works for any game or video.

Ported by hand from NVIDIA's MIT-licensed `NIS_Scaler.h` and `NIS_Config.h`. The algorithm is NVIDIA's step for step:
the brightness of the source block plus a 3-pixel border goes into threadgroup memory, a 4-direction edge map is made
from it, and each output pixel blends a 6×6 polyphase scaler (64 phases) with directional filters by the edge map,
each with unsharp masking; the change in brightness is added to a bilinear sample of the colour.

Part of a set with [lss-metal](https://github.com/MaximKotliar/lss-metal) (Lossless Scaling's spatial upscalers,
which offers NIS as one of its modes), [lsfg-metal](https://github.com/MaximKotliar/lsfg-metal) (frame generation),
[nss-metal](https://github.com/MaximKotliar/nss-metal) (Arm's neural temporal upscaler) and the app that uses them,
[Lightscale](https://github.com/MaximKotliar/lightscale).

## Speed

M4 Max, warm GPU, median per upscale:

| Input → output | GPU time |
|---|---|
| 1280×720 → 2560×1440 | 0.18 ms |
| 1728×1117 → 3456×2234 | 0.31 ms |

For Apple GPUs:

- **NVIDIA's fp16 build:** the brightness tile, edge map and coefficients are half precision in threadgroup memory,
  the maths is float. Half the threadgroup memory of the float build means twice as many blocks at once: 0.31
  against 0.52 ms at 1728×1117 → 3456×2234. Half-precision maths saves another ~10% but flips more edge decisions, so
  it's off by default.
- **Gathers:** the source is read with three texture gathers per 2×2 pixels instead of four filtered samples.
- **No coefficient textures:** NVIDIA's two 64-phase tables are compiled into the shader.
- **Compile-time options:** block size, precision and HDR mode (NVIDIA's linear and PQ modes), compiled once per
  device and shared.

NVIDIA rates NIS for 1×–2×; larger steps work too, downscaling doesn't.

## Use it

```swift
.package(url: "https://github.com/MaximKotliar/nis-metal.git", branch: "main")
```

```swift
import NISMetal

let nis = try NISUpscaler(device: device, inputSize: MTLSize(width: 1280, height: 720, depth: 1),
                          outputSize: MTLSize(width: 2560, height: 1440, depth: 1),
                          sharpness: 0.5)            // 0 (none) … 1; 0.5 is NVIDIA's default
nis.encode(commandBuffer: cb, input: frame, output: drawable.texture)   // the output needs shader-write usage
```

`encode(encoder:…)` adds it to a compute encoder you already have; `inputOrigin` and `outputOrigin` upscale a region
into a region (letterboxing). `hdr: .linear` or `.pq` selects NVIDIA's HDR tuning.

## Try it

```bash
swift run -c release nis-metal upscale frame.png -o big.png --sharpness 0.5 --scale 2
swift run -c release nis-metal bench frame.png        # every block size and precision: time, and PSNR vs float
```

## Accuracy

`nis-metal check DIR frame.png` compares it with NVIDIA's own compiled shader, translated to Metal: `DIR/243.metal`
(SDR) and `DIR/244.metal` (linear HDR). Lossless Scaling ships that shader in its `Lossless.dll` (DXBC shaders
243–246, stock NVIDIA); dump them with [lsfg-metal](https://github.com/MaximKotliar/lsfg-metal)'s `dump` command and
translate with [dxbc-spirv](https://github.com/doitsujin/dxbc-spirv) and `spirv-cross --msl`. Keep those files out of
any repository.

| Build | SDR | Linear HDR |
|---|---|---|
| float storage and maths (NVIDIA's default build) | 71.7 dB, ≤ 0.2/255 | 71.7 dB, ≤ 0.5/255 |
| half storage, float maths (default here; NVIDIA's fp16 build) | 63.8 dB; 0.14% of values > 1/255 | 56.6 dB; 0.21% of values > 1/255, rare outliers |
| half storage and maths | 59.5 dB | 52.5 dB |

(1728×1117 from 864×558, sharpness 0.5, interior. Within 4 pixels of the border the translated reference itself is
unreliable: NVIDIA's bounds check turns the first column's source x, −0.25, into an unsigned integer, which Direct3D
clamps to 0 and Metal doesn't.) Half storage flips a few edge decisions: in linear HDR, where brightness is compressed
before it's stored, a flip can move a single value a lot, so `Options` with `halfStorage = false` is the accurate choice
there, at the float build's cost.

## License

MIT (see [LICENSE](LICENSE)). The shader is ported from NVIDIA Image Scaling, MIT, Copyright (c) 2022 NVIDIA
CORPORATION & AFFILIATES; its notice is in [LICENSE](LICENSE) and at the top of the shader source.
