// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "NISMetal",
    platforms: [.macOS(.v15)],
    products: [
        // NVIDIA Image Scaling on Metal (builds for arm64 and x86_64).
        .library(name: "NISMetal", targets: ["NISMetal"]),
        // Upscale a PNG, time the variants, check against NVIDIA's compiled shader.
        .executable(name: "nis-metal", targets: ["nis-metal"]),
    ],
    targets: [
        .target(name: "NISMetal", linkerSettings: [.linkedFramework("Metal")]),
        .executableTarget(name: "nis-metal", dependencies: ["NISMetal"], swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
