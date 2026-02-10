// swift-tools-version: 5.5
import PackageDescription

let package = Package(
    name: "Vectoria",
    platforms: [
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "Vectoria",
            targets: ["Vectoria"]),
        .executable(
            name: "VectoriaExample",
            targets: ["VectoriaExample"])
    ],
    targets: [
        .target(
            name: "Vectoria",
            dependencies: []),
        .executableTarget(
            name: "VectoriaExample",
            dependencies: ["Vectoria"],
            path: "Sources/VectoriaExample"),
        .testTarget(
            name: "VectoriaTests",
            dependencies: ["Vectoria"]),
    ]
)