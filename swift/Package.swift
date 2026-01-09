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
    ],
    targets: [
        .target(
            name: "Vectoria",
            dependencies: []),
        .testTarget(
            name: "VectoriaTests",
            dependencies: ["Vectoria"]),
    ]
)
