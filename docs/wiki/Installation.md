# Installation

VECTORIA is distributed as a Python package, a Swift package, and prebuilt shared libraries.

## Python

The easiest way to install VECTORIA is via `pip`:

```bash
pip install vectoria
```

This installs the `vectoria` core and the `vectoria-trace` command-line utility.

## Swift (SPM)

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/Sulkysubject37/VECTORIA.git", from: "1.3.1")
]
```

## Manual Build

To build from source for a specific architecture:

```bash
g++ -std=c++17 -shared -fPIC -Icore/include 
    core/src/*.cpp core/src/kernels/*.cpp core/src/graph/*.cpp core/src/lowering/*.cpp 
    -o libvectoria.dylib
```
