# Swift Parity

The Swift bindings provide a high-level, typesafe wrapper around the Core C++ engine.

## Supported Operations
- `MatMul` (Matrix Multiplication)
- `BiasAdd` (Broadcast Addition)
- `ReLU` (Activation)

## Integration
The Swift package relies on `libvectoria.dylib` being available at runtime.

```swift
let runtime = try VectoriaRuntime(libraryPath: "path/to/libvectoria.dylib")
let graph = runtime.createGraph()
// ... build graph ...
```