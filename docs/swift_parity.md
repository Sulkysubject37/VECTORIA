# Swift Parity

The Swift bindings provide a high-level, typesafe wrapper around the Core C++ engine.

## Supported Operations
- **Numerical**: `MatMul`, `Add`, `Mul`, `Div`, `ReLU`, `Exp`, `Log`, `Sqrt`, `ReduceSum`, `ReduceMax`.
- **Structural**: `Transpose`, `Reshape`, `Concat`.
- **Composed**: `LayerNorm`, `LogSoftmax`, `StableSoftmax`, `CrossEntropy`, `Attention`, `MultiHeadAttention`, `TransformerEncoder`.

## Integration
The Swift package relies on `libvectoria.dylib` being available at runtime.

```swift
let runtime = try VectoriaRuntime(libraryPath: "path/to/libvectoria.dylib")
let graph = runtime.createGraph()
// ... build graph ...
```