# VECTORIA Swift Wrapper

High-level, typesafe Swift interface for VECTORIA.

## Integration
Add the `swift/` directory as a Local Swift Package dependency to your Xcode project or `Package.swift`.

## Usage
```swift
import Vectoria

let runtime = try VectoriaRuntime(libraryPath: "libvectoria.dylib")
let graph = runtime.createGraph()

let x = graph.addInput(name: "X", shape: [3], dtype: .float32)
let sm = graph.addSoftmax(input: x)
graph.setOutput(nodeId: sm)

let engine = runtime.createEngine(graph: graph, policy: .simd)
engine.compile()

// ... Set buffers and execute ...
engine.execute()
```

## Deployment
Use the `exportToCoreML` method to generate a production-ready model:
```swift
try graph.exportToCoreML(path: "MyModel.mlpackage")
```

## Requirements
- macOS 11+ / iOS 14+
- Swift 5.5+
- `libvectoria.dylib` (built for `arm64`)
