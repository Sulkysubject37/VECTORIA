# Swift Parity Plan

**Status**: Minimal Execution Parity Reached

To make VECTORIA a viable backend for Apple platforms, the Swift binding now supports native execution through the C API.

## Implemented Goals
1. **Graph Construction**: Basic support for Inputs and MatMul.
2. **Native Execution**: Swift can now drive the C++ engine via `dlopen`/`dlsym`.
3. **Trace Exposure**: Swift `struct TraceEvent` mirrors C++ and events are retrievable.
4. **Execution Control**: `KernelPolicy` selection is exposed to Swift.

## Traceability
Swift accesses traces via the C-API wrapper:
- `vectoria_engine_get_trace_size`
- `vectoria_engine_get_trace_event`

## Example Usage
```swift
let runtime = try VectoriaRuntime()
let graph = runtime.createGraph()
let x = graph.addInput(name: "X", shape: [1, 4], dtype: .float32)
// ... construct more ...
let engine = runtime.createEngine(graph: graph, policy: .simd)
engine.compile()
engine.execute()
let trace = engine.getTrace()
```

## Anti-Goals
- **Swift-side Scheduling**: Scheduling remains strictly in C++.
- **CoreML Export**: Next Phase.