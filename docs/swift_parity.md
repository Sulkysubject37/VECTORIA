# Swift Parity Plan

**Status**: Planning

To make VECTORIA a viable backend for Apple platforms, the Swift binding must reach parity with the Python frontend.

## Goals
1. **Full Graph Construction**: Support all IR nodes.
2. **Trace Exposure**: Swift `struct TraceEvent` mirroring C++.
3. **Execution Control**: `EngineConfig` exposed to Swift.

## Traceability
Swift will access traces via a C-API wrapper similar to Python.
- `vectoria_engine_get_trace_size`
- `vectoria_engine_get_trace_event`

## Kernel Selection
Swift will NOT have direct access to kernel pointers. It will control selection via:
```swift
public enum KernelPolicy {
    case reference
    case simd
}
```

## Anti-Goals
- **Swift-side Execution**: Swift will NEVER execute kernels directly. It only drives the C++ engine.
- **CoreML Export**: Deferred to Phase 5.
