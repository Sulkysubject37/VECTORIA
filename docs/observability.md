# VECTORIA Observability

VECTORIA provides built-in tracing to inspect execution behavior.

## Purpose
- **Debugging**: Verify execution order and memory allocation.
- **Auditing**: Confirm which kernels (Reference vs SIMD) were actually executed.
- **Optimization**: Analyze timestamps (though this is not a full profiler).
- **Provenance**: Traces provide evidence of exactly how a result was calculated.

## Trace Events

| Event Type | Description | Details Field |
|------------|-------------|---------------|
| `GraphCompilation` | Engine compilation phase | "Start \| Mode: [Research/Deployment]" / "End" |
| `MemoryAllocation` | Buffer allocation for a node | Size in bytes |
| `NodeExecutionStart` | Execution begins for a node | - |
| `KernelDispatch` | Kernel selection & deps | "Reference" or "SIMD [Arch]" | Inputs: [id, id] |
| `NodeExecutionEnd` | Execution finishes | - |

## Scientific Provenance
Traces are a critical part of the output. If a trace does not explicitly state `SIMD [Arch]`, then the SIMD kernel was **NOT** used.
This guarantees that you can prove which code executed for a given result.

## Python API
Traces are accessible via `Runtime.get_trace()`:

```python
trace = runtime.get_trace()
for event in trace:
    print(event)
```
