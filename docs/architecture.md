# VECTORIA Architecture

VECTORIA is structured as a series of strict computational layers designed for determinism and high performance.

## Execution Flow

1. **Graph Construction (Python/Swift)**: Users define computation using high-level bindings.
2. **IR Freezing**: The graph is serialized into the C++ Intermediate Representation (IR).
3. **Validation**: The C++ `Engine` validates graph invariants (no cycles, shape consistency).
4. **Memory Planning**: The `MemoryModel` uses the static IR to pre-allocate an `Arena`.
5. **Static Scheduling**: The `Engine` produces a deterministic execution order (topological sort).
6. **Kernel Dispatch**: The `Engine` traverses the schedule and calls optimized Assembly kernels via the Kernel ABI.

## Layers

- **Assembly**: SIMD-optimized kernels (GEMM, activations) for specific architectures (ARM64 Neon, x86 AVX).
- **C++ (Core)**: The backbone of the framework. Manages IR, Memory, and the Execution Engine.
- **Python**: Frontend for graph construction, inspection, and visualization.
- **Swift**: Safe, high-level bindings for integration into Apple ecosystem and CoreML export.

## Kernel ABI Contract (C++ ↔ Assembly)

To ensure portability and safety, all Assembly kernels must adhere to the following rules:

### Function Signatures
All kernels are exposed via `extern "C"` with fixed signatures defined in `core/include/vectoria/kernel_abi.hpp`.

### Alignment Rules
- **Memory Alignment**: All input and output buffers MUST be aligned to at least **64-byte boundaries** to support the widest SIMD instructions (AVX-512).
- **Leading Dimensions**: LDA, LDB, and LDC should ideally be multiples of the SIMD width for optimal performance.

### Error Handling
- Kernels return a `VectoriaStatus` integer.
- `0` (VECTORIA_SUCCESS) indicates success.
- Negative values indicate specific failure modes.
- Kernels must NOT throw exceptions or use signal handlers.

### Forbidden Actions
Assembly kernels are strictly prohibited from:
1. **Memory Allocation**: Kernels must only operate on buffers provided by the C++ engine.
2. **System Calls**: No I/O, file access, or network calls.
3. **Global State**: Kernels must be pure functions with no side effects on global or static variables.
4. **Threading**: Kernels must be single-threaded. Parallelism is managed by the C++ engine.

## Determinism
By ensuring the graph is immutable and the execution schedule is derived through a fixed algorithm, VECTORIA guarantees that for a given input and set of parameters, the execution path and memory layout remain constant across runs on the same hardware.
