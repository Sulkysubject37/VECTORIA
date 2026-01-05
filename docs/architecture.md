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

## Determinism
By ensuring the graph is immutable and the execution schedule is derived through a fixed algorithm, VECTORIA guarantees that for a given input and set of parameters, the execution path and memory layout remain constant across runs on the same hardware.