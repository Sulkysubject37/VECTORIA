# VECTORIA Architecture

VECTORIA is structured as a series of strict computational layers designed for determinism and high performance.

## Execution Flow

1. **Graph Construction (Python/Swift)**: Users define computation using high-level bindings.
2. **IR Freezing**: The graph is serialized into the C++ Intermediate Representation (IR).
3. **Validation**: The C++ `Engine` validates graph invariants.
4. **Memory Planning**: The `MemoryModel` pre-allocates contiguous memory.
5. **Static Scheduling**: The `Engine` produces a deterministic execution order.
6. **Kernel Dispatch**: The `Engine` dispatches kernels based on the configured **Kernel Policy**.

## Kernel Policy & Activation
VECTORIA enforces explicit kernel selection via `EngineConfig`. No implicit "auto-magic" selection is performed.

### Policies
1. **Reference (`KernelPolicy::Reference`)** [DEFAULT]
   - Uses C++ scalar implementations.
   - Guaranteed correct, portable, and bit-exact.
   - Used for debugging and validation.

2. **SIMD (`KernelPolicy::SIMD`)**
   - Uses architecture-specific Assembly (AVX2, NEON).
   - **MUST** be enabled at compile time (`-DVECTORIA_USE_ASM`).
   - **MUST** be explicitly requested in `EngineConfig`.
   - Throws a runtime error if requested but unavailable.

## Layers

- **Assembly**: SIMD-optimized kernels (GEMM, activations) for specific architectures.
- **C++ (Core)**: The backbone of the framework.
- **Python**: Frontend.
- **Swift**: Bindings.

## Determinism
By ensuring the graph is immutable and the execution schedule is derived through a fixed algorithm, VECTORIA guarantees that for a given input and set of parameters, the execution path and memory layout remain constant across runs on the same hardware.

## Capability Introspection
VECTORIA provides an explicit API to query the capabilities of the current execution environment.
- **Architecture**: Identifies the host CPU architecture (x86_64, ARM64).
- **SIMD Compilation**: Confirms if SIMD kernels were enabled at build time.
- **SIMD Host Support**: Verifies if the host CPU supports the required SIMD instructions.

This information is available in C++, Python, and Swift, allowing applications to make informed decisions about kernel policies.