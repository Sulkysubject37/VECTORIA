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
- **Lowering**: Support for exporting semantic graphs to CoreML (.mlpackage) with MIL validation.
- **Python**: Frontend.
- **Swift**: Bindings.

## Semantic Composition (High-Level Ops)
While the core engine dispatches primitive kernels (MatMul, Add, etc.), VECTORIA enables the expression of complex high-level operations through **Pure Semantic Composition**. 

These operations do not exist as monolithic kernels but are expanded into auditable subgraphs of primitives during graph construction:
*   **LayerNorm**: Composed of Reductions and element-wise math.
*   **Multi-Head Attention (MHA)**: Composed of Projections, Transpositions, and Scaled Dot-Product blocks.
*   **Transformer Encoder Block**: Composed of MHA, FFN, and Residual connections.

### Semantic Freeze
The semantic surface of these operations is **frozen**. VECTORIA guarantees that the expansion logic and numerical expectations for these blocks will remain stable. Any future optimizations (e.g., kernel fusion) must prove bitwise equivalence to these frozen reference expansions. Refer to [TRUTH.md](../TRUTH.md) for the governing principles of numerical truth in the system.

## Determinism
By ensuring the graph is immutable and the execution schedule is derived through a fixed algorithm, VECTORIA guarantees that for a given input and set of parameters, the execution path and memory layout remain constant across runs on the same hardware.

## Capability Introspection
VECTORIA provides an explicit API to query the capabilities of the current execution environment.
- **Architecture**: Identifies the host CPU architecture (x86_64, ARM64).
- **SIMD Compilation**: Confirms if SIMD kernels were enabled at build time.
- **SIMD Host Support**: Verifies if the host CPU supports the required SIMD instructions.

This information is available in C++, Python, and Swift, allowing applications to make informed decisions about kernel policies.

## Cross-Platform Support
VECTORIA is designed for portability but makes a strict distinction between **Code Portability** and **Numerical Reproducibility**.

### 1. Code Portability (Guaranteed)
- The Core Engine and Reference Kernels are written in standard C++17.
- The framework is verified to build and run on:
  - **macOS** (Clang, Apple Silicon)
  - **Linux** (GCC, x86_64)
- Python and Swift bindings are portable across their respective supported OS environments.

### 2. Numerical Reproducibility (Conditional)
- **Within the same platform**: Bitwise identical results are guaranteed.
- **Between different platforms (Reference Path)**: Guaranteed identical results (IEEE 754 compliance).
- **Between different platforms (SIMD Path)**: **NOT guaranteed**. Floating-point associativity differences and hardware-specific FMA (Fused Multiply-Add) behavior may cause tiny bitwise drifts between ARM64 and x86_64 implementations. 
- **Policy**: If cross-platform bitwise identity is required, users MUST select the `Reference` kernel policy.