# VECTORIA Reference Kernels

The reference kernels serve as the "Ground Truth" for VECTORIA's computation.

## Design Philosophy

1. **Correctness First**: Algorithms are implemented using their simplest, most readable mathematical definition (e.g., triple-loop GEMM).
2. **No Optimization**: No loop unrolling, no blocking, no SIMD intrinsics.
3. **Determinism**: By avoiding complex reduction trees or parallel accumulation, we ensure bitwise reproducible floating-point results.

## Usage

These kernels are used in two scenarios:
1. **Validation**: Optimized Assembly kernels are fuzz-tested against these reference implementations.
2. **Fallback**: If an optimized kernel is unavailable for a specific architecture or shape, the engine falls back to the reference kernel.

## Implementations

For a detailed status of which kernels are SIMD-accelerated, see the [Kernel Certification Policy](kernel_certification.md).

### GEMM (Scalar)
- **File**: `core/src/kernels/gemm_ref.cpp`
- **Algorithm**: Standard $O(M \cdot N \cdot K)$ triple loop.
- **Precision**: FP32 (accumulators match output type).

### Element-wise Ops (Scalar)
- **Add**: `core/src/kernels/add_ref.cpp` - `Out[i] = A[i] + B[i]`
- **Mul**: `core/src/kernels/mul_ref.cpp` - `Out[i] = A[i] * B[i]` (supports broadcast)
- **Sqrt**: `core/src/kernels/sqrt_ref.cpp` - `Out[i] = sqrt(A[i])`
- **Log**: `core/src/kernels/log_ref.cpp` - `Out[i] = log(A[i])`
- **ReLU**: `core/src/kernels/relu_ref.cpp` - `max(0, x)`

### Reduction (Scalar)
- **ReduceSum**: `core/src/kernels/reduce_sum_ref.cpp` - Sums along the last dimension.

### BiasAdd (Scalar)
- **File**: `core/src/kernels/bias_add_ref.cpp`
- **Algorithm**: Broadcast add. `Out[i, j] = In[i, j] + Bias[j]`.

## Composed Operations

High-level operations are implemented by expanding into subgraphs of the kernels above. See [Graph Semantics](graph_semantics.md) for details.

- **Softmax (Naïve)**: Composed of `ReduceMax`, `Sub`, `Exp`, `ReduceSum`, and `Div`. Prone to overflow; use `StableSoftmax` instead.
- **LayerNorm (Stable)**: Composed of `ReduceSum`, `Sub`, `Mul`, `Add`, `Div`, and `Sqrt`. Reference-only expansion (no fused kernel).
- **LogSoftmax (Stable)**: Composed of `ReduceMax`, `Sub`, `Exp`, `ReduceSum`, `Log`, and `Sub`. Stable expansion using max-subtraction. Reference-only.
- **StableSoftmax**: `Exp(LogSoftmax(x))`. Recommended over `Softmax` for numerical stability. Reference-only.
- **CrossEntropy (Inference-Only)**: `Sum(-Target * LogSoftmax(Logits))`. Evaluation metric. Reference-only.
- **Attention (Scaled Dot-Product)**: Semantic expansion using `MatMul`, `Transpose`, `Mul`, and `StableSoftmax`. Not a fused kernel. Reference-only.
- **MultiHeadAttention**: High-level semantic composition using projections, head-splitting (`Reshape`+`Transpose`+`Slice`), per-head `Attention`, and final projection. Not a fused kernel. Reference-only.

## Structural Operations (Reference-Only)

These operations manipulate tensor shape and layout without performing arithmetic.

- **Transpose**: Reorders axes using a permutation vector. Implemented via deterministic index mapping in the reference backend.
- **Reshape**: Reinterprets the linear memory buffer with a new shape. Reference implementation performs a strict copy to enforce ownership boundaries.
- **Concat**: Joins multiple tensors along a specified axis. Implemented as a sequential copy in the reference backend. Essential for Multi-Head Attention composition.