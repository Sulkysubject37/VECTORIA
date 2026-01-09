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
- **Mul**: `core/src/kernels/mul_ref.cpp` - `Out[i] = A[i] * B[i]`
- **ReLU**: `core/src/kernels/relu_ref.cpp` - `max(0, x)`

### Reduction (Scalar)
- **ReduceSum**: `core/src/kernels/reduce_sum_ref.cpp` - Sums along the last dimension.

### BiasAdd (Scalar)
- **File**: `core/src/kernels/bias_add_ref.cpp`
- **Algorithm**: Broadcast add. `Out[i, j] = In[i, j] + Bias[j]`.