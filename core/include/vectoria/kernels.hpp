#pragma once

#include "vectoria/kernel_abi.hpp"
#include <vector>

namespace vectoria {
namespace kernels {
namespace reference {

/**
 * Scalar reference implementation of GEMM.
 * C = alpha * (A * B) + beta * C
 * 
 * Layout assumption: Row-Major
 * 
 * @param a Input matrix A (MxK)
 * @param b Input matrix B (KxN)
 * @param c Output matrix C (MxN)
 * @param m Rows of A and C
 * @param n Cols of B and C
 * @param k Cols of A and Rows of B
 * @param lda Leading dimension (stride) of A
 * @param ldb Leading dimension (stride) of B
 * @param ldc Leading dimension (stride) of C
 * @param alpha Scalar multiplier for (A*B)
 * @param beta Scalar multiplier for C
 * @return VectoriaStatus
 */
VectoriaStatus gemm_f32(
    const float* a, 
    const float* b, 
    float* c,
    size_t m, size_t n, size_t k,
    size_t lda, size_t ldb, size_t ldc,
    float alpha, float beta
);

/**
 * Bias Add: Out = In + Bias (Broadcast)
 * In: [M, N]
 * Bias: [1, N]
 * Out: [M, N]
 */
VectoriaStatus bias_add_f32(
    const float* input,
    const float* bias,
    float* output,
    size_t m, size_t n
);

/**
 * ReLU: Out = max(0, In)
 * Element-wise.
 */
VectoriaStatus relu_f32(
    const float* input,
    float* output,
    size_t count
);

/**
 * Element-wise Add: Out = A + B
 */
VectoriaStatus add_f32(
    const float* a,
    const float* b,
    float* out,
    size_t count
);

/**
 * Add with Broadcast (Col Vector): Out[i, j] = A[i, j] + B[i]
 * Or if inner=count_a (outer=1), Out[j] = A[j] + B[0] (Scalar Add)
 */
VectoriaStatus add_broadcast_f32(
    const float* a,
    const float* b,
    float* out,
    size_t outer,
    size_t inner
);

/**
 * Element-wise Mul: Out = A * B
 */
VectoriaStatus mul_f32(
    const float* a,
    const float* b,
    float* out,
    size_t count
);

/**
 * Mul with Broadcast (Row Vector): Out[i, j] = A[i, j] * B[j]
 * A: [M, N]
 * B: [1, N]
 */
VectoriaStatus mul_broadcast_f32(
    const float* a,
    const float* b,
    float* out,
    size_t m, size_t n
);

/**
 * Reduce Sum (Last Axis): Out[i] = sum(In[i, :])
 * In: [Outer, Inner]
 * Out: [Outer]
 */
VectoriaStatus reduce_sum_f32(
    const float* input,
    float* output,
    size_t outer_dim,
    size_t inner_dim
);

/**
 * Reduce Max (Last Axis): Out[i] = max(In[i, :])
 * In: [Outer, Inner]
 * Out: [Outer]
 */
VectoriaStatus reduce_max_f32(
    const float* input,
    float* output,
    size_t outer_dim,
    size_t inner_dim
);

/**
 * Element-wise Exp: Out = exp(A)
 */
VectoriaStatus exp_f32(
    const float* input,
    float* output,
    size_t count
);

/**
 * Element-wise Sub: Out = A - B
 */
VectoriaStatus sub_f32(
    const float* a,
    const float* b,
    float* out,
    size_t count_a,
    size_t count_b
);

/**
 * Subtract with Broadcast (Col Vector): Out[i, j] = A[i, j] - B[i]
 */
VectoriaStatus sub_broadcast_f32(
    const float* a,
    const float* b,
    float* out,
    size_t outer,
    size_t inner
);

/**
 * Element-wise Div: Out = A / B
 */
VectoriaStatus div_f32(
    const float* a,
    const float* b,
    float* out,
    size_t count_a,
    size_t count_b
);

/**
 * Divide with Broadcast (Col Vector): Out[i, j] = A[i, j] / B[i]
 */
VectoriaStatus div_broadcast_f32(
    const float* a,
    const float* b,
    float* out,
    size_t outer,
    size_t inner
);

/**
 * Element-wise Sqrt: Out = sqrt(A)
 */
VectoriaStatus sqrt_f32(
    const float* input,
    float* output,
    size_t count
);

/**
 * Element-wise Log: Out = log(A)
 */
VectoriaStatus log_f32(
    const float* input,
    float* output,
    size_t count
);

/**
 * Transpose (Reference): Out[new_indices] = In[old_indices]
 */
VectoriaStatus transpose_f32(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& perm
);

} // namespace reference
} // namespace kernels
} // namespace vectoria
