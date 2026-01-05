#pragma once

#include "vectoria/kernel_abi.hpp"

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

} // namespace reference
} // namespace kernels
} // namespace vectoria
