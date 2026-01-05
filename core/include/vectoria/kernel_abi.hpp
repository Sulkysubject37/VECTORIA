#pragma once

#include <cstddef>
#include <cstdint>

/**
 * VECTORIA Kernel ABI Contract
 * 
 * This file defines the exact binary interface between C++ and Assembly kernels.
 * All Assembly kernels must adhere to these signatures and conventions.
 */

extern "C" {

/**
 * Standard error codes for kernels.
 */
enum VectoriaStatus : int32_t {
    VECTORIA_SUCCESS = 0,
    VECTORIA_ERROR_INVALID_ALIGNMENT = -1,
    VECTORIA_ERROR_INVALID_SHAPE = -2,
    VECTORIA_ERROR_UNSUPPORTED_DTYPE = -3
};

/**
 * General GEMM Signature: C = alpha * (A * B) + beta * C
 */
typedef VectoriaStatus (*gemm_f32_t)(
    const float* a, 
    const float* b, 
    float* c,
    size_t m, size_t n, size_t k,
    size_t lda, size_t ldb, size_t ldc,
    float alpha, float beta
);

/**
 * Unary Operation Signature: Out = op(In)
 */
typedef VectoriaStatus (*unary_f32_t)(
    const float* in,
    float* out,
    size_t count
);

/**
 * Binary Operation Signature: Out = op(InA, InB)
 */
typedef VectoriaStatus (*binary_f32_t)(
    const float* a,
    const float* b,
    float* out,
    size_t count
);

} // extern "C"
