#include "vectoria/kernels.hpp"

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus gemm_f32(
    const float* a, 
    const float* b, 
    float* c,
    size_t m, size_t n, size_t k,
    size_t lda, size_t ldb, size_t ldc,
    float alpha, float beta
) {
    if (!a || !b || !c) {
        return VECTORIA_ERROR_INVALID_SHAPE;
    }

    // Naive triple loop implementation
    // No blocking, no vectorization, no multi-threading.
    // Purely for correctness validation.
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                // Row-major indexing: index = row * stride + col
                float val_a = a[i * lda + p];
                float val_b = b[p * ldb + j];
                sum += val_a * val_b;
            }
            
            // C = alpha * sum + beta * C
            size_t c_idx = i * ldc + j;
            c[c_idx] = alpha * sum + beta * c[c_idx];
        }
    }

    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
