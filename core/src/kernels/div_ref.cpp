#include "vectoria/kernels.hpp"

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus div_f32(
    const float* a,
    const float* b,
    float* out,
    size_t count_a,
    size_t count_b
) {
    if (!a || !b || !out) return VECTORIA_ERROR_INVALID_SHAPE;

    if (count_a == count_b) {
        for (size_t i = 0; i < count_a; ++i) {
            out[i] = a[i] / b[i];
        }
        return VECTORIA_SUCCESS;
    }
    return VECTORIA_ERROR_INVALID_SHAPE;
}

VectoriaStatus div_broadcast_f32(
    const float* a,
    const float* b,
    float* out,
    size_t outer,
    size_t inner
) {
    // Computes A[i, j] / B[i]
    if (!a || !b || !out) return VECTORIA_ERROR_INVALID_SHAPE;
    
    for (size_t i = 0; i < outer; ++i) {
        float val_b = b[i];
        for (size_t j = 0; j < inner; ++j) {
            out[i * inner + j] = a[i * inner + j] / val_b;
        }
    }
    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
