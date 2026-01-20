#include "vectoria/kernels.hpp"

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus mul_f32(
    const float* a,
    const float* b,
    float* out,
    size_t count
) {
    if (!a || !b || !out) return VECTORIA_ERROR_INVALID_SHAPE;

    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
    return VECTORIA_SUCCESS;
}

VectoriaStatus mul_broadcast_f32(
    const float* a,
    const float* b,
    float* out,
    size_t m,
    size_t n
) {
    if (!a || !b || !out) return VECTORIA_ERROR_INVALID_SHAPE;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            out[i * n + j] = a[i * n + j] * b[j];
        }
    }
    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
