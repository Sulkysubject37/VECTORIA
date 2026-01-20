#include "vectoria/kernels.hpp"

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus add_f32(
    const float* a,
    const float* b,
    float* out,
    size_t count
) {
    if (!a || !b || !out) return VECTORIA_ERROR_INVALID_SHAPE;

    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
    return VECTORIA_SUCCESS;
}

VectoriaStatus add_broadcast_f32(
    const float* a,
    const float* b,
    float* out,
    size_t outer,
    size_t inner
) {
    if (!a || !b || !out) return VECTORIA_ERROR_INVALID_SHAPE;

    for (size_t i = 0; i < outer; ++i) {
        for (size_t j = 0; j < inner; ++j) {
            out[i * inner + j] = a[i * inner + j] + b[i];
        }
    }
    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
