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

} // namespace reference
} // namespace kernels
} // namespace vectoria
