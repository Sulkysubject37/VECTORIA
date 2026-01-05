#include "vectoria/kernels.hpp"
#include <algorithm>

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus relu_f32(
    const float* input,
    float* output,
    size_t count
) {
    if (!input || !output) return VECTORIA_ERROR_INVALID_SHAPE;

    for (size_t i = 0; i < count; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
