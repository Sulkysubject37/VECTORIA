#include "vectoria/kernels.hpp"

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus bias_add_f32(
    const float* input,
    const float* bias,
    float* output,
    size_t m, size_t n
) {
    if (!input || !bias || !output) return VECTORIA_ERROR_INVALID_SHAPE;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            output[i * n + j] = input[i * n + j] + bias[j];
        }
    }
    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
