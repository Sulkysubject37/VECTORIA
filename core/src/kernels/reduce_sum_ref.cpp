#include "vectoria/kernels.hpp"

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus reduce_sum_f32(
    const float* input,
    float* output,
    size_t outer_dim,
    size_t inner_dim
) {
    if (!input || !output) return VECTORIA_ERROR_INVALID_SHAPE;

    for (size_t i = 0; i < outer_dim; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < inner_dim; ++j) {
            sum += input[i * inner_dim + j];
        }
        output[i] = sum;
    }
    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
