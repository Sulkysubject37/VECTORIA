#include "vectoria/kernels.hpp"
#include <algorithm>
#include <limits>

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus reduce_max_f32(
    const float* input,
    float* output,
    size_t outer_dim,
    size_t inner_dim
) {
    if (!input || !output) return VECTORIA_ERROR_INVALID_SHAPE;

    for (size_t i = 0; i < outer_dim; ++i) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < inner_dim; ++j) {
            float val = input[i * inner_dim + j];
            if (val > max_val) {
                max_val = val;
            }
        }
        output[i] = max_val;
    }
    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
