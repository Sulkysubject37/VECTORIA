#include "vectoria/kernels.hpp"
#include <cstring>
#include <numeric>

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus concat_f32(
    const std::vector<const float*>& inputs,
    float* output,
    const std::vector<std::vector<int64_t>>& input_shapes,
    int64_t axis
) {
    if (inputs.empty() || !output) return VECTORIA_ERROR_INVALID_SHAPE;

    size_t rank = input_shapes[0].size();
    size_t outer_count = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        outer_count *= input_shapes[0][i];
    }

    size_t inner_count = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < rank; ++i) {
        inner_count *= input_shapes[0][i];
    }

    float* dst = output;
    for (size_t o = 0; o < outer_count; ++o) {
        for (size_t k = 0; k < inputs.size(); ++k) {
            size_t concat_dim_size = input_shapes[k][axis];
            size_t copy_size = concat_dim_size * inner_count;
            
            // offset in input k for current outer block o
            const float* src = inputs[k] + o * copy_size;
            
            std::memcpy(dst, src, copy_size * sizeof(float));
            dst += copy_size;
        }
    }

    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
