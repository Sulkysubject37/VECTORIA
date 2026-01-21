#include "vectoria/kernels.hpp"
#include <cstring>

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus slice_f32(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    int64_t axis,
    int64_t start,
    int64_t end
) {
    if (!input || !output) return VECTORIA_ERROR_INVALID_SHAPE;

    size_t rank = input_shape.size();
    size_t outer_count = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        outer_count *= input_shape[i];
    }

    size_t inner_count = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < rank; ++i) {
        inner_count *= input_shape[i];
    }

    size_t slice_dim_size = end - start;
    size_t input_dim_size = input_shape[axis];

    for (size_t o = 0; o < outer_count; ++o) {
        const float* src = input + (o * input_dim_size + start) * inner_count;
        float* dst = output + (o * slice_dim_size) * inner_count;
        std::memcpy(dst, src, slice_dim_size * inner_count * sizeof(float));
    }

    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
