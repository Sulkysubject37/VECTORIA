#include "vectoria/kernels.hpp"
#include <vector>
#include <numeric>

namespace vectoria {
namespace kernels {
namespace reference {

// Helper to convert flat index to multi-dim index
static void unravel_index(size_t index, const std::vector<int64_t>& shape, std::vector<size_t>& multi_indices) {
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        multi_indices[i] = index % shape[i];
        index /= shape[i];
    }
}

// Helper to convert multi-dim index to flat index
static size_t ravel_index(const std::vector<size_t>& multi_indices, const std::vector<int64_t>& shape) {
    size_t index = 0;
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        index += multi_indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}

VectoriaStatus transpose_f32(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& perm
) {
    if (!input || !output) return VECTORIA_ERROR_INVALID_SHAPE;

    size_t count = 1;
    for (auto d : input_shape) count *= d;

    // Output shape calculation
    std::vector<int64_t> output_shape(input_shape.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        output_shape[i] = input_shape[perm[i]];
    }

    std::vector<size_t> in_indices(input_shape.size());
    std::vector<size_t> out_indices(input_shape.size());

    // Iterate over input in linear order
    for (size_t i = 0; i < count; ++i) {
        unravel_index(i, input_shape, in_indices);
        
        // Map to output indices based on permutation
        for (size_t d = 0; d < perm.size(); ++d) {
            out_indices[d] = in_indices[perm[d]];
        }
        
        size_t out_idx = ravel_index(out_indices, output_shape);
        output[out_idx] = input[i];
    }

    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
