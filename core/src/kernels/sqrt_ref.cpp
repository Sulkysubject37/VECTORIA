#include "vectoria/kernels.hpp"
#include <cmath>

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus sqrt_f32(const float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::sqrt(input[i]);
    }
    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
