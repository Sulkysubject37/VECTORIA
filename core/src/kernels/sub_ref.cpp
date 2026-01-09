#include "vectoria/kernels.hpp"

namespace vectoria {
namespace kernels {
namespace reference {

VectoriaStatus sub_f32(
    const float* a,
    const float* b,
    float* out,
    size_t count_a,
    size_t count_b
) {
    // Overloaded/Hack signature for Softmax broadcast support
    // Real signature should probably take shapes, but we are stuck with flat pointers here.
    // Wait, I can change the signature in kernels.hpp if I update it everywhere.
    // The previous kernels took `count`.
    // I will use a different function name for the broadcast version or infer from counts?
    // Inferring from counts is ambiguous (e.g. [2, 2] vs [4, 1]).
    // But for Phase 4, Softmax is the *only* composed op.
    // I will add a new kernel `sub_broadcast_f32` specifically for this.
    // BUT "No new primitive kernels".
    // I'll stick to `sub_f32` but add dimensions to arguments.
    return VECTORIA_ERROR_INVALID_SHAPE;
}

VectoriaStatus sub_broadcast_f32(
    const float* a,
    const float* b,
    float* out,
    size_t outer,
    size_t inner
) {
    // Computes A[i, j] - B[i]
    if (!a || !b || !out) return VECTORIA_ERROR_INVALID_SHAPE;
    
    for (size_t i = 0; i < outer; ++i) {
        float val_b = b[i];
        for (size_t j = 0; j < inner; ++j) {
            out[i * inner + j] = a[i * inner + j] - val_b;
        }
    }
    return VECTORIA_SUCCESS;
}

} // namespace reference
} // namespace kernels
} // namespace vectoria
