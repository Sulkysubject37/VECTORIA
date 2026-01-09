#include "vectoria/kernels.hpp"
#include "vectoria/kernel_abi.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>

// Forward declare
extern "C" {
    VectoriaStatus reduce_sum_f32_neon(const float* in, float* out, size_t outer, size_t inner);
    VectoriaStatus reduce_max_f32_neon(const float* in, float* out, size_t outer, size_t inner);
    VectoriaStatus reduce_sum_f32_avx2(const float* in, float* out, size_t outer, size_t inner);
    VectoriaStatus reduce_max_f32_avx2(const float* in, float* out, size_t outer, size_t inner);
}

void verify(const std::vector<float>& ref, const std::vector<float>& simd, const char* name) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::abs(ref[i] - simd[i]);
        if (diff > max_diff) max_diff = diff;
        
        // Relaxed tolerance for reductions due to associativity
        if (diff > 1e-3f) {
            std::cerr << name << " Mismatch at " << i << ": Ref=" << ref[i] << " SIMD=" << simd[i] 
                      << " Diff=" << diff << std::endl;
            exit(1);
        }
    }
    std::cout << name << " PASSED (Max Drift: " << std::scientific << max_diff << ")" << std::endl;
}

int main() {
    std::cout << "Validating SIMD Reduction Kernels..." << std::endl;
    
    size_t outer = 10;
    size_t inner = 1024 + 7;
    std::vector<float> input(outer * inner);
    std::vector<float> ref(outer), simd(outer);
    
    // Init
    for(size_t i=0; i<input.size(); ++i) {
        input[i] = (float)((i % 100) + 1) * 0.1f;
    }
    
#if defined(__aarch64__)
    // ReduceSum
    vectoria::kernels::reference::reduce_sum_f32(input.data(), ref.data(), outer, inner);
    printf("Calling NEON: in=%p out=%p outer=%zu inner=%zu\n", input.data(), simd.data(), outer, inner);
    reduce_sum_f32_neon(input.data(), simd.data(), outer, inner);
    verify(ref, simd, "ReduceSum [NEON]");
    
    // ReduceMax
    vectoria::kernels::reference::reduce_max_f32(input.data(), ref.data(), outer, inner);
    reduce_max_f32_neon(input.data(), simd.data(), outer, inner);
    verify(ref, simd, "ReduceMax [NEON]");

#elif defined(__x86_64__)
    // ReduceSum
    vectoria::kernels::reference::reduce_sum_f32(input.data(), ref.data(), outer, inner);
    reduce_sum_f32_avx2(input.data(), simd.data(), outer, inner);
    verify(ref, simd, "ReduceSum [AVX2]");
    
    // ReduceMax
    vectoria::kernels::reference::reduce_max_f32(input.data(), ref.data(), outer, inner);
    reduce_max_f32_avx2(input.data(), simd.data(), outer, inner);
    verify(ref, simd, "ReduceMax [AVX2]");
#else
    std::cout << "Skipping reduction validation on unknown host." << std::endl;
#endif

    return 0;
}
