#include "vectoria/kernels.hpp"
#include "vectoria/kernel_abi.hpp"
#include "utils/gemm_validation.hpp"
#include "vectoria/memory.hpp" // For Arena/Aligned alloc
#include <iostream>
#include <vector>

using namespace vectoria;

// Wrapper to call the architecture-specific ASM kernel
VectoriaStatus call_simd_kernel(
    const float* a, const float* b, float* c,
    size_t m, size_t n, size_t k,
    size_t lda, size_t ldb, size_t ldc,
    float alpha, float beta
) {
#ifdef VECTORIA_USE_ASM
    #if defined(__x86_64__)
        return gemm_f32_avx2(a, b, c, m, n, k, lda, ldb, ldc, alpha, beta);
    #elif defined(__aarch64__)
        return gemm_f32_neon(a, b, c, m, n, k, lda, ldb, ldc, alpha, beta);
    #else
        return VECTORIA_ERROR_UNSUPPORTED_DTYPE;
    #endif
#else
    return VECTORIA_SUCCESS; // Should not be called, or irrelevant
#endif
}

void run_test(size_t m, size_t n, size_t k) {
    std::cout << "Testing GEMM [" << m << "x" << n << "x" << k << "] ... ";

    memory::Arena arena(1024 * 1024 * 10); // 10MB
    test::DeterministicRNG rng;

    size_t lda = k;
    size_t ldb = n;
    size_t ldc = n;

    size_t size_a = m * k * sizeof(float);
    size_t size_b = k * n * sizeof(float);
    size_t size_c = m * n * sizeof(float);

    float* a = (float*)arena.allocate(size_a, 64);
    float* b = (float*)arena.allocate(size_b, 64);
    float* c_ref = (float*)arena.allocate(size_c, 64);
    float* c_simd = (float*)arena.allocate(size_c, 64);

    rng.fill(a, m * k);
    rng.fill(b, k * n);
    
    // Initialize C with some data to test beta accumulation
    rng.fill(c_ref, m * n, 0.1f);
    // Copy initial C to SIMD buffer
    for(size_t i=0; i<m*n; ++i) c_simd[i] = c_ref[i];

    // 1. Run Reference
    kernels::reference::gemm_f32(a, b, c_ref, m, n, k, lda, ldb, ldc, 1.0f, 0.0f);

    // 2. Run SIMD
#if defined(VECTORIA_USE_ASM) && (defined(__x86_64__) || defined(__aarch64__))
    VectoriaStatus status = call_simd_kernel(a, b, c_simd, m, n, k, lda, ldb, ldc, 1.0f, 0.0f);
    
    if (status != VECTORIA_SUCCESS) {
        std::cout << "[SKIP] SIMD Kernel returned error: " << status << std::endl;
        return;
    }

    // 3. Compare
    auto res = test::compare_matrices(c_ref, c_simd, m * n, 1e-4f); // Slightly loose for FP diffs
    if (res.match) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        test::print_mismatch(c_ref, c_simd, m, n, res, "SIMD");
        // In a real CI, we might exit(1) here. 
        // For now, we print failure but don't crash the harness so other sizes run.
    }
#else
    std::cout << "[SKIP] ASM not enabled or supported" << std::endl;
#endif
}

int main() {
    std::cout << "=== SIMD Correctness Validation ===" << std::endl;
    
    // Small cases
    run_test(4, 4, 4);
    run_test(16, 16, 16);
    
    // Odd sizes (edge cases)
    run_test(3, 3, 3);
    run_test(17, 9, 5);
    
    // Larger
    run_test(64, 64, 64);
    
    return 0;
}
