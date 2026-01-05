#if defined(__aarch64__)

.text
.global gemm_f32_neon
.p2align 4

/*
 * vectoria_gemm_f32_neon
 *
 * Signature matches kernel_abi.hpp:
 * int32_t gemm_f32_t(
 *     const float* a,      // x0
 *     const float* b,      // x1
 *     float* c,            // x2
 *     size_t m,            // x3
 *     size_t n,            // x4
 *     size_t k,            // x5
 *     size_t lda,          // stack [sp]
 *     size_t ldb,          // stack [sp, #8]
 *     size_t ldc,          // stack [sp, #16]
 *     float alpha,         // s0
 *     float beta           // s1
 * )
 *
 * STUB IMPLEMENTATION - DOES NOTHING
 */
gemm_f32_neon:
    // Return 0 (VECTORIA_SUCCESS)
    mov w0, #0
    ret

#endif
