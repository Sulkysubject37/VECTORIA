#if defined(__x86_64__)

.section .text
.global gemm_f32_avx2

/*
 * vectoria_gemm_f32_avx2
 * 
 * Signature matches kernel_abi.hpp:
 * int32_t gemm_f32_t(
 *     const float* a,      // rdi
 *     const float* b,      // rsi
 *     float* c,            // rdx
 *     size_t m,            // rcx
 *     size_t n,            // r8
 *     size_t k,            // r9
 *     size_t lda,          // stack
 *     size_t ldb,          // stack
 *     size_t ldc,          // stack
 *     float alpha,         // xmm0
 *     float beta           // xmm1
 * )
 *
 * STUB IMPLEMENTATION - DOES NOTHING
 */
gemm_f32_avx2:
    // Standard prologue
    pushq %rbp
    movq %rsp, %rbp

    // Return 0 (VECTORIA_SUCCESS)
    xorl %eax, %eax

    // Standard epilogue
    popq %rbp
    ret

#endif
