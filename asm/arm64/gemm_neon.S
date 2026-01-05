#if defined(__aarch64__)

.text
#if defined(__APPLE__)
.global _gemm_f32_neon
_gemm_f32_neon:
#else
.global gemm_f32_neon
gemm_f32_neon:
#endif
.p2align 4
    // Return 0 (VECTORIA_SUCCESS)
    mov w0, #0
    ret

#endif
