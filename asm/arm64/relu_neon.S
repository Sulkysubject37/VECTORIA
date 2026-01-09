#if defined(__aarch64__)

.text
.p2align 4

#if defined(__APPLE__)
.global _relu_f32_neon
_relu_f32_neon:
#else
.global relu_f32_neon
relu_f32_neon:
#endif

// VectoriaStatus relu_f32_neon(const float* in, float* out, size_t count)
// x0 = in, x1 = out, x2 = count

    cbz x2, .L_end
    movi v1.4s, #0

.L_loop:
    cmp x2, #4
    blt .L_scalar

    ld1 {v0.4s}, [x0], #16
    fmax v0.4s, v0.4s, v1.4s
    st1 {v0.4s}, [x1], #16

    sub x2, x2, #4
    b .L_loop

.L_scalar:
    cbz x2, .L_end
    
    ldr s0, [x0], #4
    fmax s0, s0, s1 // s1 is 0.0 from v1
    str s0, [x1], #4
    
    sub x2, x2, #1
    b .L_scalar

.L_end:
    mov w0, #0
    ret

#endif

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
