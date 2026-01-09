#if defined(__aarch64__)

.text
.p2align 4

#if defined(__APPLE__)
.global _sub_f32_neon
_sub_f32_neon:
#else
.global sub_f32_neon
sub_f32_neon:
#endif

// VectoriaStatus sub_f32_neon(const float* a, const float* b, float* out, size_t count)
// x0 = a, x1 = b, x2 = out, x3 = count

    cbz x3, .L_end

.L_loop:
    cmp x3, #4
    blt .L_scalar

    ld1 {v0.4s}, [x0], #16
    ld1 {v1.4s}, [x1], #16
    fsub v0.4s, v0.4s, v1.4s
    st1 {v0.4s}, [x2], #16

    sub x3, x3, #4
    b .L_loop

.L_scalar:
    cbz x3, .L_end
    
    ldr s0, [x0], #4
    ldr s1, [x1], #4
    fsub s0, s0, s1
    str s0, [x2], #4
    
    sub x3, x3, #1
    b .L_scalar

.L_end:
    mov w0, #0
    ret

#endif

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
