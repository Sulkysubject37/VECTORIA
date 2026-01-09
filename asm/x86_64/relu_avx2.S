#if defined(__x86_64__)

.text
.p2align 4
.global relu_f32_avx2

// VectoriaStatus relu_f32_avx2(const float* in, float* out, size_t count)
// rdi = in, rsi = out, rdx = count

relu_f32_avx2:
    testq %rdx, %rdx
    jz .L_end
    
    vxorps %ymm1, %ymm1, %ymm1 // Zero

.L_loop:
    cmpq $8, %rdx
    jl .L_scalar

    vmovups (%rdi), %ymm0
    vmaxps %ymm1, %ymm0, %ymm0
    vmovups %ymm0, (%rsi)

    addq $32, %rdi
    addq $32, %rsi
    subq $8, %rdx
    jmp .L_loop

.L_scalar:
    testq %rdx, %rdx
    jz .L_end
    
    vmovss (%rdi), %xmm0
    vmaxss %xmm1, %xmm0, %xmm0
    vmovss %xmm0, (%rsi)
    
    addq $4, %rdi
    addq $4, %rsi
    decq %rdx
    jmp .L_scalar

.L_end:
    xorl %eax, %eax
    ret

#endif

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
