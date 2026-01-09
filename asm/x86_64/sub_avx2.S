#if defined(__x86_64__)

.text
.p2align 4
.global sub_f32_avx2

// VectoriaStatus sub_f32_avx2(const float* a, const float* b, float* out, size_t count)
// rdi = a, rsi = b, rdx = out, rcx = count

sub_f32_avx2:
    testq %rcx, %rcx
    jz .L_end

.L_loop:
    cmpq $8, %rcx
    jl .L_scalar

    vmovups (%rdi), %ymm0
    vmovups (%rsi), %ymm1
    vsubps %ymm1, %ymm0, %ymm0
    vmovups %ymm0, (%rdx)

    addq $32, %rdi
    addq $32, %rsi
    addq $32, %rdx
    subq $8, %rcx
    jmp .L_loop

.L_scalar:
    testq %rcx, %rcx
    jz .L_end
    
    vmovss (%rdi), %xmm0
    vsubss (%rsi), %xmm0, %xmm0
    vmovss %xmm0, (%rdx)
    
    addq $4, %rdi
    addq $4, %rsi
    addq $4, %rdx
    decq %rcx
    jmp .L_scalar

.L_end:
    xorl %eax, %eax
    ret

#endif

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
