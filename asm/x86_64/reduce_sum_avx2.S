#if defined(__x86_64__)

.text
.p2align 4
.global reduce_sum_f32_avx2

// VectoriaStatus reduce_sum_f32_avx2(const float* in, float* out, size_t outer, size_t inner)
// rdi = in, rsi = out, rdx = outer, rcx = inner

reduce_sum_f32_avx2:
    testq %rdx, %rdx
    jz .L_end
    testq %rcx, %rcx
    jz .L_end

    // Pre-calc inner_vec (multiple of 8)
    movq %rcx, %r8
    andq $-8, %r8 // inner_vec = inner & ~7
    movq %rcx, %r9
    subq %r8, %r9 // inner_tail

.L_outer_loop:
    movq %rdi, %rax // inner_ptr
    movq %r8, %r10 // vec_cnt
    
    vxorps %ymm0, %ymm0, %ymm0 // Sum acc

    testq %r10, %r10
    jz .L_vector_reduction

.L_inner_loop:
    vaddps (%rax), %ymm0, %ymm0
    addq $32, %rax
    subq $8, %r10
    jnz .L_inner_loop

.L_vector_reduction:
    // Horizontal reduction of ymm0 [y7..y0]
    vextractf128 $1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    // Now we have 4 sums in xmm0: [z3, z2, z1, z0]
    vshufps $0xb1, %xmm0, %xmm0, %xmm1 // [z2, z3, z0, z1]
    vaddps %xmm1, %xmm0, %xmm0         // [z3+z2, z2+z3, z1+z0, z0+z1]
    // Move high 64-bit (z3+z2) to low 64-bit
    vunpckhpd %xmm0, %xmm0, %xmm1     // Lane 0 of xmm1 is now lane 1 of xmm0 (z3+z2)
    vaddss %xmm1, %xmm0, %xmm0         // (z1+z0) + (z3+z2)
    // xmm0[0] holds final sum

    movq %r9, %r10 // tail_cnt
    testq %r10, %r10
    jz .L_store

.L_tail_loop:
    vaddss (%rax), %xmm0, %xmm0
    addq $4, %rax
    decq %r10
    jnz .L_tail_loop

.L_store:
    vmovss %xmm0, (%rsi)
    addq $4, %rsi
    
    // Advance rdi by inner * 4
    movq %rcx, %r11
    shlq $2, %r11
    addq %r11, %rdi
    
    decq %rdx
    jnz .L_outer_loop

.L_end:
    xorl %eax, %eax
    ret

#endif

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
