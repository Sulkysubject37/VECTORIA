#if defined(__x86_64__)

.text
.p2align 4
.global reduce_max_f32_avx2

// VectoriaStatus reduce_max_f32_avx2(const float* in, float* out, size_t outer, size_t inner)
// rdi = in, rsi = out, rdx = outer, rcx = inner

reduce_max_f32_avx2:
    testq %rdx, %rdx
    jz .L_end
    testq %rcx, %rcx
    jz .L_end

    movq %rcx, %r8
    andq $-8, %r8
    movq %rcx, %r9
    subq %r8, %r9

    // Prepare -Inf
    movl $0xFF800000, %eax
    vmovd %eax, %xmm2
    vbroadcastss %xmm2, %ymm2 

.L_outer_loop:
    movq %rdi, %rax // inner_ptr
    movq %r8, %r10 // vec_cnt
    
    vmovaps %ymm2, %ymm0 // Max acc

    testq %r10, %r10
    jz .L_vector_reduction

.L_inner_loop:
    vmaxps (%rax), %ymm0, %ymm0
    addq $32, %rax
    subq $8, %r10
    jnz .L_inner_loop

.L_vector_reduction:
    vextractf128 $1, %ymm0, %xmm1
    vmaxps %xmm1, %xmm0, %xmm0
    // [z3, z2, z1, z0]
    vshufps $0xb1, %xmm0, %xmm0, %xmm1 // [z2, z3, z0, z1]
    vmaxps %xmm1, %xmm0, %xmm0         // [max(z3,z2), max(z2,z3), max(z1,z0), max(z0,z1)]
    vunpckhpd %xmm0, %xmm0, %xmm1     // Lane 0 of xmm1 is now lane 1 of xmm0 (max(z3,z2))
    vmaxss %xmm1, %xmm0, %xmm0

    movq %r9, %r10
    testq %r10, %r10
    jz .L_store

.L_tail_loop:
    vmaxss (%rax), %xmm0, %xmm0
    addq $4, %rax
    decq %r10
    jnz .L_tail_loop

.L_store:
    vmovss %xmm0, (%rsi)
    addq $4, %rsi
    
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
