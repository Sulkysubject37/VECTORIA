#if defined(__x86_64__)

.section .text
.global gemm_f32_avx2

/*
 * vectoria_gemm_f32_avx2
 */
gemm_f32_avx2:
    // Prologue
    pushq %rbp
    movq %rsp, %rbp
    
    // Save callee-saved registers
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    
    // Load stack arguments (relative to RBP)
    // [rbp+16] = lda
    // [rbp+24] = ldb
    // [rbp+32] = ldc
    
    movq 16(%rbp), %r10 // lda
    movq 24(%rbp), %r11 // ldb
    movq 32(%rbp), %r12 // ldc
    
    // Convert strides to bytes
    shlq $2, %r10
    shlq $2, %r11
    shlq $2, %r12
    
    // Move m to r13
    movq %rcx, %r13
    
    // Broadcast alpha/beta
    vbroadcastss %xmm0, %ymm0
    vbroadcastss %xmm1, %ymm1
    
    // Loop M
    // rdi = A Row Start
    // rdx = C Row Start
.L_m_loop:
    testq %r13, %r13
    jz .L_end
    
    // Setup N Loop
    // r14 = n (counter)
    movq %r8, %r14
    
    // rbx = b_col_ptr (init from rsi)
    movq %rsi, %rbx
    
    // r9 = c_col_ptr (init from rdx) -> Wait, r9 is K.
    // I can use r9 as scratch? No, I need K later for loop.
    // I need to save K somewhere?
    // r9 is scratch? No, r9 is argument 6.
    // I can move K to a callee-saved reg? No, I used them all?
    // r12, r13, r14, r15 used.
    // rbx used.
    // r10, r11, r12 used for strides.
    // rax used for A working.
    // rcx used for B working inside K loop.
    
    // I need one more register for C working ptr!
    // I can use `rax`? No, used in K loop.
    // I can use `rcx`? No, used in K loop.
    
    // Wait. `r13` is M counter.
    // `r14` is N counter.
    // `r15` is K counter.
    // `r9` holds K value (constant).
    // `r8` holds N value (constant).
    
    // I can use `r8` as C working ptr.
    // But I need N value (`r8`) to reset `r14`?
    // `movq %r8, %r14` is done at start of M loop.
    // But `r8` must be preserved across M loops.
    // `r8` is caller-saved. I can clobber it?
    // Yes, but I need the value of N for the NEXT M iteration.
    // So I must preserve N.
    
    // I am out of registers?
    // x86_64 has 16 GPRs.
    // allocated:
    // rdi (A ptr)
    // rsi (B ptr)
    // rdx (C ptr)
    // rcx (M arg -> r13 counter) -> RCX free?
    // r8  (N arg -> ?)
    // r9  (K arg -> ?)
    // r10 (lda)
    // r11 (ldb)
    // r12 (ldc)
    // r13 (M counter)
    // r14 (N counter)
    // r15 (K counter)
    // rbx (B working)
    // rax (A working)
    
    // Free: RCX (after move to r13).
    // Free: RAX (after K loop? No used inside).
    // So RCX is free outside K loop?
    // Inside K loop I used `rcx` as `b_k_ptr`.
    // I can use `r8` or `r9` inside K loop if I save them?
    
    // Better plan: Store N and K on stack?
    // Or just use RCX as C working ptr?
    // Inside N loop:
    // `movq %rdx, %rcx`. (C working).
    // Inside K loop:
    // I need `b_k_ptr`. I was using `rcx`.
    // I can use `rax` for `b_k_ptr`? No, `rax` is `a_k_ptr`.
    
    // I really need one more register.
    // I can put strides `r10, r11, r12` on stack?
    // They are loop invariants.
    // But accessed every K iteration (ldb).
    // Accessing memory in inner loop is slow? L1 cache is fast.
    // Maybe keep `ldb` (r11) in register.
    // `lda` (r10) used once per K iter (A load).
    // `ldc` (r12) used once per N iter.
    
    // Let's spill `ldc` (r12) to stack?
    // `r12` becomes available.
    
    // Let's optimize usage:
    // Keep `r9` (K) valid.
    // Keep `r8` (N) valid.
    
    // Use `rax` and `rcx` as scratch in K loop.
    
    // C working ptr needs to live across K loop.
    // N loop:
    //   C_ptr = ...
    //   K loop (uses rax, rcx).
    //   C_ptr access.
    
    // So C_ptr cannot be rax or rcx.
    // Must be rbx? rbx is B working ptr.
    // B working ptr updates across N loop?
    // N loop:
    //   B_col_ptr (rbx)
    //   K loop uses copy of rbx?
    //   advance rbx.
    
    // So `rbx` holds `b_col_ptr`.
    // I need `c_col_ptr`.
    
    // Let's spill `lda` (r10).
    // Use `r10` as C working ptr.
    // Inside K loop:
    //   Load `lda` from stack? Or `lda_bytes`?
    //   `addq <mem>, %rax`.
    
    // Wait. `lda_bytes` is only used to update `rdi` (A row ptr) at end of M loop.
    // It is NOT used in inner loops!
    // A advances by 4 bytes (scalar) in K loop.
    // So `r10` (lda) is unused in inner loops.
    // PERFECT.
    
    // Plan:
    // `r10` holds C working ptr.
    // `lda_bytes` stays on stack (or calculated at end).
    // Wait, `lda` passed on stack. I loaded it to `r10`.
    // I will just LOAD it again at end of M loop.
    
    // Argument 7 (lda) is at `16(%rbp)`.
    // I need `lda_bytes` = `lda * 4`.
    // I can compute it at end of loop.
    
    // Implementation:
    // Don't load `lda` into `r10`.
    // Use `r10` as C working ptr.
    
    // Setup N Loop
    movq %r8, %r14  // N counter
    movq %rsi, %rbx // B col ptr
    movq %rdx, %r10 // C col ptr
    
.L_n_loop_vec:
    cmpq $8, %r14
    jl .L_n_loop_scalar
    
    // Vector Body
    vxorps %ymm2, %ymm2, %ymm2
    
    movq %r9, %r15 // K counter
    movq %rdi, %rax // A k ptr
    movq %rbx, %rcx // B k ptr
    
.L_k_loop_vec:
    testq %r15, %r15
    jz .L_k_loop_vec_end
    
    vbroadcastss (%rax), %ymm3
    addq $4, %rax
    
    vmovups (%rcx), %ymm4
    addq %r11, %rcx // B += ldb_bytes
    
    vfmadd231ps %ymm4, %ymm3, %ymm2
    
    decq %r15
    jmp .L_k_loop_vec
    
.L_k_loop_vec_end:
    vmulps %ymm0, %ymm2, %ymm2
    vmovups (%r10), %ymm5
    vfmadd231ps %ymm5, %ymm1, %ymm2
    vmovups %ymm2, (%r10)
    
    addq $32, %rbx
    addq $32, %r10
    subq $8, %r14
    jmp .L_n_loop_vec

.L_n_loop_scalar:
    testq %r14, %r14
    jz .L_n_loop_end
    
    vxorps %xmm2, %xmm2, %xmm2
    
    movq %r9, %r15
    movq %rdi, %rax
    movq %rbx, %rcx
    
.L_k_loop_scalar:
    testq %r15, %r15
    jz .L_k_loop_scalar_end
    
    vmovss (%rax), %xmm3
    addq $4, %rax
    
    vmovss (%rcx), %xmm4
    addq %r11, %rcx
    
    vfmadd231ps %xmm4, %xmm3, %xmm2
    
    decq %r15
    jmp .L_k_loop_scalar
    
.L_k_loop_scalar_end:
    vmulss %xmm0, %xmm2, %xmm2
    vmovss (%r10), %xmm5
    vfmadd231ps %xmm5, %xmm1, %xmm2
    vmovss %xmm2, (%r10)
    
    addq $4, %rbx
    addq $4, %r10
    decq %r14
    jmp .L_n_loop_scalar
    
.L_n_loop_end:
    // Advance Row Pointers
    // Need lda_bytes
    movq 16(%rbp), %rax // lda
    shlq $2, %rax
    addq %rax, %rdi // A += lda_bytes
    
    // Need ldc_bytes
    // r12 holds ldc_bytes (computed at start)
    addq %r12, %rdx // C += ldc_bytes
    
    decq %r13
    jmp .L_m_loop

.L_end:
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    xorl %eax, %eax
    ret

#endif
