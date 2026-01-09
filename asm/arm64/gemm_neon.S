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

/*
 * vectoria_gemm_f32_neon
 * Fixed Logic: Row pointers x0, x2 updated in outer loop.
 */

    // Prologue
    stp x29, x30, [sp, -80]!
    mov x29, sp
    stp x19, x20, [sp, 16]
    stp x21, x22, [sp, 32]
    stp x23, x24, [sp, 48]
    stp x25, x26, [sp, 64]

    // Load arguments
    // x0..x5 in regs
    // x6 (lda), x7 (ldb) in regs
    // ldc at [sp, 80]
    ldr x11, [sp, 80]
    mov x9, x6
    mov x10, x7

    // Convert strides to bytes
    lsl x9, x9, #2
    lsl x10, x10, #2
    lsl x11, x11, #2

    // Broadcast alpha and beta
    dup v0.4s, v0.s[0]
    dup v1.4s, v1.s[0]
    
    // M Loop (Rows)
    mov x19, x3

.L_m_loop:
    cbz x19, .L_end
    
    // Set working pointers from row heads
    mov x22, x0  // a_row_ptr
    mov x24, x2  // c_row_ptr

    // N Loop (Cols)
    mov x20, x4
    mov x23, x1  // b_col_start

.L_n_loop_vec:
    cmp x20, #4
    blt .L_n_loop_scalar

    // Vectorized N body
    movi v2.4s, #0

    // K Loop
    mov x21, x5
    mov x26, x22 // a_k_ptr
    mov x25, x23 // b_k_ptr

.L_k_loop_vec:
    cbz x21, .L_k_loop_vec_end
    
    // Load A scalar
    ldr s3, [x26], #4
    dup v3.4s, v3.s[0]

    // Load B vector
    // Use ldur for safety if B row starts are unaligned
    ldur q4, [x25]
    add x25, x25, x10

    // Accumulate
    fmla v2.4s, v3.4s, v4.4s

    sub x21, x21, #1
    b .L_k_loop_vec

.L_k_loop_vec_end:
    fmul v2.4s, v2.4s, v0.4s // alpha

    // Load C
    ldur q5, [x24]
    
    // C = acc + beta * C
    fmla v2.4s, v5.4s, v1.4s

    // Store C
    stur q2, [x24]
    add x24, x24, #16

    add x23, x23, #16
    sub x20, x20, #4
    b .L_n_loop_vec

.L_n_loop_scalar:
    cbz x20, .L_n_loop_end

    fmov s2, wzr

    // K Loop
    mov x21, x5
    mov x26, x22
    mov x25, x23

.L_k_loop_scalar:
    cbz x21, .L_k_loop_scalar_end

    ldr s3, [x26], #4
    ldr s4, [x25]
    add x25, x25, x10

    fmadd s2, s3, s4, s2

    sub x21, x21, #1
    b .L_k_loop_scalar

.L_k_loop_scalar_end:
    fmul s2, s2, s0 // alpha

    ldr s5, [x24]
    fmadd s2, s5, s1, s2 // beta

    str s2, [x24], #4

    add x23, x23, #4
    sub x20, x20, #1
    b .L_n_loop_scalar

.L_n_loop_end:
    // Advance Row Pointers
    add x0, x0, x9
    add x2, x2, x11
    
    sub x19, x19, #1
    b .L_m_loop

.L_end:
    ldp x25, x26, [sp, 64]
    ldp x23, x24, [sp, 48]
    ldp x21, x22, [sp, 32]
    ldp x19, x20, [sp, 16]
    ldp x29, x30, [sp], 80
    mov w0, #0
    ret

#endif
#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
