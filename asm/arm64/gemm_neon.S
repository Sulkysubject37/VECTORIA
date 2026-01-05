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
 *
 * Arguments:
 *   x0: const float* a
 *   x1: const float* b
 *   x2: float* c
 *   x3: size_t m
 *   x4: size_t n
 *   x5: size_t k
 *   [sp]:    size_t lda
 *   [sp+8]:  size_t ldb
 *   [sp+16]: size_t ldc
 *   s0: float alpha
 *   s1: float beta
 *
 * Registers:
 *   x9:  lda_bytes
 *   x10: ldb_bytes
 *   x11: ldc_bytes
 *   x19: m_counter
 *   x20: n_counter
 *   x21: k_counter
 *   x22: a_ptr
 *   x23: b_col_ptr (ptr to column j of B)
 *   x24: c_ptr
 *   x25: b_k_ptr (ptr walking down column j of B)
 *   x26: a_k_ptr (ptr walking across row i of A)
 *
 *   v0: alpha (broadcast)
 *   v1: beta (broadcast)
 *   v2: accumulator (vector)
 *   v3: a_val (broadcast)
 *   v4: b_val (vector)
 *   v5: c_val (vector)
 */

    // Prologue
    stp x29, x30, [sp, -80]!
    mov x29, sp
    stp x19, x20, [sp, 16]
    stp x21, x22, [sp, 32]
    stp x23, x24, [sp, 48]
    stp x25, x26, [sp, 64]

    // Load stack arguments (lda, ldb, ldc)
    // Caller stack frame is above ours. Our frame size is 80.
    // So arguments are at [sp, 80 + 0], [sp, 80 + 8], [sp, 80 + 16]
    ldr x9, [sp, 80]
    ldr x10, [sp, 88]
    ldr x11, [sp, 96]

    // Convert strides to bytes
    lsl x9, x9, #2
    lsl x10, x10, #2
    lsl x11, x11, #2

    // Broadcast alpha and beta
    // s0 and s1 are already populated
    // We want them in all lanes of v0, v1 for vectorized ops
    dup v0.4s, v0.s[0]
    dup v1.4s, v1.s[0]
    
    // M Loop (Rows)
    mov x19, x3
    mov x22, x0  // a_row_start
    mov x24, x2  // c_row_start

.L_m_loop:
    cbz x19, .L_end

    // N Loop (Cols)
    mov x20, x4
    mov x23, x1  // b_col_start (points to B[0, current_col])
    
    // Split N into multiples of 4 and remainder
.L_n_loop_vec:
    cmp x20, #4
    blt .L_n_loop_scalar

    // Vectorized N body (4 floats)
    // Initialize accumulator to 0
    movi v2.4s, #0

    // K Loop
    mov x21, x5
    mov x26, x22 // a_k_ptr = a_row_start
    mov x25, x23 // b_k_ptr = b_col_start

.L_k_loop_vec:
    cbz x21, .L_k_loop_vec_end
    
    // Load A[i, k] (scalar) -> Broadcast
    ldr s3, [x26], #4        // Post-index: load float, then add 4
    dup v3.4s, v3.s[0]

    // Load B[k, j...j+3] (vector)
    ldr q4, [x25]            // Load 128-bit
    add x25, x25, x10        // b_k_ptr += ldb_bytes

    // Accumulate: acc += a * b
    fmla v2.4s, v3.4s, v4.4s

    sub x21, x21, #1
    b .L_k_loop_vec

.L_k_loop_vec_end:
    // Scale acc by alpha
    fmul v2.4s, v2.4s, v0.4s

    // Load C[i, j...j+3]
    ldr q5, [x24]
    
    // C = acc + beta * C
    fmla v2.4s, v5.4s, v1.4s  // v2 = v2 + (v5 * v1) => alpha*sum + beta*C_old

    // Store C
    str q2, [x24], #16

    // Advance loop pointers
    add x23, x23, #16        // b_col_start += 4 floats
    sub x20, x20, #4
    b .L_n_loop_vec

.L_n_loop_scalar:
    cbz x20, .L_n_loop_end

    // Scalar N body (1 float)
    // Initialize accumulator s2 (using scalar reg)
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

    fmadd s2, s3, s4, s2     // s2 = s3*s4 + s2

    sub x21, x21, #1
    b .L_k_loop_scalar

.L_k_loop_scalar_end:
    // Scale acc by alpha (use s0 low element of v0)
    fmul s2, s2, v0.s[0]

    // Load C
    ldr s5, [x24]

    // C = acc + beta * C
    fmadd s2, s5, v1.s[0], s2

    // Store C
    str s2, [x24], #4

    add x23, x23, #4
    sub x20, x20, #1
    b .L_n_loop_scalar

.L_n_loop_end:
    // Advance Row Pointers
    add x22, x22, x9         // a_row_start += lda_bytes
    add x24, x24, x11        // c_row_start += ldc_bytes
    
    sub x19, x19, #1
    b .L_m_loop

.L_end:
    // Epilogue
    ldp x25, x26, [sp, 64]
    ldp x23, x24, [sp, 48]
    ldp x21, x22, [sp, 32]
    ldp x19, x20, [sp, 16]
    ldp x29, x30, [sp], 80
    
    mov w0, #0
    ret

#endif