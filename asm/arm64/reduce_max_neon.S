#if defined(__aarch64__)

.text
.p2align 4

#if defined(__APPLE__)
.global _reduce_max_f32_neon
_reduce_max_f32_neon:
#else
.global reduce_max_f32_neon
reduce_max_f32_neon:
#endif

// VectoriaStatus reduce_max_f32_neon(const float* in, float* out, size_t outer, size_t inner)
// x0 = in, x1 = out, x2 = outer, x3 = inner

    cbz x2, .L_end
    cbz x3, .L_end

    // Pre-calculate inner_vec (multiple of 4) and inner_tail
    and x7, x3, #~3 // inner_vec = inner & ~3
    sub x8, x3, x7  // inner_tail = inner - inner_vec

    // Load -Infinity
    mov w9, #0xFF800000 
    fmov s2, w9

.L_outer_loop:
    mov x4, x0  // inner_ptr
    mov x5, x7  // vec_counter = inner_vec
    
    // Init vector accumulator with -Inf
    dup v0.4s, v2.s[0]

    cbz x5, .L_vector_reduction

.L_inner_loop:
    ldur q1, [x4]
    add x4, x4, #16
    fmax v0.4s, v0.4s, v1.4s
    
    sub x5, x5, #4
    cbnz x5, .L_inner_loop

.L_vector_reduction:
    // Horizontal Max
    fmaxv s0, v0.4s
    
    mov x5, x8 // tail_counter = inner_tail
    cbz x5, .L_store_result

.L_tail_loop:
    ldr s1, [x4], #4
    fmax s0, s0, s1
    sub x5, x5, #1
    cbnz x5, .L_tail_loop

.L_store_result:
    str s0, [x1], #4
    
    lsl x6, x3, #2
    add x0, x0, x6
    
    sub x2, x2, #1
    cbnz x2, .L_outer_loop

.L_end:
    mov w0, #0
    ret

#endif

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
