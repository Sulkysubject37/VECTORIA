# x86_64 SIMD Strategy

**Status**: Implementation Strategy (Deferred)

This document outlines the strategy for implementing x86_64 optimizations in VECTORIA.

## Target Architecture
We target **AVX2** (Advanced Vector Extensions 2) as the baseline. 
- **Width**: 256-bit (8 floats).
- **FMA**: Fused Multiply-Add (FMA3) is assumed.
- **Availability**: Haswell (2013) and later.

## Register Blocking Strategy
Unlike the current ARM64 kernel (which computes 4 outputs at a time), AVX2 has 16 YMM registers.
To hide FMA latency (typically 4-5 cycles), we must compute multiple accumulators in parallel.

### Proposed Micro-Kernel (6x16)
- **Registers**: 
  - 12 YMM registers for Accumulators (`6 rows * 2 vectors` = 12 regs).
  - 2 YMM registers for B loads.
  - 1 YMM register for A broadcast.
- **Throughput**: Theoretical peak requires careful unrolling.

## ABI Implications
- **System V AMD64 ABI** (Linux/macOS):
  - Arguments: `rdi, rsi, rdx, rcx, r8, r9`.
  - Additional integer args on stack.
  - Float args in `xmm0`..`xmm7`.
  - **Difference from ARM64**: Caller cleans stack. Stride arguments `lda, ldb, ldc` will likely be on stack (arg 7, 8, 9).
- **MS VC++ ABI** (Windows):
  - Completely different. Requires separate assembly file or logic.
  - **Decision**: VECTORIA initially supports System V ABI only.

## Why Deferred?
1. **Focus**: ARM64 parity was the priority.
2. **Complexity**: x86_64 register pressure is higher (fewer GPRs than ARM64).
3. **Determinism**: AVX FMA rounding can differ from SSE or Reference if not careful.

## Implementation Plan
1. Copy `asm/x86_64/gemm_avx2.S` stub.
2. Implement 1x8 vector loop (simplest).
3. Validate against Reference.
4. Optimize blocking.
