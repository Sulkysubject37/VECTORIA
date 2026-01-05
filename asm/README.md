# VECTORIA Assembly Kernels

This directory contains architecture-specific SIMD kernels.

## Status: ACTIVE
- **ARM64 (NEON)**: Implemented and Validated. Standard Dot-Product GEMM.
- **x86_64 (AVX2)**: Stub only.

## Structure
- `x86_64/`: AVX2/AVX-512 kernels.
- `arm64/`: NEON kernels.

## Conventions
All kernels must adhere to the ABI defined in `core/include/vectoria/kernel_abi.hpp`.
- No memory allocation.
- No system calls.
- Return `VECTORIA_SUCCESS` (0) on success.
