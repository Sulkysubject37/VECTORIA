# VECTORIA Assembly Kernels

This directory contains architecture-specific assembly implementations of core kernels.

## Supported Architectures

### ARM64 (NEON)
- `gemm_neon.S`: SGEMM
- `add_neon.S`: Elementwise Add
- `mul_neon.S`: Elementwise Mul
- `sub_neon.S`: Elementwise Sub
- `div_neon.S`: Elementwise Div
- `relu_neon.S`: Elementwise ReLU

### x86_64 (AVX2)
- `gemm_avx2.S`: SGEMM
- `add_avx2.S`: Elementwise Add
- `mul_avx2.S`: Elementwise Mul
- `sub_avx2.S`: Elementwise Sub
- `div_avx2.S`: Elementwise Div
- `relu_avx2.S`: Elementwise ReLU

## Conventions
- **Precision**: FP32 (Single Precision)
- **Calling Convention**: System V AMD64 ABI (Linux/macOS)
- **Symbol Names**: `_function_name` (macOS), `function_name` (Linux)
- **Determinism**: Kernels must produce bitwise identical results to the Reference implementation on the same hardware.