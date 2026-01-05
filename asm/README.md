# VECTORIA Assembly Kernels

This directory contains architecture-specific SIMD kernels used by the Core Engine.

## Status
- **ARM64 (NEON)**: Verified and Active.
- **x86_64 (AVX2)**: Implemented (Pending hardware validation).

## Structure
- `arm64/`: NEON implementations (Apple Silicon, etc).
- `x86_64/`: AVX2 implementations (Intel/AMD).

See [Kernels Documentation](../docs/kernels.md) for ABI details.
