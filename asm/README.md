# VECTORIA Assembly Kernels

This directory contains architecture-specific SIMD kernels used by the VECTORIA Core Engine.

## Status

### ARM64 (NEON)
- Fully implemented
- Correctness validated against reference kernels
- Continuously tested on Apple Silicon via GitHub Actions

### x86_64 (AVX2)
- Fully implemented
- Correctness validated against reference kernels
- Continuously tested on Linux via GitHub Actions

## Structure

```
arm64/
  NEON implementations (Apple Silicon, AArch64)

x86_64/
  AVX2 implementations (Intel/AMD)
```

## Notes

- SIMD kernels are opt-in and never selected implicitly.
- Reference kernels remain the ground truth.
- Results are bitwise deterministic on the same architecture.

See [docs/kernels.md](../docs/kernels.md) for ABI and validation details.