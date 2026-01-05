# VECTORIA Assembly Kernels

This directory contains architecture-specific SIMD kernels used by the VECTORIA Core Engine.

## Status

### ARM64 (NEON)
- Fully implemented
- Correctness validated against reference kernels
- Continuously tested on Apple Silicon via GitHub Actions

### x86_64 (AVX2)
- Implemented
- Correctness validated locally
- Not yet validated on CI due to lack of AVX2-enabled runners

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
- Lack of CI validation does not imply incorrectness, only unverified execution on public runners.

See [docs/kernels.md](../docs/kernels.md) for ABI and validation details.
