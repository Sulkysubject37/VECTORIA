# Continuous Integration & Validation

VECTORIA uses GitHub Actions not just for build hygiene, but as a formal **Correctness Proof** mechanism.

## Architecture-Aware Testing

CI is split into architecture-specific workflows to reflect the reality of hardware availability.

### ARM64 NEON (Validated)
- **Runner**: `macos-latest` (Apple Silicon)
- **Scope**: Full SIMD validation.
- **Tests**: 
  - `test_gemm_simd`: Compares NEON kernels against Reference.
  - `test_multi_op`: Verifies composite graph execution.
  - `test_graph_equivalence`: Ensures Reference/SIMD parity.
  - `test_determinism_stress`: Asserts bitwise reproducibility.
- **Status**: Results in this environment are considered **Proven**.

### x86_64 AVX2 (Unvalidated)
- **Runner**: `ubuntu-latest`
- **Scope**: Reference implementation and build integrity.
- **Note**: AVX2 kernels are compiled but **NOT** executed/validated due to lack of supported hardware on standard public runners.
- **Status**: Results in this environment are considered **Unverified** for SIMD.

## Validation Logs
Logs from correctness tests are preserved as CI artifacts. These logs are part of the scientific provenance of every VECTORIA release.
