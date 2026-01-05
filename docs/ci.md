# Continuous Integration & Validation

VECTORIA uses GitHub Actions not just for build hygiene, but as a formal **Correctness Proof** mechanism.

## Architecture-Aware Testing

CI is split into architecture-specific workflows to reflect the reality of hardware availability.

### ARM64 NEON (Validated)
- **Runner**: `macos-latest` (Apple Silicon)
- **Scope**: Full SIMD validation.
- **Status**: Results in this environment are considered **Proven**.

### x86_64 AVX2 (Validated)
- **Runner**: `ubuntu-latest`
- **Scope**: Full SIMD validation (detected at runtime).
- **Status**: Results in this environment are considered **Proven**.

## Tests Executed on CI
Every commit triggers the following suites on both validated architectures:
- `test_gemm_simd`: Compares SIMD kernels against Reference.
- `test_multi_op`: Verifies composite graph execution.
- `test_graph_equivalence`: Ensures Reference/SIMD parity.
- `test_determinism_stress`: Asserts bitwise reproducibility over 50 iterations.

## Validation Logs
Logs from correctness tests are preserved as CI artifacts. These logs are part of the scientific provenance of every VECTORIA release.