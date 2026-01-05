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

### x86_64 AVX2 (Validated)
- **Runner**: `ubuntu-latest`
- **Scope**: Full SIMD validation (conditional on hardware availability).
- **Note**: Most GitHub Ubuntu runners support AVX2. The workflow detects this at runtime and executes the SIMD path if possible.
- **Status**: Results in this environment are considered **Proven** when AVX2 is detected.

## Validation Logs
Logs from correctness tests are preserved as CI artifacts. These logs are part of the scientific provenance of every VECTORIA release.
