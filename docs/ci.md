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
- `test_gemm_simd`: Compares SIMD kernels against bit-exact Reference.
- `test_multi_op`: Verifies composite graph execution and deterministic scheduling.
- `test_graph_equivalence`: Ensures Reference/SIMD parity within validated tolerance.
- `test_determinism_stress`: Asserts bitwise reproducibility over 50 iterations.
- **High-Level Semantic Suites (Reference-Only)**:
    - `test_layernorm`: Statistical property and broadcast validation.
    - `test_attention`: Numerical equivalence for Scaled Dot-Product expansion.
    - `test_multi_head_attention`: Structural splitting and join validation.
    - `test_transformer_encoder`: Full block integration proof.

## Supported Validation Platforms
CI explicitly proves correctness on:
- **macOS (Latest)**: ARM64 NEON backend + CoreML lowering.
- **Ubuntu (Latest)**: x86_64 AVX2 backend.

## Reproducibility Checklist
To achieve bitwise identity between a local environment and CI:
1.  **Fixed Binary**: Use the exact same compiled artifact.
2.  **Identical Architecture**: Compare ARM64 to ARM64 or x86_64 to x86_64.
3.  **Kernel Policy**: Ensure both use `KernelPolicy::Reference` if cross-architecture identity is required.
4.  **No OS Noise**: VECTORIA's single-threaded model eliminates OS scheduling noise from the result.

## Scope of CI Validation
**Tested**:
- Bit-exact Reference correctness.
- SIMD vs. Reference parity within tolerance.
- Structural op metadata preservation.
- CoreML MIL generation and validation.

**NOT Tested**:
- Multi-threaded execution (Engine is strictly serial).
- Third-party library interference (NumPy/PyTorch internal math).
- Driver-level optimization overrides.

## Validation Logs
Logs from correctness tests are preserved as CI artifacts. These logs are part of the scientific provenance of every VECTORIA release.