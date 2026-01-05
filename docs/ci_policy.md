# CI Failure & Merge Policy

## 1. Zero-Tolerance for Regression
Any failure in the **Reference Path** (on any architecture) or the **ARM64 NEON Path** (on Apple Silicon) is a blocking event.
- Merges to `main` are strictly prohibited while these paths are failing.
- Release candidates are immediately invalidated if CI fails.

## 2. Unvalidated Architectures
For architectures without CI hardware support (e.g., x86_64 AVX2):
- Build integrity must still pass.
- Local validation logs must be provided by the contributor for any changes affecting these kernels.
- The status remains "Unvalidated" on CI until hardware is available.

## 3. Experimental Kernels
New SIMD kernels are considered **Experimental** until they pass the full `test_gemm_simd` and `test_graph_equivalence` suites.
- If a kernel is unstable or produces non-deterministic results, it must be **disabled by default** in the build configuration.

## 4. Scientific Provenance
CI logs are considered part of the project's permanent record. Tampering with or ignoring CI failure logs is a violation of the correctness contract.
