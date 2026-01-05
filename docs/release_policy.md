# Release Policy

This document defines the criteria for official VECTORIA releases and our versioning discipline.

## Versioning Strategy
VECTORIA follows [Semantic Versioning (SemVer)](https://semver.org/).
- **Major**: Breaking IR changes or ABI changes.
- **Minor**: New operations, architectures, or bindings.
- **Patch**: Bug fixes, performance tweaks (non-semantic).

Current Version: **v1.0.0-beta**

## Release Criteria (v0.1.0)
A release candidate qualifies for v0.1.0 if and only if it meets the following:

1. **Semantic Truth**: All operations must have a reference C++ implementation that defines bitwise correctness.
2. **Platform Parity**: ARM64 and x86_64 must pass the same SIMD validation harness.
3. **Traceability**: 100% of kernel dispatch decisions must be visible in the trace.
4. **Stress Tested**: Determinism stress tests must pass with zero divergence.
5. **No Fusion**: At this stage, no cross-op fusion is allowed to ensure compositions are predictable.

## Validation Artifacts
Every release must include:
- `SIMD_VALIDATION_REPORT`: Output of `test_gemm_simd` and `test_cross_arch_equivalence`.
- `DETERMINISM_AUDIT`: Results from `test_determinism_stress`.
- `BENCHMARK_BASELINE`: Reference vs SIMD throughput numbers for regression detection.

## Unstable Areas
The following are explicitly considered **unstable** and subject to change without major version bumps in the 0.x series:
- CoreML lowering paths.
- Python bridge internal memory mapping.
- Trace event string formats.
