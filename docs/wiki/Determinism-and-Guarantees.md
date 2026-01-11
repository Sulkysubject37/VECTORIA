# Determinism and Guarantees

**Purpose:** Defines the hard boundaries of what is and isn't guaranteed regarding reproducibility.

VECTORIA distinguishes between **Architectural Determinism** and **Bitwise Reproducibility**.

## Hard Guarantees

1.  **Topology:** For a given sequence of API calls, the resulting Graph IR and `Engine` schedule are identical.
2.  **Memory Layout:** The `Arena` allocator assigns bit-exact same offsets for buffers across runs on the same architecture.
3.  **Reference Bitwise Identity:** The Reference kernels produce bitwise identical results across **all** supported platforms (x86_64 and ARM64).

## Conditional Guarantees (SIMD)

When `VECTORIA_USE_ASM` (SIMD) is enabled:
*   **Intra-Platform:** Results are bitwise deterministic on the *same* machine (e.g., run 1 vs run 2 on M1 Max).
*   **Cross-Platform:** Results **may drift** between ARM64 NEON and x86_64 AVX2 due to differences in Fused Multiply-Add (FMA) implementation and rounding at the hardware level.

## Non-Guarantees

*   **Floating Point Associativity:** We do not use compensated summation (like Kahan). Large reductions may accumulate standard floating-point errors, though these errors are deterministic per architecture.
*   **Concurrent Execution:** The current engine is strictly single-threaded. No guarantees are made for experimental multi-threaded forks.

## Validation

Determinism is continuously verified by `core/tests/test_determinism_stress.cpp`, which executes complex graphs repeatedly to detect state leaks.

## References

*   [docs/determinism.md](../determinism.md)
*   `core/tests/test_determinism_stress.cpp`