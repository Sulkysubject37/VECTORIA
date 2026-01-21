# Release v1.3.0-stable

**Semantic Surface Frozen | Documentation & Reproducibility Consolidated**

This is a stabilization release that formalizes the semantic surface of VECTORIA following the completion of the Phase 8 Transformer stack. As of this release, the mathematical definitions and expansion logic for all operations are frozen to ensure a reliable foundation for downstream integration.

## 🛡️ Semantic Freeze

*   The expansion logic for high-level operations (`LayerNorm`, `Attention`, `MHA`, `EncoderBlock`) is locked.
*   Numerical behavior is governed by the constitutional [TRUTH.md](TRUTH.md).
*   Any future performance optimizations (e.g., kernel fusion) must prove bitwise identity to the frozen reference expansions.

## 🌟 Capabilities (v1.3.0)

*   **Complete Transformer Stack**: Supports full Transformer Encoder Blocks using purely semantic, composed graphs.
*   **Structural Integrity**: Full support for `Transpose`, `Reshape`, `Concat`, and `Slice` operations with explicit IR nodes.
*   **Numerical Stability**: Numerically robust `LogSoftmax` and `StableSoftmax` implementations integrated by default.
*   **Auditability**: Canonical tracing walkthroughs ensure that the execution of complex blocks is fully transparent.

## 🛠 Hardening & Documentation

*   **Documentation Unification**: Repository-wide audit to ensure consistent terminology and mathematical formatting (LaTeX).
*   **CI Reproducibility**: Hardened documentation for CI validation paths and reproducibility checklists.
*   **No Code Changes**: This release introduces no new numerical kernels or performance shortcuts, preserving the proven stability of the `v1.2.1` core.

## ⚠️ Notes

*   **Inference-Only**: VECTORIA remains an inference-forward framework.
*   **Single-Threaded**: The core engine remains strictly serial to preserve bitwise determinism.