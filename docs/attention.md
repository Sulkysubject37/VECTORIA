# Scaled Dot-Product Attention (Semantic Specification)

**OpType:** Composed (No intrinsic kernel)
**Status:** Active (Inference-Only, Reference Execution)

## Definition

Scaled Dot-Product Attention is the core mechanism of Transformer architectures. It computes a weighted sum of values, where the weights are determined by the compatibility (dot product) of a query with a set of keys.

Given input tensors:
*   **Query ($Q$)**: Shape $[T, d_k]$
*   **Key ($K$)**: Shape $[T, d_k]$
*   **Value ($V$)**: Shape $[T, d_v]$

The operation is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

Where:
*   $d_k$ is the dimension of the keys (last axis of $Q$ and $K$).
*   The softmax is applied along the last axis.
*   $\sqrt{d_k}$ scaling is mandatory for numerical stability to prevent vanishing gradients (even in inference, it maintains distribution scale).

## Semantics & Constraints

*   **Precision:** All operations are performed in **FP32** (Single Precision).
*   **Composition:** This operation is expanded at graph construction time into the following explicit sequence:
    1.  `Transpose` of $K$ to get $K^\top$ (Shape $[d_k, T]$).
    2.  `MatMul` of $Q$ and $K^\top$ to get Scores $S$ (Shape $[T, T]$).
    3.  `Mul` by scalar constant $1/\sqrt{d_k}$ to get Scaled Scores.
    4.  `StableSoftmax` (LogSoftmax + Exp) applied to Scaled Scores.
    5.  `MatMul` of the attention weights and $V$ to get Output $O$ (Shape $[T, d_v]$).
*   **Broadcasting:** Currently strictly defined for matching sequence lengths $T$ (single block). Batching semantics are not implemented in the current specification.

## Determinism

*   This operation relies on the determinism of its constituent parts: `MatMul` (Reference), `Transpose`, `Mul`, and `StableSoftmax`.
*   Result is bitwise deterministic on the same architecture.

## Non-Goals

*   **Multi-Head:** This specification covers single-head attention only.
*   **Masking:** No causal or padding masks are applied.
*   **Fused Kernel:** There is no `OpType::Attention`. This is a purely semantic expansion.
*   **Performance:** No FlashAttention or memory-efficient optimizations are used. All intermediate matrices (Scores, Weights) are materialized in the Arena.
