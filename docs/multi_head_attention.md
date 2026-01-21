# Multi-Head Attention (Semantic Specification)

**OpType:** Composed (No intrinsic kernel)
**Status:** Active (Inference-Only, Reference Execution)

## Definition

Multi-Head Attention (MHA) allows the model to jointly attend to information from different representation subspaces at different positions. It is the fundamental building block of the Transformer architecture.

Given:
*   **Input ($X$):** Shape $[T, d_{model}]$
*   **Heads ($h$):** Number of attention heads.
*   **Head Dimension ($d_k$):** $d_k = d_{model} / h$.
*   **Projections ($W_Q, W_K, W_V, W_O$):** Learned weight matrices of shape $[d_{model}, d_{model}]$.

The operation is defined as:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O $$
$$ \text{where head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) $$

## Semantic Expansion

To maintain absolute transparency and adhere to the "No Magic" principle of VECTORIA, MHA is expanded into an explicit graph of primitive and composed operations:

1.  **Projections:**
    *   $Q_{all} = X W_Q$
    *   $K_{all} = X W_K$
    *   $V_{all} = X W_V$
2.  **Head Splitting:**
    *   Reshape $[T, d_{model}] \to [T, h, d_k]$
    *   Transpose $[T, h, d_k] \to [h, T, d_k]$
    *   **Slice:** Extract $h$ individual tensors of shape $[T, d_k]$ from the transposed result.
3.  **Scaled Dot-Product Attention:**
    *   Invoke `add_attention_composed` for each $(Q_i, K_i, V_i)$ triplet.
4.  **Recomposition:**
    *   **Concat:** Join $h$ head outputs along the last axis to form $H$ of shape $[T, d_{model}]$.
5.  **Output Projection:**
    *   $O = H W_O$

## Constraints & Requirements

*   **Divisibility:** $d_{model}$ must be perfectly divisible by $h$.
*   **Structural Purity:** Head splitting and concatenation are performed using structural operations (`Reshape`, `Transpose`, `Slice`, `Concat`).
*   **Determinism:** Since every step is an explicit node in the graph, the execution is fully deterministic and traceable.

## Relationship to TRUTH.md

MHA preserves numerical truth by decomposing a complex, high-level concept into validated, bit-exact reference primitives. The use of `StableSoftmax` within the attention heads ensures numerical stability across all platforms.

## Non-Goals

*   **Fused MHA:** No monolithic "FlashAttention" style kernels.
*   **Efficiency:** This implementation prioritizes auditability and correctness. The materialization of intermediate heads and scores is expected behavior.
*   **Training:** Gradients are not supported.
