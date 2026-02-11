# Transformer Encoder Block (Semantic Specification)

**OpType:** Composed (No intrinsic kernel)
**Status:** Active (Inference-Only, Reference Execution)

## Definition

The Transformer Encoder Block is a high-level semantic unit that combines Multi-Head Attention (MHA) and a Position-wise Feed-Forward Network (FFN) with residual connections and layer normalization.

Given input tensor $X$ of shape $[T, d_{model}]$ and necessary parameters:

1.  **Multi-Head Attention:**
    $$ M = \text{MHA}(X) $$
2.  **First Residual & Norm:**
    $$ Y = \text{LayerNorm}(X + M) $$
3.  **Feed-Forward Network:**
    $$ F = \text{Linear}_2(\text{ReLU}(\text{Linear}_1(Y))) $$
4.  **Second Residual & Norm:**
    $$ O = \text{LayerNorm}(Y + F) $$

## Semantic Expansion

The block is expanded into an explicit graph of primitive and composed operations to ensure absolute traceability:

*   **Sub-block 1 (Attention):**
    *   `add_multi_head_attention_composed`: Expands into projections, head splitting, scaled dot-product attention, and head concatenation.
    *   `Add`: Residual connection between $X$ and $M$.
    *   `add_layernorm_composed`: Normalization of the sum.
*   **Sub-block 2 (FFN):**
    *   `MatMul` + `BiasAdd`: First linear projection (Expansion dimension $d_{ff}$). 
    *   `ReLU`: Non-linear activation.
    *   `MatMul` + `BiasAdd`: Second linear projection back to $d_{model}$.
    *   `Add`: Residual connection between $Y$ and $F$.
    *   `add_layernorm_composed`: Final normalization.

## Constraints & Requirements

*   **Shape Consistency:** The dimensions of $X$ must be compatible with the projection weights. $d_{model}$ must match the MHA and FFN configurations.
*   **Parameter Completeness:** All weight and bias tensors must be explicitly provided as graph nodes.
*   **Reference Backend:** Every node in the expanded graph is executed using VECTORIA's bit-exact reference kernels.

## Relationship to TRUTH.md

The Transformer Encoder Block is a primary semantic unit of composition in VECTORIA. It demonstrates that complex, state-of-the-art architectures can be modeled without resorting to "magic" black-box kernels. Every floating-point operation is part of an auditable topological sort.

## Non-Goals

*   **Fused Block:** There is no `OpType::TransformerEncoder`.
*   **Training:** Gradients and dropout are not supported.
*   **Optimization:** No intermediate buffer reuse or kernel fusion. All tensors are materialized for full auditability.
