# Transpose (Semantic Specification)

**OpType:** Structural (Reference Execution Only)
**Status:** Active

## Definition

Transpose reorders the axes of a tensor without altering its values. It maps indices from the input space to the output space based on a permutation vector.

Given tensor $x$ of shape $[d_0, d_1, \dots, d_{n-1}]$:

$\\text{Transpose}(x, \text{perm})$ produces tensor $y$ such that:
*   $y.\text{shape}[i] = x.\text{shape}[\text{perm}[i]]$
*   $y[ i_0, i_1, \dots, i_{n-1} ] = x[ i_{\text{perm}^{-1}(0)}, \dots ]$

## Properties

*   **Pure Reindexing:** The operation does not perform any arithmetic computation. It strictly permutes the indices.
*   **Determinism:** The mapping from input index to output index is deterministic and defined solely by the permutation vector.
*   **Traceability:** The operation is explicitly represented in the graph and execution trace.

## Semantics & Constraints

*   **Permutation:** The `perm` vector must be a valid permutation of $[0, 1, \dots, n-1]$. It must contain every index exactly once.
*   **Rank Preservation:** The rank of the output tensor is equal to the rank of the input tensor.
*   **Data Type:** Supports all valid data types (FP32, Int32, etc.) as no numerical operations are performed.

## Non-Goals

*   **Performance:** This is a reference implementation. While optimizations like pointer aliasing or strided views are possible, the initial implementation prioritizes correctness and may perform a full memory copy.
*   **Implicit Transpose:** No implicit transpositions (e.g., in MatMul arguments) are supported. Transpose must be an explicit graph node.
