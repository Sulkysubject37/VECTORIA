# Reshape (Semantic Specification)

**OpType:** Structural (Reference Execution Only)
**Status:** Active

## Definition

Reshape changes the dimensions of a tensor without changing the order of its elements in linear memory.

Given tensor $x$ with shape $[d_0, d_1, \dots, d_{n-1}]$ and a new target shape $[e_0, e_1, \dots, e_{k-1}]$:

$\\text{Reshape}(x, \text{new\_shape})$ is valid if and only if the total number of elements remains constant:
$$ \prod_{i=0}^{n-1} d_i = \prod_{j=0}^{k-1} e_j $$

## Properties

*   **Linear Interpretation:** The operation treats the input tensor as a flattened buffer in row-major order and re-interprets strides based on the new shape.
*   **No Data Movement:** In an idealized reference execution, this operation does not move data, only metadata. However, strict arena allocators may copy data to a new buffer to enforce ownership boundaries.
*   **Determinism:** The output values are identical to the input values, in the same linear order.

## Semantics & Constraints

*   **Explicit Shape:** The `new_shape` must be fully specified. Dynamic shapes or inferred dimensions (e.g., `-1`) are not supported in this strict reference implementation.
*   **Element Count:** The total element count must strictly match. Mismatches result in a runtime error.

## Non-Goals

*   **Implicit Squeeze/Unsqueeze:** Reshape handles rank changes explicitly. There are no separate Squeeze/Unsqueeze ops; Reshape covers these cases.
*   **Broadcasting:** Reshape does not replicate data. It is not a broadcast operation.
