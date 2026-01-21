# Concatenation (Semantic Specification)

**OpType:** Structural (Reference Execution Only)
**Status:** Active

## Definition

Concatenation combines multiple input tensors along a specified existing axis. It is a structural operation that reorders data in linear memory without performing any numerical transformation on the values themselves.

Given a list of input tensors $X_1, X_2, \dots, X_n$:

$\\text{Concat}(X_1, \dots, X_n, \text{axis}=a)$ produces a tensor $Y$ such that:
*   $Y.\text{shape}[a] = \sum_{i=1}^n X_i.\text{shape}[a]$
*   $Y.\text{shape}[j] = X_1.\text{shape}[j]$ for all $j \neq a$
*   The data from $X_1, \dots, X_n$ is laid out sequentially along axis $a$ in the order of the input list.

## Properties

*   **Structural Purity:** Performs no arithmetic (addition, multiplication, etc.). It strictly moves data.
*   **Determinism:** The data layout in the output buffer is a deterministic function of the input shapes and the concatenation order.
*   **Traceability:** The operation is explicitly recorded in the execution trace, noting the participating nodes and the chosen axis.

## Semantics & Constraints

*   **Rank Consistency:** All input tensors must have the same rank (number of dimensions).
*   **Dimension Compatibility:** All dimensions except the concatenation axis must be identical across all input tensors.
*   **Axis Validity:** The axis $a$ must be a valid dimension index ($0 \le a < \text{rank}$). 
*   **Data Type:** All input tensors must have the same data type.

## Memory Layout Guarantees

In the reference implementation, Concatenation materializes a new contiguous buffer in the Arena. The memory mapping ensures that slices along the concatenation axis correspond exactly to the input buffers.

## Role in Multi-Head Attention (MHA)

Concatenation is the prerequisite for Multi-Head Attention. It allows the outputs of multiple independent attention heads (e.g., shape $[T, d_k]$) to be unified into a single representation (e.g., shape $[T, h \cdot d_k]$) before the final linear projection.

## Relationship to TRUTH.md

Concatenation preserves "Numerical Truth" by maintaining bit-exact identity between input elements and their corresponding positions in the output tensor. It adheres to the single-threaded, deterministic scheduling model of VECTORIA.

## Non-Goals

*   **Implicit Stacking:** This operation does not create a new axis (use Reshape + Concat or a future Stack op for that).
*   **Zero-Copy Views:** To maintain strict ownership and avoid complex memory aliasing bugs in the reference backend, Concatenation materializes its output.
*   **Dynamic Axis Inference:** The axis must be explicitly provided at graph construction time.
