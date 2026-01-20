# Stable Softmax (Semantic Specification)

**OpType:** Composed (No intrinsic kernel)
**Status:** Active (Reference Execution Only)

## Definition

Stable Softmax is defined as the exponentiation of the numerically stable LogSoftmax operation. It operates over the **last axis** of the input tensor.

Given input tensor $x$ of shape $[..., D]$:

1.  **LogSoftmax:**
    $$ \text{log\_softmax}(x_i) = x_i - m - \log\left( \sum_{j} \exp(x_j - m) \right) $$ 
    where $m = \max(x, \text{axis} = -1)$.

2.  **Exponentiation:**
    $$ \text{softmax}(x_i) = \exp(\text{log\_softmax}(x_i)) $$ 

## Properties

*   **Numerical Stability:** By reusing the LogSoftmax formulation (specifically the subtraction of the maximum value $m$), this implementation avoids intermediate overflow in the exponential calculation. The largest value passed to $\exp$ inside the log-sum-exp is 0.
*   **Normalization:** The output is a probability distribution over the last axis, such that $\sum \text{softmax}(x) \approx 1$.
*   **Identity:** $\text{StableSoftmax}(x) \equiv \exp(\text{LogSoftmax}(x))$.

## Relationship to Naïve Softmax

*   **Naïve Softmax:** Computes $\frac{\exp(x_i)}{\sum \exp(x_j)}$. This is prone to overflow if any $x_i$ is large (e.g., > 88 for FP32).
*   **Stable Softmax:** Computes via the log domain. It handles large positive inputs gracefully. It is computationally more expensive due to the extra `Log` and `Exp` calls but is strictly more robust.

## Semantics & Constraints

*   **Precision:** All operations are performed in **FP32** (Single Precision).
*   **Composition:** This operation is expanded at graph construction time into:
    *   `LogSoftmax` (itself expanded into ReduceMax, Sub, Exp, ReduceSum, Log, Sub)
    *   `Exp` (element-wise)
*   **Traceability:** The execution trace will explicitly show the full sequence:
    `ReduceMax` → `Sub` → `Exp` → `ReduceSum` → `Log` → `Sub` → `Exp`

## Non-Goals

*   **Performance:** This is a reference implementation prioritizing stability and correctness over throughput.
*   **Fused Kernel:** There is no `OpType::StableSoftmax`.
*   **Training:** Gradients are not supported.
