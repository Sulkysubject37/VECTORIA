# LogSoftmax (Semantic Specification)

**OpType:** Composed (No intrinsic kernel)
**Status:** Active (Reference Execution Only)

## Definition

LogSoftmax is defined over the **last axis** of the input tensor. It computes the logarithm of the softmax function, but does so in a numerically stable way by using the "Log-Sum-Exp" trick.

Given input tensor $x$ of shape $[..., D]$:

Let $m = \max(x, \text{axis} = -1)$

Then:
$$ \text{log\_softmax}(x_i) = x_i - m - \log\left( \sum_{j} \exp(x_j - m) \right) $$

where the summation is over the last axis.

Properties:
*   **Numerical Stability:** Subtracting the maximum value $m$ ensures that the largest argument to $\exp$ is 0, preventing overflow. The result of the sum will be at least 1 (since $e^0 = 1$), so the argument to $\log$ is $\ge 1$, preventing underflow and domain errors (log of negative/zero).
*   **Normalization:** The operation normalizes the input over the last dimension $D$.
*   **Identity:** $\sum \exp(\text{log\_softmax}(x)) \approx 1$.

## Semantics & Constraints

*   **Precision:** All operations are performed in **FP32** (Single Precision).
*   **Composition:** This operation is expanded at graph construction time into the following sequence of primitives:
    1.  `ReduceMax` (to find $m$)
    2.  `Sub` (to compute $x - m$, broadcasted)
    3.  `Exp` (element-wise)
    4.  `ReduceSum` (sum of exponentials)
    5.  `Log` (element-wise)
    6.  `Sub` (final subtraction: $(x - m) - \text{log\_sum}$, broadcasted)
*   **Broadcasting:**
    *   $m$ (shape $[..., 1]$ or reduced rank) is broadcast to subtract from $x$.
    *   The log-sum term (reduced rank) is broadcast to subtract from the shifted input $x - m$.

## Determinism

*   This operation relies on the determinism of `ReduceMax`, `ReduceSum`, `Exp`, `Log`, and `Sub`.
*   Reduction order is strictly defined by the reference implementation of reduction kernels.
*   No approximate math instructions are used.

## Non-Goals

*   **Performance:** This is a reference implementation composed of multiple passes over memory. It is not optimized for cache locality or fused execution.
*   **Fused Kernel:** There is no `OpType::LogSoftmax`.
*   **Training:** Gradients are not supported.
