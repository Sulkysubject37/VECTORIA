# Layer Normalization (Semantic Specification)

**OpType:** Composed (No intrinsic kernel)
**Status:** Active (Reference Execution Only)

## Definition

Layer Normalization (LayerNorm) is defined over the **last axis** of the input tensor. It normalizes the input to have zero mean and unit variance, then applies a learnable affine transformation.

Given input tensor $x$ of shape $[..., D]$:

1.  **Mean:** $\mu = \frac{1}{D} \sum_{i=0}^{D-1} x_i$
2.  **Variance:** $\sigma^2 = \frac{1}{D} \sum_{i=0}^{D-1} (x_i - \mu)^2$
3.  **Normalization:** $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
4.  **Affine Transform:** $y = \hat{x} \cdot \gamma + \beta$

Where:
*   The reduction axis is strictly the last dimension ($D$).
*   $\epsilon$ (epsilon) is a fixed scalar constant: **1e-5**.
*   $\gamma$ (gamma) and $\beta$ (beta) are learnable parameters of shape $[D]$.
*   $\cdot$ denotes element-wise multiplication (broadcasting applies if necessary, but $\gamma, \beta$ match the last dim).

## Semantics & Constraints

*   **Precision:** All operations are performed in **FP32** (Single Precision).
*   **Variance:** The population variance formula is used (divide by $N$, not $N-1$).
*   **Composition:** This operation is not a single kernel. It is expanded at graph construction time into the following sequence of primitives:
    *   `ReduceSum`
    *   `Sub`
    *   `Mul` (and `Div` by constant $N$)
    *   `Sqrt`
    *   `Add`
    *   `Div`

## Broadcasting

*   The statistics $\mu$ and $\sigma^2$ are computed per-sample (reducing the last axis).
*   They are broadcast back to the original shape $[..., D]$ for subtraction/division.
*   $\gamma$ and $\beta$ are broadcast from $[D]$ to $[..., D]$.

## Determinism

*   This operation relies on the determinism of its constituent parts.
*   Since `ReduceSum` is used, the order of summation is fixed by the reference implementation.
*   No fused "fast math" approximations (like `rsqrt`) are used; strict `1.0 / sqrt(x)` semantics apply.

## Non-Goals

*   **Performance:** This implementation is for correctness and semantic validation. It generates multiple intermediate kernels and memory accesses.
*   **Fused Kernel:** There is no `OpType::LayerNorm`.
*   **Training:** Gradients are not supported.
