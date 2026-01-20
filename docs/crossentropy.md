# CrossEntropy (Semantic Specification)

**OpType:** Composed (No intrinsic kernel)
**Status:** Active (Inference-Only, Reference Execution)

## Definition

CrossEntropy is defined for **inference and evaluation purposes only**. It measures the divergence between a predicted probability distribution (logits) and a target probability distribution.

Given:
*   **Logits** tensor $x$ of shape $[..., D]$
*   **Target** tensor $t$ of shape $[..., D]$ (representing probabilities or one-hot encoding)

The operation is defined as:

1.  **LogSoftmax:**
    $$
    \text{log\_probs}(x) = \text{LogSoftmax}(x)
    $$

2.  **Cross Entropy:**
    $$
    \text{CrossEntropy}(x, t) = - \sum_{i} (t_i \cdot \text{log\_probs}(x)_i)
    $$

Where the summation is over the last axis ($D$). 

## Properties

*   **Inference Only:** This operation does not support gradients, backward passes, or training semantics. It is strictly for forward-pass evaluation.
*   **Numerical Stability:** By building upon the stable `LogSoftmax` implementation (which uses the log-sum-exp trick), this operation avoids underflow associated with computing $\log(\text{Softmax}(x))$.
*   **Target Format:** The target $t$ is expected to be a valid probability distribution (summing to 1 over the last axis) or a one-hot vector. The implementation does not strictly enforce normalization but computes the weighted sum as defined.
*   **Output Shape:** The reduction occurs over the last axis, resulting in a tensor of rank $N-1$. For a single sample input $[D]$ and target $[D]$, the output is a scalar.

## Semantics & Constraints

*   **Precision:** All operations are performed in **FP32** (Single Precision).
*   **Composition:** This operation is expanded at graph construction time into:
    *   `LogSoftmax` (ReduceMax, Sub, Exp, ReduceSum, Log, Sub)
    *   `Mul` (element-wise multiplication of target and log-probs)
    *   `ReduceSum` (summation over last axis)
    *   `Mul` (by scalar -1.0) or `Sub` (0 - sum) to negate
*   **Broadcasting:**
    *   Logits and Targets must have compatible shapes for element-wise multiplication. Typically, they are identical shapes.
    *   No implicit batch reduction (e.g., "mean over batch") is performed. The output preserves the batch dimensions.

## Determinism

*   This operation relies on the determinism of `LogSoftmax` and `ReduceSum`.
*   Reduction order is strictly defined by the reference implementation.

## Non-Goals

*   **Training:** No backpropagation support.
*   **Sparse Targets:** This implementation does not support integer class indices (sparse labels). Targets must be dense tensors of the same shape as logits.
*   **Performance:** This is a reference implementation composed of multiple passes over memory.
