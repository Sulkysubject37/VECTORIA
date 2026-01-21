# Tracing and Observability

**Purpose:** Explains the semantic meaning of traces and their use in auditing execution.

Tracing in VECTORIA is not just for debugging; it is a feature for **execution provenance**. It answers the question: *"How exactly was this result computed?"*

## Trace Events

The `Tracer` captures a linear log of events (`trace::EventType`) during execution:
*   **GraphCompilation:** Validation status and memory allocation sizing.
*   **NodeExecutionStart/End:** Strict timing boundaries for each node.
*   **KernelDispatch:** The specific kernel variant used (e.g., `SIMD [ARM64]` vs `Reference`).

## What is Logged

*   **Dispatch Mode:** Explicit confirmation of SIMD usage or fallback.
*   **Timestamps:** Nanosecond-precision duration (wall clock).
*   **Inputs:** Node IDs participating in the operation.

## Canonical Walkthrough: Transformer Encoder
Executing a Transformer Encoder block reveals the full semantic expansion in the trace:
1.  **Projections**: `KernelDispatch` for Query, Key, and Value MatMuls.
2.  **Attention Core**: Sequential events for `MatMul` (Scores), `Mul` (Scale), `StableSoftmax`, and `MatMul` (Context).
3.  **FFN**: Trace shows the `MatMul` → `BiasAdd` → `ReLU` → `MatMul` → `BiasAdd` chain.
4.  **Residual connections**: `Add` events clearly linking block inputs to sub-block outputs.

## What is NOT Logged

*   **Tensor Values:** To prevent performance degradation and log bloat, actual numerical data is never written to the trace.
*   **Internal State:** Loop counters or register states are not exposed.

## Execution Modes

*   **Research Mode:** Full tracing enabled.
*   **Deployment Mode:** Tracing remains active but strict validation is enforced (see Deployment Mode).

## References

*   `core/src/trace.cpp`
*   [docs/observability.md](../observability.md)