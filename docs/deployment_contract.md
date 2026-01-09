# Deployment Execution Contract

VECTORIA introduces `ExecutionMode::Deployment` to ensure that graphs intended for export (e.g., CoreML) adhere to strict semantic and structural constraints.

## Purpose
Research mode prioritizes flexibility and inspectability. Deployment mode prioritizes **compatibility** and **predictability**. A graph that compiles in Deployment mode is guaranteed to be exportable to supported backends with known semantics.

## Constraints
When `EngineConfig.mode` is set to `Deployment`:

1.  **Restricted Op Set**: Only the following operations are allowed:
    *   `MatMul`
    *   `BiasAdd`
    *   `ReLU`
    *   `Add`, `Mul`, `Sub`, `Div`
    *   `ReduceSum`, `ReduceMax` (Last axis only)
    *   `Exp` (As part of Softmax)

2.  **Explicit Failure**: The engine will throw an exception during `compile()` if an unsupported operation is encountered. No silent fallback or omission is permitted.

3.  **Traceability**: The trace log will explicitly record validation success or failure, providing an audit trail for why a model was deemed deployable (or not).

## Workflow
1.  Build Graph in Python/Swift.
2.  Configure Engine with `ExecutionMode::Deployment`.
3.  Call `compile()`.
    *   **Success**: The graph is safe for `CoreML` export.
    *   **Failure**: The graph contains unsupported ops or shapes.
4.  Export using the Lowering API.
