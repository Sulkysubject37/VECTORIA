# Deployment Mode and CoreML

**Purpose:** Details the contract for deploying graphs to constrained environments like Apple devices.

VECTORIA supports a specific **Deployment** execution mode designed to bridge research code with production mobile environments via CoreML (`.mlpackage`).

## Deployment Contract

When `ExecutionMode::Deployment` is active:
1.  **Strict Op Validation:** The `Engine` rejects any graph containing operations not supported by the CoreML backend.
2.  **Lowering Validation:** Before export, the graph topology is verified against MIL (Model Intermediate Language) constraints.

## Supported Operations

The following operations are validated for CoreML export:
*   **Arithmetic:** `Add`, `Sub`, `Mul`, `Div`
*   **Linear Algebra:** `MatMul`, `BiasAdd`
*   **Activation:** `Relu`, `Softmax` (via `Exp`)
*   **Reduction:** `ReduceSum`, `ReduceMax`

## Lowering to MIL

The lowering process translates the immutable IR directly into Apple's MIL format:
*   `ir::DataType::Float32` → `fp32`
*   `ir::OpType::MatMul` → `matmul(x=..., y=...)`
*   `ir::OpType::Relu` → `relu(x=...)`

## Fallback Behavior

If a graph contains an unsupported op (e.g., a custom experimental kernel), `export_to_coreml` will throw a runtime error. Partial export is not supported to ensure the deployed model behaves exactly as the research model.

## References

*   `core/src/lowering/coreml_lowering.cpp`
*   `core/src/lowering/validation.cpp`
*   [docs/deployment.md](../deployment.md)