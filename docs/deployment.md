# VECTORIA Deployment & CoreML Export

VECTORIA provides a strict, proven path for deploying research models to production environments via Apple CoreML.

## Philosophy: Deployment as a Contract
We treat deployment not as a "feature" but as a **contract**. If you set the execution mode to `Deployment`, VECTORIA guarantees:
1.  **Semantic Equivalence**: The exported graph will behave mathematically identically to the C++ reference execution (within documented FP32/FP16 tolerances).
2.  **Structural Validity**: The graph contains only operations supported by the target backend.
3.  **Traceability**: The decision to lower specific kernels is recorded in the execution trace.

## Supported Operations
The following kernels are certified for CoreML export:
*   `MatMul`
*   `Add`, `Sub`, `Mul`, `Div` (Elementwise & Broadcast)
*   `ReLU`
*   `ReduceSum`, `ReduceMax` (Last axis)
*   `Softmax` (Exported as a composed subgraph of primitives, or fused if structurally identical)

## Export Process
### C++
```cpp
#include "vectoria/lowering/coreml.hpp"
// ... build graph ...
vectoria::lowering::export_to_coreml(graph, "MyModel.mlpackage");
```

### Swift
```swift
try graph.exportToCoreML(path: "MyModel.mlpackage")
```

## Numerical Guarantees
*   **Precision**: Exports typically use FP32 or FP16 depending on the CoreML configuration.
*   **Tolerance**: We validate equivalence to `1e-4` on CPU/GPU. ANE (Neural Engine) may exhibit larger drift due to quantization.

## Limitations
*   **No Training**: Export is inference-only.
*   **No Custom Layers**: We do not inject custom C++ kernels into CoreML. We strictly map to standard MIL ops to ensure portability.
*   **Static Shapes**: Dynamic shapes are not currently supported in the export path.
