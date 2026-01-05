# CoreML Lowering Strategy

This document outlines the design for exporting VECTORIA graphs to Apple's CoreML format (`.mlmodel` / `.mlpackage`).

## Semantic Mapping
To ensure determinism, we map VECTORIA operations to specific CoreML layers that offer the closest mathematical behavior.

| VECTORIA Op | CoreML / MIL Op | Notes |
|-------------|-----------------|-------|
| `MatMul(A, B)` | `linear` / `matmul` | Uses CoreML's default linear algebra path. |
| `BiasAdd(In, B)` | `add` | Broadcast handled by CoreML. |
| `Relu(In)` | `relu` | standard ReLU. |

## Determinism Risks
1. **Hardware Acceleration**: CoreML may choose between CPU, GPU (Metal), or ANE (Apple Neural Engine).
2. **ANE Rounding**: The ANE often uses FP16 or quantized logic, which will **BREAK** bitwise determinism against VECTORIA's FP32 reference.
3. **Implicit Fusion**: CoreML perform layer fusion (e.g., Conv+ReLU). This is opaque to VECTORIA.

## Lowering Guarantees
- **Reference as Truth**: Any CoreML export must be validated against the C++ Reference execution on the same device.
- **Explicit Target**: Users must specify if they want `MIL` (Model Intermediate Language) or `neuralnetwork` format.
- **Fail-Fast**: Ops not supported by MIL (e.g., future custom kernels) will result in export failure, not fallback.

## Validation Strategy
1. Export Graph to `.mlpackage`.
2. Load via `CoreML.framework` in Swift.
3. Execute with same inputs.
4. Compare output against `VectoriaEngine` (Reference Path).
5. Error > `1e-4` results in "Non-Deterministic Model" warning.

## Non-Goals
- **ANE Optimization**: We do NOT guarantee ANE compatibility if it requires lossy quantization.
- **Swift-to-CoreML JIT**: Lowering is a static export process.
