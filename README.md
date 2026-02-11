# VECTORIA

**Deterministic, Inspectable, High-Performance Computational Kernels**

![ARM64 NEON](https://img.shields.io/badge/ARM64_NEON-Validated-success)
![x86_64 AVX2](https://img.shields.io/badge/x86__64_AVX2-Validated-success)
![CoreML Export](https://img.shields.io/badge/CoreML_Export-Active-blue)

VECTORIA is a deterministic, cross-platform computational kernel framework designed as the robust "engine room" for higher-level applications requiring absolute control over numerical execution and memory layout.

It prioritizes **correctness over peak throughput** and **inspectability over magic**.

Current Version: **v1.3.2-stable (Semantic Surface Frozen)**

## 🛡️ Semantic Freeze & Stability

As of version v1.3.2-stable, the semantic surface of VECTORIA is considered **frozen**. 
The mathematical definitions of all operations (Primitive and Composed) are locked to ensure absolute reproducibility for downstream applications. 
Future additions or changes to these semantics will require explicit, versioned opt-in to prevent silent numerical drift.

All execution behavior is governed by [TRUTH.md](TRUTH.md).

## 🏛 Manifesto

For a formal definition of "truth," "determinism," and our constitutional guarantees, see [TRUTH.md](TRUTH.md).
This document is the authoritative source for the project's numerical philosophy and non-negotiable boundaries.

## 🚀 Key Features

- **Strict Determinism**: Bitwise identical results across repeated runs on the same hardware.
- **Cross-Platform Portability**: Verified execution on macOS (Apple Silicon) and Linux (x86_64).
- **CoreML Lowering**: Export semantic graphs to `.mlpackage` while maintaining mathematical equivalence.
- **Explicit Memory Model**: Arena-based allocation with predictable lifetimes and no garbage collection.
- **Auditable Execution**: Full tracing of every kernel dispatch, memory allocation, and deployment decision.
- **Semantic Truth**: All SIMD kernels are validated against bit-exact C++ reference implementations.

### Composed Semantic Operations
- **LayerNorm**: Stable, broadcast-aware normalization.
- **LogSoftmax**: Numerically stable log-probability computation.
- **StableSoftmax**: Robust probability distribution (recommended over naïve Softmax).
- **CrossEntropy**: Inference-only evaluation metric.
- **Attention**: Scaled dot-product semantic expansion.
- **TransformerEncoder**: Full encoder block including MHA and FFN.

## 🛠 Architecture

VECTORIA is built in strict layers:

| Layer | Responsibility | Languages |
|-------|----------------|-----------|
| **Frontend** | Graph construction, inspection | Python, Swift |
| **Core** | IR, Scheduling, Memory, Validation | C++17 |
| **Lowering** | CoreML/MIL Export | C++17 |
| **Kernels** | SIMD computation (SGEMM, etc.) | Assembly (AVX2, NEON) |

### Platform Parity & Validation
| Architecture | Implementation | CI Validation | Reproducibility |
|--------------|----------------|---------------|-----------------|
| **ARM64 (NEON)** | Active | **Full** (macos-latest) | Bitwise (Intra-platform) |
| **x86_64 (AVX2)** | Active | **Full** (ubuntu-latest) | Bitwise (Intra-platform) |
| **Reference** | Active | **Full** | **Bitwise (Cross-platform)** |

*Note: SIMD results may drift cross-platform due to hardware FMA differences. Use the `Reference` policy for absolute cross-arch bitwise identity.*


### Supported Operations (All Bindings)
- `MatMul`: Matrix Multiplication (FP32)
- `Add`, `Sub`, `Mul`, `Div`: Elementwise & Vector Broadcast
- `ReLU`: Rectified Linear Unit
- `ReduceSum`, `ReduceMax`: Last-axis reductions
- `Softmax`: High-stability composed implementation (LogSoftmax + Exp)
- `LayerNorm`, `CrossEntropy`: High-level semantic blocks
- `Attention`, `TransformerEncoder`: Transformer-grade semantic units

## 📦 Installation & Usage

### Python (v1.3.2)
```bash
pip install vectoria
```
Includes `vectoria-trace` CLI for introspection.

### Swift
Add `https://github.com/Sulkysubject37/VECTORIA.git` to your SPM dependencies.

### Prerequisites
- C++17 Compiler (GCC/Clang)
- Python 3.8+ (for bindings)
- Swift 5.5+ (for Apple platforms)

### Building from Source
```bash
# Standard Build
g++ -std=c++17 -shared -fPIC -Icore/include \
    core/src/*.cpp core/src/kernels/*.cpp core/src/graph/*.cpp core/src/lowering/*.cpp \
    -o libvectoria.dylib

# Build with SIMD Optimizations (ARM64/AVX2)
# Ensure correct assembly file for your architecture
g++ -std=c++17 -shared -fPIC -DVECTORIA_USE_ASM -Icore/include \
    core/src/*.cpp core/src/kernels/*.cpp core/src/graph/*.cpp core/src/lowering/*.cpp \
    asm/arm64/gemm_neon.S -o libvectoria.dylib
```

### Python Example
```python
from vectoria import Graph, DType
from vectoria.runtime import Runtime

# Build Graph
g = Graph()
x = g.add_input("X", [2, 2], DType.FLOAT32)
w = g.add_parameter("W", [2, 2], DType.FLOAT32, 0)
op = g.add_op("MatMul", [x, w], [2, 2], DType.FLOAT32)
g.set_output(op)
g.compile()

# Execute
rt = Runtime()
rt.load_graph(g)
rt.set_input(x.id, [1.0, 0.0, 0.0, 1.0])
rt.set_input(w.id, [0.5, 0.5, 0.5, 0.5])
rt.execute()

# Inspect Trace
for event in rt.get_trace():
    print(event)
```

### Tooling & Introspection
- [**Official Wiki**](https://github.com/Sulkysubject37/VECTORIA/wiki) - The authoritative guide on execution, kernels, and determinism.
- [Trace Analysis & Tooling](docs/tooling.md)
- [Trace Schema](docs/trace_schema.md)
- [Intermediate Representation (IR)](docs/ir.md)
- [Memory Model](docs/memory_model.md)
- [Architecture & ABI](docs/architecture.md)
- [Kernels & Optimization](docs/kernels.md)
- [Determinism Guarantees](docs/determinism.md)
- [Python API](docs/python_api.md)
- [Swift Parity](docs/swift_parity.md)
- [Release Policy](docs/release_policy.md)
- [CI & Validation](docs/ci.md)
- [Performance Model](docs/performance_model.md)
- [Deployment & CoreML](docs/deployment.md)

## ⚠️ Philosophy (What this is NOT)
- **Not an ML Framework**: No auto-grad, no optimizers.
- **Not a Black Box**: No heuristics, no hidden threads.
- **Not "Fastest at all Costs"**: We will disable SIMD if it drifts from the reference.

## Copyright & License

Copyright (c) 2026 Sulkysubject37.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
