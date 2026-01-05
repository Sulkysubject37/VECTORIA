# VECTORIA

**VECTORIA** is a deterministic, inspectable, cross-platform computational kernel framework. 

It is designed to be the robust "engine room" for higher-level applications that require absolute control over numerical execution and memory layout.

## ⚠️ What VECTORIA is NOT
- **It is NOT an ML Framework**: It does not replace PyTorch or TensorFlow. It is a lower-level kernel runner.
- **It is NOT an Auto-Grad Library**: It is a forward-only execution engine.
- **It is NOT "Magic"**: There is no dynamic dispatch, no JIT compilation, and no garbage collection in the hot path.

## Core Invariants
1. **Determinism**: The same graph + same inputs = identical bitwise output, always.
2. **Explicit Memory**: We use an Arena-based memory model. Allocations are pre-planned and contiguous.
3. **Production-Grade**: Built for stability and predictability, not for rapid prototyping.

## Architecture

| Layer | Responsibility | Languages |
|-------|----------------|-----------|
| **Frontend** | Graph construction, inspection | Python, Swift |
| **Core** | IR, Scheduling, Memory, Validation | C++17 |
| **Kernels** | SIMD computation (GEMM, etc.) | Assembly (AVX, NEON) |

## Documentation
- [Intermediate Representation (IR)](docs/ir.md)
- [Memory Model](docs/memory_model.md)
- [Architecture & ABI](docs/architecture.md)
- [Kernels](docs/kernels.md)
- [Determinism Audit](docs/determinism.md)
- [Python API](docs/python_api.md)
- [Testing Philosophy](docs/testing.md)
- [Release Policy](docs/release_policy.md)

## License
MIT License. See [LICENSE](LICENSE) for details.

## Citation
MD. Arshad, (c) Sulkysubject37, 2026

