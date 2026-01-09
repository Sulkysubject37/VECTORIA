# VECTORIA Core

The heart of the VECTORIA framework, containing the IR, Execution Engine, and Reference Kernels.

## Structure
- `include/vectoria/`: Public headers.
- `src/`: Engine implementation.
- `src/kernels/`: Reference C++ implementations (Truth).
- `src/graph/`: Composed high-level operations (e.g., Softmax).
- `src/lowering/`: Deployment bridges (e.g., CoreML).
- `tests/`: Internal C++ validation suite.

## Design
The Core is written in modern C++17. It enforces:
1. **Static Scheduling**: All memory is pre-allocated in an arena.
2. **Deterministic Dispatch**: Kernel selection is based on strict policies.
3. **Observability**: Every significant action is logged to a `Tracer`.

## Building
Core is typically built as part of the shared library:
```bash
g++ -std=c++17 -shared -fPIC -Iinclude src/*.cpp src/kernels/*.cpp src/graph/*.cpp src/lowering/*.cpp -o libvectoria.dylib
```
