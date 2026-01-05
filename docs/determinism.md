# VECTORIA Determinism Audit

**Last Updated**: 2026-01-05
**Scope**: Core Engine & Reference Kernels

## Guarantees

### 1. Architectural Determinism
VECTORIA guarantees that the **Graph Topology** and **Execution Schedule** are identical for the same input graph construction sequence.
- **Mechanism**: The IR is frozen before execution. The `Engine` uses a deterministic topological sort.
- **Implication**: Memory allocation patterns (address offsets) are identical across runs.

### 2. Bitwise Reproducibility (Reference Kernels)
When using **Reference Kernels** (default), VECTORIA guarantees bitwise identical floating-point results across:
- **Time**: Repeated runs on the same machine.
- **Space**: Runs on different machines with IEEE 754 compliant FPU.
- **Mechanism**: Strictly serial execution. No parallel reduction trees. Fixed accumulation order (Row-Major).

### 3. Memory Layout
The `Arena` allocator ensures that internal buffer offsets are deterministic.
- **Guarantee**: If Input A is at offset `0x0` and Input B is at `0x1000` in Run 1, they will be at the same relative offsets in Run 2.

## Limitations & Non-Guarantees

### 1. Assembly Kernels (Opt-in)
When `VECTORIA_USE_ASM` is enabled:
- **Cross-Platform**: Results may differ between x86_64 (AVX2) and ARM64 (NEON) due to different FMA (Fused Multiply-Add) behaviors or instruction implementations.
- **Intra-Platform**: Results are deterministic on the same CPU.

### 2. Floating Point Associativity
Vectoria does **NOT** use Kahan summation or compensated algorithms in the default kernels. 
- Large matrix multiplications may suffer from standard accumulation error.
- However, this error is *deterministic* (always the same wrong value).

### 3. Concurrency
VECTORIA currently executes **sequentially** on a single thread. 
- If threading is introduced in Phase 3, strict rules will be applied to maintain determinism (e.g., deterministic work stealing or static partitioning).

## Checklist for Contributors
- [ ] Do not use `std::unordered_map` for anything affecting execution order.
- [ ] Do not use thread-local storage for state that affects results.
- [ ] Do not use `rand()` or `time()` inside kernels.
