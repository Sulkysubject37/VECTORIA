# TRUTH.md — Determinism & Numerical Truth in VECTORIA

**Version:** 1.0.0 (Normative)  
**Applicable to:** VECTORIA v1.3.0-stable and later  

---

## 1. Purpose of This Manifesto

This document defines the constitutional standards for numerical correctness and determinism within the VECTORIA computational kernel framework. It is a **normative** document, not a descriptive one. 

It explicitly defines what constitutes "truth" within the system, establishes the hierarchy of validation, and delineates the specific boundaries of guaranteed behavior.

**Rule:** If the behavior of the VECTORIA engine contradicts this document, the behavior is a bug, regardless of performance implications.

## 2. Definition of Numerical Truth

In VECTORIA, "truth" is distinct from "mathematical intent." 

*   **Mathematical Intent:** The abstract mathematical operation (e.g., matrix multiplication $C = A \times B$ in $\mathbb{R}$). 
*   **Numerical Truth:** The specific, bitwise-exact result produced by the execution of a concrete sequence of IEEE-754 floating-point operations (FP32).

VECTORIA defines Numerical Truth purely in terms of its execution model. Truth is not defined by an external framework (e.g., PyTorch, TensorFlow) nor by higher-precision emulations (FP64), but by the deterministic execution of its own Reference Standard.

## 3. The Gold Standard

The ultimate authority on correctness within VECTORIA is the **Scalar C++ Reference Implementation**.

*   **Location:** `core/src/kernels/*_ref.cpp`
*   **Type:** Serial, scalar, single-precision (FP32) C++.
*   **Properties:** 
    *   No SIMD instructions.
    *   Strictly defined loop order.
    *   Standard C++ arithmetic operators.

**Why this is the Gold Standard:**
1.  **Auditability:** The code is readable and maps directly to the algebraic definition of the operation.
2.  **Portability:** It executes with maximum consistency across different compilers and architectures, serving as the stable baseline.
3.  **Stability:** It is immune to platform-specific reordering or instruction fusion (except where allowed by standard C++ compliance).

Any optimized kernel (SIMD, Assembly) or lowered representation (CoreML) is considered "correct" only insofar as it deviates from this Reference Standard by an acceptable, bounded margin. The Reference Standard never deviates; it *is* the definition.

## 4. Determinism Guarantees

VECTORIA provides strict determinism guarantees rooted in its architecture (Immutable IR, Static Arena, Single-Threaded Execution).

**The Hard Guarantees:** 

1.  **Structural Determinism:**
    *   Given an identical `ir::Graph`, the `Engine` will *always* produce an identical topological sort and execution schedule.
    *   Reference: `core/src/engine.cpp`

2.  **Memory Determinism:**
    *   Given an identical schedule, the `MemoryManager` will *always* assign identical byte offsets within the `Arena` for every intermediate tensor.
    *   Reference: `core/include/vectoria/memory.hpp`

3.  **Intra-Platform Execution Determinism:**
    *   On the **same hardware architecture** (e.g., x86_64 AVX2), running the **same compiled binary** with the **same inputs** will produce **bitwise-identical output**.
    *   This holds regardless of system load, time of day, or operating system version, due to the single-threaded, arena-based design.

4.  **Traceability:**
    *   Every execution produces a trace that allows for the exact reconstruction of the operator sequence and memory state.
    *   Reference: `core/include/vectoria/trace.hpp`

## 5. Boundaries of Determinism

Determinism is not absolute across all boundaries. VECTORIA explicitly **DOES NOT GUARANTEE** bitwise identity across the following: 

1.  **Cross-Architecture Execution:**
    *   Results on ARM64 (NEON) are **not guaranteed** to be bitwise identical to results on x86_64 (AVX2).
    *   *Reason:* Differences in Fused Multiply-Add (FMA) instructions, reciprocal approximations, and register width.

2.  **Optimization Levels (Reference vs. SIMD):**
    *   Results from the `KernelPolicy::Reference` are **not guaranteed** to be bitwise identical to `KernelPolicy::SIMD`.
    *   *Reason:* SIMD implementations may change the associativity of reduction operations (e.g., tree reduction vs. linear accumulation), resulting in floating-point rounding differences.

3.  **Compiler Variations:**
    *   Binaries compiled with different compilers (Clang vs. GCC vs. MSVC) or different optimization flags (`-O2` vs `-O3`) are **not guaranteed** to be bitwise identical, though they must remain within tolerance.
    *   Determinism guarantees apply to a **fixed binary**. Recompilation with different compilers or flags is out of scope and constitutes a different system state.

## 6. Numerical Drift & Tolerance

Since floating-point arithmetic is not associative ($(a+b)+c \neq a+(b+c)$), differences between Reference and Optimized kernels are expected. This is defined as **Numerical Drift**.

VECTORIA manages drift via **Strict Tolerance Validation**:

*   **Validation Mechanism:** The `Validator` (`core/src/lowering/validation.cpp`) compares Optimized outputs against Reference outputs.
*   **Tolerance Definitions:** 
    *   Tolerance is not a redefinition of truth, but a check on the validity of an optimization.
    *   Tolerances are operation-specific (e.g., `Gemm` has a looser tolerance than `Add` due to accumulation depth).
    *   These tolerance checks apply universally: during offline kernel certification (CI) and runtime deployment validation.
*   **Drift Rejection:** If an optimized kernel produces drift exceeding the defined tolerance, the optimization is considered **invalid** and the system must fall back to Reference or fail certification.

## 7. Reference vs. Optimized Execution

The relationship between the Reference implementation and Optimized (SIMD/Assembly) implementations is hierarchical:

1.  **Reference:** The Semantic Authority. Defines *what* the value is.
2.  **SIMD/Optimized:** The Performance Approximation. Defines *how fast* we can get a result that is "close enough" to the Reference.

**Policy:**
*   SIMD kernels are an optimization detail.
*   If a conflict arises where an SIMD kernel is fast but numerically unstable (deviates from Reference), the SIMD kernel is **incorrect**.
*   VECTORIA prioritizes the stability of the Reference definition over the throughput of the SIMD implementation.

## 8. External Oracles (What They Are and Are Not)

VECTORIA utilizes external environments (Python/NumPy, Swift) for validation.

*   **Role:** These are **Oracles**. They are used to generate test cases, verify broad correctness, and detect gross logical errors during development (`python/tests/`).
*   **Limitation:** They are **NOT** the source of Numerical Truth for VECTORIA.
    *   If NumPy (using OpenBLAS/MKL/FP64) produces `1.000000001` and VECTORIA Reference produces `1.000000002`:
    *   **VECTORIA is correct by definition within its declared execution model**.
    *   The Oracle helps us ensure we aren't calculating a completely different function, but it does not dictate the specific LSB (Least Significant Bit) of the result.

## 9. Implications for Users

**Users CAN rely on:**
*   Bitwise reproducibility of their pipelines on a fixed deployment target (e.g., a fleet of identical servers or specific mobile devices).
*   A guarantee that no hidden state or dynamic scheduling will alter results between runs.
*   Full transparency into which kernels (Reference or SIMD) were executed via the Trace API.

**Users MUST validate:**
*   That FP32 precision is sufficient for their specific domain application.
*   That the numerical drift between their research environment (e.g., Python/Reference) and deployment environment (e.g., CoreML/SIMD) is acceptable for their specific use case.

## 10. Non-Goals and Explicit Rejections

To maintain the integrity of the system, VECTORIA explicitly rejects the following:

*   **REJECTED:** "Fastest at any cost." 
    *   We will not use fast approximate instructions (e.g., `rsqrt`) unless they are explicitly opted-in or strictly validated against the Reference.
*   **REJECTED:** Silent Numerical Changes.
    *   Updates to kernels must pass regression tests ensuring they do not inadvertently alter numerical behavior beyond accepted version-to-version drift.
*   **REJECTED:** Heuristic Scheduling.
    *   We do not reorder graphs based on runtime heuristics. The execution path is fixed at compile/graph-construction time.

VECTORIA values **Correctness, Determinism, and Observability** above raw throughput.
