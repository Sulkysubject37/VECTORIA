# VECTORIA Testing Philosophy

## Determinism Over Coverage
In VECTORIA, a test that passes "most of the time" is a failed test. We prioritize **deterministic correctness** over broad statistical coverage.

### Rules
1. **No Random Seeds**: Tests must use hard-coded, known-good inputs and outputs.
2. **Bitwise Checks**: Where possible, check for exact bitwise equality. For floating point, use extremely strict epsilon bounds derived from numerical analysis, not arbitrary "close enough" values.
3. **Loud Failures**: Tests must exit with a non-zero code immediately upon the first mismatch.

## Integration Tests
Integration tests (like `core/tests/test_gemm.cpp`) verify the entire stack:
1. Graph Construction (IR)
2. Memory Allocation (Arena)
3. Scheduling (Engine)
4. Kernel Execution (Reference/ASM)

## Unit Tests
Unit tests target specific components (e.g., `Arena` logic, `IR` validation) in isolation.
