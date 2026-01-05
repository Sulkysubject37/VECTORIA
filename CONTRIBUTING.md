# Contributing to VECTORIA

## Core Philosophy
VECTORIA is **infrastructure**, not a playground. We value:
1. **Determinism over Cleverness**: If an optimization makes execution order non-deterministic, it is rejected.
2. **Explicit Semantics**: No magic. Memory, types, and ownership must be obvious.
3. **Strict Boundaries**: Know which layer (Asm, C++, Python, Swift) you are in and respect its limitations.

## Governance & workflow

### Commit Discipline
- **Atomic Commits**: Each commit must do one thing (e.g., "fix bug", "add feature", "update docs").
- **Clear Messages**: Use the format `scope: message` (e.g., `core: fix memory leak`, `docs: update ABI spec`).
- **No Broken Builds**: Every commit must compile.

### Documentation
- **Code with Docs**: New features must include updating the relevant `docs/*.md` file.
- **Why, Not What**: Comments should explain *why* a decision was made, not just describe the code.

### Optimization Policy
See [Optimization Governance](docs/optimization_policy.md) for strict rules on adding SIMD kernels.

### CI & Merge Policy
See [CI Failure Policy](docs/ci_policy.md) for rules on regression and unvalidated code.

### Testing
- **determinism**: Tests must verify that results are identical across runs.
- **Valgrind/ASAN**: Memory safety is paramount. Run checks before submitting.

## Pull Requests
1. Describe the problem and solution clearly.
2. Link to relevant issues.
3. Ensure all CI checks pass.
