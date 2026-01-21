# VECTORIA Intermediate Representation (IR)

## Design Goals
1. **Determinism**: The IR represents a static computation graph. The execution order is derived deterministically from the graph topology.
2. **Inspectability**: The IR structure is simple and flat, allowing for easy serialization and debugging.
3. **Explicit Semantic**: Every node has a fixed shape, data type, and role (Input, Parameter, or Operation).
4. **Static Scheduling**: The IR provides enough information for the memory model to perform pre-allocation and lifetime analysis.

## What IR Explicitly Does NOT Do
1. **No In-place Mutation**: Once a graph is defined, it is immutable. Optimizations that reuse buffers are handled at the execution/memory layer, not by mutating the IR.
2. **No Dynamic Shapes**: All tensor dimensions must be known at graph construction time.
3. **No Auto-Differentiation**: VECTORIA is a forward-only execution kernel framework.
4. **No Control Flow**: The IR is a Directed Acyclic Graph (DAG) without loops or conditional branching.

## Operations
VECTORIA IR supports a strictly defined set of operations:
- **Numerical**: `MatMul`, `Add`, `Sub`, `Mul`, `Div`, `ReLU`, `Exp`, `Log`, `Sqrt`.
- **Reductions**: `ReduceSum`, `ReduceMax` (Last-axis).
- **Structural**: `Transpose`, `Reshape`, `Concat`, `Slice`.
- **Composed**: `LayerNorm`, `Softmax`, `Attention`, `MHA`, `TransformerEncoder`.

## Buffer Ownership
The IR nodes do not own the raw data buffers. `ParameterNode` contains a `buffer_id` which the `MemoryModel` resolves to physical memory. `InputNode` buffers are provided at execution time.
