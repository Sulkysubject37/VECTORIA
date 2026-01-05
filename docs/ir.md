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

## Example IR Graph (Pseudocode)

```python
# Conceptual representation of a simple Linear layer (MatMul + Add)

Node(0): Input(name="X", shape=[1, 784], dtype=F32)
Node(1): Parameter(name="W", shape=[784, 128], dtype=F32, buffer_id=0xABC)
Node(2): Parameter(name="B", shape=[1, 128], dtype=F32, buffer_id=0xDEF)

Node(3): Op(type=MatMul, inputs=[0, 1], shape=[1, 128], dtype=F32)
Node(4): Op(type=Add, inputs=[3, 2], shape=[1, 128], dtype=F32)

Outputs: [4]
```

## Buffer Ownership
The IR nodes do not own the raw data buffers. `ParameterNode` contains a `buffer_id` which the `MemoryModel` resolves to physical memory. `InputNode` buffers are provided at execution time.