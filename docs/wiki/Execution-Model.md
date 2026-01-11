# Execution Model

**Purpose:** Explains the immutable and static nature of the execution pipeline.

VECTORIA employs a strict **Compile-Schedule-Execute** lifecycle. Once a graph is compiled, its structure and memory footprint are frozen.

## 1. Immutable Intermediate Representation (IR)

The graph is defined using immutable structures:
*   **Graph:** A collection of `Node`s and `NodeId` outputs.
*   **Node:** A variant of `InputNode`, `ParameterNode`, or `OpNode`.
*   **OpType:** A strictly typed enum (e.g., `MatMul`, `Add`, `Relu`).

Once constructed, the `Graph` object is treated as a read-only specification by the Engine.

## 2. Static Scheduling

During `Engine::compile()`, the system:
1.  **Validates** the topology (checks for cycles and invalid inputs).
2.  **Linearizes** the graph into a fixed execution `schedule_` (`std::vector<size_t>`).
3.  **Calculates** the exact memory requirement for every tensor.

## 3. Arena Memory Model

Memory is managed by a linear `Arena` allocator.
*   **Allocation:** All necessary memory is allocated in a single block during `compile()`.
*   **Addressing:** Each node is assigned a fixed pointer (`node_buffers_[i]`) into this arena.
*   **Lifecycle:** Memory persists for the lifetime of the `Engine` and is reset only upon recompilation. This eliminates runtime allocation overhead and fragmentation.

## Execution Flow

The `execute()` method simply iterates through the pre-computed `schedule_`:

```text
[Start] -> [Fetch Node] -> [Resolve Inputs from Arena] -> [Dispatch Kernel] -> [Write Output to Arena] -> [Next Node]
```

## References

*   `core/include/vectoria/ir.hpp`
*   `core/src/engine.cpp`
*   `core/src/memory.cpp`