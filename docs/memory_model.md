# VECTORIA Memory Model

## Allocation Strategy
VECTORIA uses an **Arena-based allocation** model. 
- Memory is allocated in large contiguous blocks from the system.
- Sub-allocations are carved out of these blocks sequentially.
- There is no individual `free()` operation for sub-allocations.
- All memory is released simultaneously when the Arena is destroyed or reset.

## Lifetime Guarantees
- **Static Graph Lifetime**: Memory for parameters and constant tensors is allocated during graph initialization and persists for the lifetime of the `Engine` or the `Graph`.
- **Transient Session Lifetime**: Memory for intermediate activations is allocated at the start of an execution session and is reset after completion.

## Determinism
By using a bump-pointer allocator within an Arena, memory addresses and layout are deterministic for a given sequence of allocations. This minimizes fragmentation and provides predictable performance.

## Why GC and Reference Counting are Excluded
1. **Performance**: GC pauses and reference counting overhead are incompatible with deterministic low-latency kernel execution.
2. **Predictability**: Manual Arena management ensures that memory usage peaks are explicitly understood and controlled at development time.
3. **Simplicity**: Avoiding complex ownership graphs makes the C++ ↔ Assembly and C++ ↔ Swift boundaries much safer and easier to reason about.