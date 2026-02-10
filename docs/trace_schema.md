# Trace Schema

VECTORIA traces are represented as a JSON list of event objects.

## Event Object Fields

- `type` (string): One of `GraphCompilation`, `MemoryAllocation`, `NodeExecutionStart`, `NodeExecutionEnd`, `KernelDispatch`.
- `timestamp_ns` (integer): Nanosecond timestamp from system clock.
- `node_id` (integer): ID of the node associated with the event (-1 if not applicable).
- `details` (string): Metadata specific to the event type.

## Event Type Details

- **GraphCompilation**: Contains mode and phase info.
- **MemoryAllocation**: Contains allocation size in bytes.
- **NodeExecutionStart/End**: Boundary markers for node processing.
- **KernelDispatch**: Contains the kernel policy used (Reference vs. SIMD) and input node IDs.
