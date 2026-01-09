# VECTORIA Python API

The Python API provides two layers:
1. **Frontend (`vectoria.graph`)**: A pure Python graph builder.
2. **Runtime (`vectoria.runtime`)**: A wrapper around the C++ engine.

## Philosophy
Python is a **control surface**, not an execution engine.
- No heavy computation happens in Python.
- No automatic data conversion (e.g., from NumPy lists). Users must provide flat lists or raw bytes.
- Explicit lifecycle management.

## Minimal Runtime Bridge
The `Runtime` class loads `libvectoria.dylib` and mirrors the C++ Engine's lifecycle:
1. `load_graph(graph)`: Reconstructs the IR in C++.
2. `compile()`: Allocates C++ memory.
3. `set_input(id, data)`: Copies data to C++ buffers.
4. `execute()`: Runs kernels.
5. `get_output(id)`: Reads data back.

## Observability
You can inspect the execution trace after `execute()`:

```python
trace = runtime.get_trace()
for event in trace:
    print(event)
```

## Limitations
- **No NumPy**: We avoid strict dependencies for now.
- **Op Support**: `MatMul`, `BiasAdd`, and `ReLU` are supported.
- **Manual Mapping**: The bridge manually reconstructs the graph.