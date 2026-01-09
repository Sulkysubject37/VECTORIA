# VECTORIA Python Bindings

Python control surface for the VECTORIA engine.

## Usage
```python
from vectoria import Graph, DType
from vectoria.runtime import Runtime

# 1. Build Graph
g = Graph()
x = g.add_input("X", [3], DType.FLOAT32)
sm = g.add_softmax(x)
g.set_output(sm)

# 2. Run
rt = Runtime()
rt.load_graph(g)
rt.set_input(x.id, [1.0, 2.0, 3.0])
rt.execute()

# 3. Results & Traces
print(rt.get_output(sm.id, 3))
for event in rt.get_trace():
    print(event)
```

## Requirements
- Python 3.8+
- `libvectoria.dylib` must be in the library path or project root.
- `pytest` for running tests.

## Development
Run tests:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest tests/
```
