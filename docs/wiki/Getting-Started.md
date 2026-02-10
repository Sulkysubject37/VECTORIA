# Getting Started

Welcome to VECTORIA. This guide will help you build your first deterministic graph.

## 1. Installation

First, install the package:
```bash
pip install vectoria
```

## 2. Your First Graph

Build a simple Matrix Multiplication graph in Python:

```python
from vectoria import Graph, DType
from vectoria.runtime import Runtime

# 1. Define the computation
g = Graph()
x = g.add_input("X", [2, 2], DType.FLOAT32)
w = g.add_parameter("W", [2, 2], DType.FLOAT32, 0)
matmul = g.add_matmul(x, w, [2, 2], DType.FLOAT32)
g.set_output(matmul)
g.compile()

# 2. Execute with the Runtime
rt = Runtime()
rt.load_graph(g)
rt.set_input(x.id, [1.0, 0.0, 0.0, 1.0])
rt.set_input(w.id, [0.5, 0.5, 0.5, 0.5])
rt.execute()

# 3. Inspect results and trace
results = rt.get_output(matmul.id, 4)
print(f"Results: {results}")

for event in rt.get_trace():
    print(event)
```

## 3. Next Steps
- Learn about [Memory Layout](Memory-Model.md)
- Explore [Semantic Operations](Home.md)
- [Verify Determinism](Verifying-Determinism.md)
