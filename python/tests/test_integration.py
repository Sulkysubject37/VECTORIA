import pytest
import math
from vectoria import Graph, DType
from vectoria.runtime import Runtime

def test_matmul():
    g = Graph()
    x = g.add_input("X", [2, 2], DType.FLOAT32)
    w = g.add_parameter("W", [2, 2], DType.FLOAT32, 0)
    op = g.add_matmul(x, w, [2, 2], DType.FLOAT32)
    g.set_output(op)
    
    rt = Runtime()
    rt.load_graph(g)
    
    # Identity
    rt.set_input(x.id, [1.0, 0.0, 0.0, 1.0])
    rt.set_input(w.id, [1.0, 0.0, 0.0, 1.0])
    rt.execute()
    
    out = rt.get_output(op.id, 4)
    assert out == [1.0, 0.0, 0.0, 1.0]

def test_bias_add():
    g = Graph()
    x = g.add_input("X", [2, 2], DType.FLOAT32)
    b = g.add_parameter("B", [1, 2], DType.FLOAT32, 0)
    op = g.add_bias_add(x, b)
    g.set_output(op)
    
    rt = Runtime()
    rt.load_graph(g)
    
    rt.set_input(x.id, [1.0, 2.0, 3.0, 4.0])
    rt.set_input(b.id, [0.5, 1.0]) # Broadcasts to rows
    rt.execute()
    
    out = rt.get_output(op.id, 4)
    # Row 1: 1.0+0.5, 2.0+1.0 -> 1.5, 3.0
    # Row 2: 3.0+0.5, 4.0+1.0 -> 3.5, 5.0
    expected = [1.5, 3.0, 3.5, 5.0]
    for o, e in zip(out, expected):
        assert math.isclose(o, e, rel_tol=1e-5)

def test_relu():
    g = Graph()
    x = g.add_input("X", [2, 2], DType.FLOAT32)
    op = g.add_relu(x)
    g.set_output(op)
    
    rt = Runtime()
    rt.load_graph(g)
    
    rt.set_input(x.id, [-1.0, 0.0, 1.0, 2.0])
    rt.execute()
    
    out = rt.get_output(op.id, 4)
    expected = [0.0, 0.0, 1.0, 2.0]
    for o, e in zip(out, expected):
        assert math.isclose(o, e, rel_tol=1e-5)

def test_integration_chain():
    # MatMul -> BiasAdd -> ReLU
    g = Graph()
    x = g.add_input("X", [1, 2], DType.FLOAT32)
    w = g.add_parameter("W", [2, 2], DType.FLOAT32, 0)
    b = g.add_parameter("B", [1, 2], DType.FLOAT32, 0)
    
    mm = g.add_matmul(x, w, [1, 2], DType.FLOAT32)
    ba = g.add_bias_add(mm, b)
    relu = g.add_relu(ba)
    
    g.set_output(relu)
    
    rt = Runtime()
    rt.load_graph(g)
    
    # X = [1, -1]
    rt.set_input(x.id, [1.0, -1.0])
    
    # W = [[1, 2], [3, 4]]
    # X*W = [1*1 + (-1)*3, 1*2 + (-1)*4] = [1-3, 2-4] = [-2, -2]
    rt.set_input(w.id, [1.0, 2.0, 3.0, 4.0])
    
    # B = [1, 3]
    # BiasAdd = [-2+1, -2+3] = [-1, 1]
    rt.set_input(b.id, [1.0, 3.0])
    
    rt.execute()
    
    # ReLU = [0, 1]
    out = rt.get_output(relu.id, 2)
    expected = [0.0, 1.0]
    
    for o, e in zip(out, expected):
        assert math.isclose(o, e, rel_tol=1e-5)
