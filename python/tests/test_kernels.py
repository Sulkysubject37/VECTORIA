import pytest
import math
from vectoria import Graph, DType
from vectoria.runtime import Runtime

def test_add():
    g = Graph()
    a = g.add_input("A", [4], DType.FLOAT32)
    b = g.add_input("B", [4], DType.FLOAT32)
    op = g.add_add(a, b)
    g.set_output(op)
    
    rt = Runtime()
    rt.load_graph(g)
    rt.set_input(a.id, [1.0, 2.0, 3.0, 4.0])
    rt.set_input(b.id, [10.0, 20.0, 30.0, 40.0])
    rt.execute()
    
    out = rt.get_output(op.id, 4)
    assert out == [11.0, 22.0, 33.0, 44.0]

def test_mul():
    g = Graph()
    a = g.add_input("A", [4], DType.FLOAT32)
    b = g.add_input("B", [4], DType.FLOAT32)
    op = g.add_mul(a, b)
    g.set_output(op)
    
    rt = Runtime()
    rt.load_graph(g)
    rt.set_input(a.id, [2.0, 3.0, 4.0, 5.0])
    rt.set_input(b.id, [0.5, 2.0, -1.0, 0.0])
    rt.execute()
    
    out = rt.get_output(op.id, 4)
    assert out == [1.0, 6.0, -4.0, 0.0]

def test_reduce_sum():
    g = Graph()
    x = g.add_input("X", [2, 3], DType.FLOAT32)
    op = g.add_reduce_sum(x)
    g.set_output(op)
    
    rt = Runtime()
    rt.load_graph(g)
    
    # [[1, 2, 3], [4, 5, 6]]
    rt.set_input(x.id, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rt.execute()
    
    out = rt.get_output(op.id, 2)
    # Sums: 1+2+3=6, 4+5+6=15
    assert math.isclose(out[0], 6.0, rel_tol=1e-5)
    assert math.isclose(out[1], 15.0, rel_tol=1e-5)
