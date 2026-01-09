import pytest
import math
from vectoria import Graph, DType
from vectoria.runtime import Runtime

def test_softmax_py():
    g = Graph()
    x = g.add_input("X", [3], DType.FLOAT32)
    op = g.add_softmax(x)
    g.set_output(op)
    
    rt = Runtime()
    rt.load_graph(g)
    
    # Input: [1.0, 2.0, 3.0]
    rt.set_input(x.id, [1.0, 2.0, 3.0])
    rt.execute()
    
    out = rt.get_output(op.id, 3)
    
    e1 = math.exp(1.0)
    e2 = math.exp(2.0)
    e3 = math.exp(3.0)
    s = e1 + e2 + e3
    
    assert math.isclose(out[0], e1/s, rel_tol=1e-5)
    assert math.isclose(out[1], e2/s, rel_tol=1e-5)
    assert math.isclose(out[2], e3/s, rel_tol=1e-5)
