import pytest
import numpy as np
from vectoria import Graph, Runtime, DType

def test_transpose_reshape():
    g = Graph()
    x_shape = [2, 3]
    perm = [1, 0]
    
    in_node = g.add_input("Input", x_shape, DType.FLOAT32)
    t_node = g.add_transpose(in_node, perm)
    r_node = g.add_reshape(t_node, [6])
    
    g.set_output(r_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    rt.set_input(in_node.id, x_val.flatten().tolist())
    rt.execute()
    
    out = rt.get_output(r_node.id, 6)
    out_np = np.array(out)
    
    expected = x_val.transpose(perm).reshape([6])
    
    np.testing.assert_allclose(out_np, expected, atol=1e-5)
