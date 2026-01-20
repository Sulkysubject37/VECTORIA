import pytest
import numpy as np
from vectoria import Graph, Runtime, DType
from scipy.special import softmax

def test_softmax_stable_correctness():
    g = Graph()
    x_shape = [2, 5]
    
    in_node = g.add_input("Input", x_shape, DType.FLOAT32)
    s_node = g.add_softmax_stable(in_node)
    
    g.set_output(s_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    # Data: Random with some large values
    x_val = np.random.randn(*x_shape).astype(np.float32) * 10
    
    rt.set_input(in_node.id, x_val.flatten().tolist())
    rt.execute()
    
    out = rt.get_output(s_node.id, 2*5)
    out_np = np.array(out).reshape(x_shape)
    
    expected = softmax(x_val, axis=-1)
    
    np.testing.assert_allclose(out_np, expected, rtol=1e-4, atol=1e-5)

def test_softmax_stable_large_inputs():
    """Test with inputs that would overflow naive exp()"""
    g = Graph()
    x_shape = [1, 3]
    in_node = g.add_input("Input", x_shape, DType.FLOAT32)
    s_node = g.add_softmax_stable(in_node)
    g.set_output(s_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    # Inputs that would cause exp(1000) to overflow float32
    x_val = np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float32)
    
    rt.set_input(in_node.id, x_val.flatten().tolist())
    rt.execute()
    
    out = rt.get_output(s_node.id, 3)
    out_np = np.array(out)
    
    # Expected: 1/3
    expected = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float32)
    
    np.testing.assert_allclose(out_np, expected, rtol=1e-4, atol=1e-5)
