import pytest
import numpy as np
from vectoria import Graph, Runtime, DType

def test_concatenation_correctness():
    g = Graph()
    
    # [2, 3]
    x1_shape = [2, 3]
    x2_shape = [2, 2]
    
    in1 = g.add_input("X1", x1_shape, DType.FLOAT32)
    in2 = g.add_input("X2", x2_shape, DType.FLOAT32)
    
    c_node = g.add_concat([in1, in2], axis=1)
    g.set_output(c_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    x1_val = np.random.randn(*x1_shape).astype(np.float32)
    x2_val = np.random.randn(*x2_shape).astype(np.float32)
    
    rt.set_input(in1.id, x1_val.flatten().tolist())
    rt.set_input(in2.id, x2_val.flatten().tolist())
    
    rt.execute()
    
    out = rt.get_output(c_node.id, 2*5)
    out_np = np.array(out).reshape(2, 5)
    
    expected = np.concatenate([x1_val, x2_val], axis=1)
    
    np.testing.assert_allclose(out_np, expected, atol=1e-5)

def test_concatenation_axis0():
    g = Graph()
    x1_shape = [2, 3]
    x2_shape = [4, 3]
    
    in1 = g.add_input("X1", x1_shape, DType.FLOAT32)
    in2 = g.add_input("X2", x2_shape, DType.FLOAT32)
    
    c_node = g.add_concat([in1, in2], axis=0)
    g.set_output(c_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    x1_val = np.random.randn(*x1_shape).astype(np.float32)
    x2_val = np.random.randn(*x2_shape).astype(np.float32)
    
    rt.set_input(in1.id, x1_val.flatten().tolist())
    rt.set_input(in2.id, x2_val.flatten().tolist())
    
    rt.execute()
    
    out = rt.get_output(c_node.id, 6*3)
    out_np = np.array(out).reshape(6, 3)
    
    expected = np.concatenate([x1_val, x2_val], axis=0)
    
    np.testing.assert_allclose(out_np, expected, atol=1e-5)
