import pytest
import numpy as np
from vectoria import Graph, Runtime, DType
from scipy.special import log_softmax

def cross_entropy_numpy(logits, target):
    log_probs = log_softmax(logits, axis=-1)
    return -np.sum(target * log_probs, axis=-1)

def test_crossentropy_correctness():
    g = Graph()
    x_shape = [2, 5]
    
    logits_node = g.add_input("Logits", x_shape, DType.FLOAT32)
    target_node = g.add_input("Target", x_shape, DType.FLOAT32)
    
    ce_node = g.add_crossentropy(logits_node, target_node)
    g.set_output(ce_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    logits_val = np.random.randn(*x_shape).astype(np.float32)
    # One-hot targets
    target_val = np.zeros(x_shape, dtype=np.float32)
    target_val[0, 0] = 1.0
    target_val[1, 2] = 1.0
    
    rt.set_input(logits_node.id, logits_val.flatten().tolist())
    rt.set_input(target_node.id, target_val.flatten().tolist())
    
    rt.execute()
    
    out = rt.get_output(ce_node.id, 2)
    out_np = np.array(out)
    
    expected = cross_entropy_numpy(logits_val, target_val)
    
    np.testing.assert_allclose(out_np, expected, rtol=1e-4, atol=1e-5)

def test_crossentropy_probabilistic_targets():
    g = Graph()
    x_shape = [1, 3]
    logits_node = g.add_input("Logits", x_shape, DType.FLOAT32)
    target_node = g.add_input("Target", x_shape, DType.FLOAT32)
    ce_node = g.add_crossentropy(logits_node, target_node)
    g.set_output(ce_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    logits_val = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    target_val = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    
    rt.set_input(logits_node.id, logits_val.flatten().tolist())
    rt.set_input(target_node.id, target_val.flatten().tolist())
    rt.execute()
    
    out = rt.get_output(ce_node.id, 1)
    
    expected = cross_entropy_numpy(logits_val, target_val)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
