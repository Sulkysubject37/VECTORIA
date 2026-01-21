import pytest
import numpy as np
from vectoria import Graph, Runtime, DType
from scipy.special import softmax

def attention_numpy(q, k, v):
    dk = q.shape[-1]
    scores = np.matmul(q, k.T) / np.sqrt(dk)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v)

def test_attention_correctness():
    g = Graph()
    
    T, dk, dv = 3, 4, 2
    q_node = g.add_input("Q", [T, dk], DType.FLOAT32)
    k_node = g.add_input("K", [T, dk], DType.FLOAT32)
    v_node = g.add_input("V", [T, dv], DType.FLOAT32)
    
    attn_node = g.add_attention(q_node, k_node, v_node)
    g.set_output(attn_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    q_val = np.random.randn(T, dk).astype(np.float32)
    k_val = np.random.randn(T, dk).astype(np.float32)
    v_val = np.random.randn(T, dv).astype(np.float32)
    
    rt.set_input(q_node.id, q_val.flatten().tolist())
    rt.set_input(k_node.id, k_val.flatten().tolist())
    rt.set_input(v_node.id, v_val.flatten().tolist())
    
    rt.execute()
    
    out = rt.get_output(attn_node.id, T*dv)
    out_np = np.array(out).reshape(T, dv)
    
    expected = attention_numpy(q_val, k_val, v_val)
    
    np.testing.assert_allclose(out_np, expected, rtol=1e-4, atol=1e-5)

def test_attention_adversarial():
    """Test with large inputs to ensure stable softmax is used."""
    g = Graph()
    T, dk, dv = 2, 2, 2
    q_node = g.add_input("Q", [T, dk], DType.FLOAT32)
    k_node = g.add_input("K", [T, dk], DType.FLOAT32)
    v_node = g.add_input("V", [T, dv], DType.FLOAT32)
    
    attn_node = g.add_attention(q_node, k_node, v_node)
    g.set_output(attn_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    # Large inputs that would overflow naive softmax
    q_val = np.array([[1000.0, 0.0], [0.0, 1000.0]], dtype=np.float32)
    k_val = np.array([[1000.0, 0.0], [0.0, 1000.0]], dtype=np.float32)
    v_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    rt.set_input(q_node.id, q_val.flatten().tolist())
    rt.set_input(k_node.id, k_val.flatten().tolist())
    rt.set_input(v_node.id, v_val.flatten().tolist())
    
    rt.execute()
    
    out = rt.get_output(attn_node.id, T*dv)
    out_np = np.array(out).reshape(T, dv)
    
    # Expected: softmax(1000000/sqrt(2), ...) -> very polarized weights
    # Since dk=2, sqrt(dk) approx 1.414. Scores approx [[1e6/1.414, 0], [0, 1e6/1.414]]
    # Probs approx [[1, 0], [0, 1]]
    # Output approx V
    expected = attention_numpy(q_val, k_val, v_val)
    
    np.testing.assert_allclose(out_np, expected, atol=1e-5)
