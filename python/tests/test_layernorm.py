import pytest
import numpy as np
from vectoria import Graph, Runtime, DType

def layernorm_numpy(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True) # default is population var
    std = np.sqrt(var + eps)
    return gamma * (x - mean) / std + beta

def test_layernorm_correctness():
    g = Graph()
    
    # [2, 5]
    x_shape = [2, 5]
    gamma_shape = [5]
    beta_shape = [5]
    
    in_node = g.add_input("Input", x_shape, DType.FLOAT32)
    gamma_node = g.add_parameter("Gamma", gamma_shape, DType.FLOAT32, 0)
    beta_node = g.add_parameter("Beta", beta_shape, DType.FLOAT32, 1)
    
    ln_node = g.add_layernorm(in_node, gamma_node, beta_node)
    g.set_output(ln_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    # Data
    x_val = np.random.randn(*x_shape).astype(np.float32)
    gamma_val = np.random.randn(*gamma_shape).astype(np.float32)
    beta_val = np.random.randn(*beta_shape).astype(np.float32)
    
    rt.set_input(in_node.id, x_val.flatten().tolist())
    rt.set_input(gamma_node.id, gamma_val.flatten().tolist())
    rt.set_input(beta_node.id, beta_val.flatten().tolist())
    
    rt.execute()
    
    out = rt.get_output(ln_node.id, 2*5)
    out_np = np.array(out).reshape(x_shape)
    
    expected = layernorm_numpy(x_val, gamma_val, beta_val)
    
    np.testing.assert_allclose(out_np, expected, rtol=1e-4, atol=1e-5)
