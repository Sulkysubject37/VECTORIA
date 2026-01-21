import pytest
import numpy as np
from vectoria import Graph, Runtime, DType
from scipy.special import softmax

def attention_numpy(q, k, v):
    dk = q.shape[-1]
    scores = np.matmul(q, k.T) / np.sqrt(dk)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v)

def mha_numpy(x, wq, wk, wv, wo, h):
    T, d_model = x.shape
    dk = d_model // h
    
    Q = np.matmul(x, wq)
    K = np.matmul(x, wk)
    V = np.matmul(x, wv)
    
    # Split heads
    Q = Q.reshape(T, h, dk).transpose(1, 0, 2)
    K = K.reshape(T, h, dk).transpose(1, 0, 2)
    V = V.reshape(T, h, dk).transpose(1, 0, 2)
    
    heads = []
    for i in range(h):
        heads.append(attention_numpy(Q[i], K[i], V[i]))
    
    H = np.concatenate(heads, axis=-1)
    return np.matmul(H, wo)

def test_mha_correctness():
    g = Graph()
    T, d_model, h = 3, 4, 2
    
    x_node = g.add_input("X", [T, d_model], DType.FLOAT32)
    wq_node = g.add_parameter("WQ", [d_model, d_model], DType.FLOAT32, 0)
    wk_node = g.add_parameter("WK", [d_model, d_model], DType.FLOAT32, 1)
    wv_node = g.add_parameter("WV", [d_model, d_model], DType.FLOAT32, 2)
    wo_node = g.add_parameter("WO", [d_model, d_model], DType.FLOAT32, 3)
    
    mha_node = g.add_multi_head_attention(x_node, wq_node, wk_node, wv_node, wo_node, h)
    g.set_output(mha_node)
    g.compile()
    
    rt = Runtime()
    rt.load_graph(g)
    
    x_val = np.random.randn(T, d_model).astype(np.float32)
    wq_val = np.random.randn(d_model, d_model).astype(np.float32)
    wk_val = np.random.randn(d_model, d_model).astype(np.float32)
    wv_val = np.random.randn(d_model, d_model).astype(np.float32)
    wo_val = np.random.randn(d_model, d_model).astype(np.float32)
    
    rt.set_input(x_node.id, x_val.flatten().tolist())
    rt.set_input(wq_node.id, wq_val.flatten().tolist())
    rt.set_input(wk_node.id, wk_val.flatten().tolist())
    rt.set_input(wv_node.id, wv_val.flatten().tolist())
    rt.set_input(wo_node.id, wo_val.flatten().tolist())
    
    rt.execute()
    
    out = rt.get_output(mha_node.id, T*d_model)
    out_np = np.array(out).reshape(T, d_model)
    
    expected = mha_numpy(x_val, wq_val, wk_val, wv_val, wo_val, h)
    
    np.testing.assert_allclose(out_np, expected, rtol=1e-4, atol=1e-5)
