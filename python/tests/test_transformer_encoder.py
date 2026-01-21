import pytest
import numpy as np
from vectoria import Graph, Runtime, DType
from scipy.special import softmax

def layernorm_np(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b

def attention_np(q, k, v):
    dk = q.shape[-1]
    scores = np.matmul(q, k.T) / np.sqrt(dk)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v)

def mha_np(x, wq, wk, wv, wo, h):
    T, d_model = x.shape
    dk = d_model // h
    Q = np.matmul(x, wq).reshape(T, h, dk).transpose(1, 0, 2)
    K = np.matmul(x, wk).reshape(T, h, dk).transpose(1, 0, 2)
    V = np.matmul(x, wv).reshape(T, h, dk).transpose(1, 0, 2)
    heads = [attention_np(Q[i], K[i], V[i]) for i in range(h)]
    return np.matmul(np.concatenate(heads, axis=-1), wo)

def encoder_np(x, wq, wk, wv, wo, h, g1, b1, wf1, bf1, wf2, bf2, g2, b2):
    m = mha_np(x, wq, wk, wv, wo, h)
    y = layernorm_np(x + m, g1, b1)
    f = np.matmul(np.maximum(0, np.matmul(y, wf1) + bf1), wf2) + bf2
    return layernorm_np(y + f, g2, b2)

def test_encoder_correctness():
    g = Graph()
    T, d_model, h, d_ff = 2, 4, 2, 8
    
    x = g.add_input("X", [T, d_model], DType.FLOAT32)
    wq = g.add_parameter("WQ", [d_model, d_model], DType.FLOAT32, 0)
    wk = g.add_parameter("WK", [d_model, d_model], DType.FLOAT32, 1)
    wv = g.add_parameter("WV", [d_model, d_model], DType.FLOAT32, 2)
    wo = g.add_parameter("WO", [d_model, d_model], DType.FLOAT32, 3)
    g1 = g.add_parameter("G1", [d_model], DType.FLOAT32, 4)
    b1 = g.add_parameter("B1", [d_model], DType.FLOAT32, 5)
    wf1 = g.add_parameter("WF1", [d_model, d_ff], DType.FLOAT32, 6)
    bf1 = g.add_parameter("BF1", [d_ff], DType.FLOAT32, 7)
    wf2 = g.add_parameter("WF2", [d_ff, d_model], DType.FLOAT32, 8)
    bf2 = g.add_parameter("BF2", [d_model], DType.FLOAT32, 9)
    g2 = g.add_parameter("G2", [d_model], DType.FLOAT32, 10)
    b2 = g.add_parameter("B2", [d_model], DType.FLOAT32, 11)

    enc = g.add_transformer_encoder(x, wq, wk, wv, wo, h, g1, b1, wf1, bf1, wf2, bf2, g2, b2)
    g.set_output(enc)
    g.compile()

    rt = Runtime()
    rt.load_graph(g)

    # Init data
    vals = {node['id']: np.random.randn(*node['shape']).astype(np.float32) for node in g.nodes if node['type'] in ["Input", "Parameter"]}
    for nid, val in vals.items():
        rt.set_input(nid, val.flatten().tolist())

    rt.execute()
    out = np.array(rt.get_output(enc.id, T*d_model)).reshape(T, d_model)

    expected = encoder_np(vals[x.id], vals[wq.id], vals[wk.id], vals[wv.id], vals[wo.id], h, 
                          vals[g1.id], vals[b1.id], vals[wf1.id], vals[bf1.id], 
                          vals[wf2.id], vals[bf2.id], vals[g2.id], vals[b2.id])

    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
