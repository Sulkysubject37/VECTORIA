# Graph Semantics & Composition

VECTORIA distinguishes between **Primitive Kernels** (atomic units of computation) and **Composed Operations** (subgraphs built from primitives).

## Compositional Philosophy

Composed operations are used to express complex mathematical functions without bloating the low-level kernel set. This approach ensures:
1. **Traceability**: Every internal step of a composed op is visible in the execution trace.
2. **Predictability**: No hidden optimizations or fusion occurs unless explicitly defined as a new kernel.
3. **Correctness**: Composed ops inherit the deterministic guarantees of the underlying reference kernels.

## Supported Composed Operations

### Softmax (Last Axis)
Softmax is implemented as an explicit expansion into 5 primitive operations (`ReduceMax`, `Sub`, `Exp`, `ReduceSum`, `Div`).

### Stable Softmax
Reuses `LogSoftmax` logic to prevent exponential overflow. Structure: `Exp(LogSoftmax(x))`.

### LayerNorm
Performs variance-based normalization. Expanded into `ReduceMean`, `Sub`, `Mul`, `Sqrt`, `Add`, `Div`.

### Multi-Head Attention
Decomposes into projection matrices, head-splitting (`Reshape`+`Transpose`+`Slice`), and scaled dot-product blocks.

### Transformer Encoder
The top-level semantic unit integrating MHA, FFN, and Residuals.

**Why Composed?**
While many frameworks provide a fused Softmax kernel for performance, VECTORIA prioritizes **numerical inspectability**. By expanding Softmax, a user can verify the intermediate stability shift (Sub) and the normalization step (Div) directly from the execution trace.

## Observability

When a Composed Operation is executed, the `trace::Tracer` will log multiple `KernelDispatch` events. For example, a single Softmax call in Python will result in five distinct kernel dispatches in the C++ core.
