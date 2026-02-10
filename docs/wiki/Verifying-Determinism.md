# Verifying Determinism

VECTORIA's core promise is bitwise numerical truth. You can verify the determinism of any installation using the included trace tools.

## The Verification Loop

1. **Run a benchmark graph:** Generate an execution trace from a standardized graph.
2. **Export the trace:** Save the trace to a JSON file.
3. **Repeat execution:** Run the exact same graph and inputs again.
4. **Diff the traces:** Use `vectoria-trace diff` to ensure identity.

```bash
# Generate first trace
python3 my_benchmark.py --trace trace_1.json

# Generate second trace
python3 my_benchmark.py --trace trace_2.json

# Verify bitwise identity
vectoria-trace diff trace_1.json trace_2.json
```

## Cross-Platform Considerations

Note that while intra-platform determinism is guaranteed (same binary, same hardware), bitwise identity is **not** guaranteed across different architectures (e.g., ARM64 vs x86_64) when using SIMD kernels due to hardware FMA differences. 

To achieve cross-platform bitwise identity, use `KernelPolicy::Reference`.
