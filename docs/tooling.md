# Tooling & Introspection

VECTORIA provides standalone tooling for analyzing and visualizing execution traces. These tools observe existing execution behavior and do not modify runtime semantics.

## Trace Analysis

The `trace_analyzer.py` tool computes execution order, kernel dispatch breakdown (SIMD vs. Reference), and per-node memory footprint.

```bash
python3 tools/trace/trace_analyzer.py path/to/trace.json
```

## Determinism Verification

The `trace_diff.py` tool compares two traces to assert bitwise identical execution paths and memory offsets.

```bash
python3 tools/trace/trace_diff.py trace_a.json trace_b.json
```

## Visualization

The `trace_viz.py` tool generates SVG timelines and DOT graphs of the execution flow.

```bash
python3 tools/trace/trace_viz.py trace.json output_base
```

**Note:** Tooling observes execution; it does not influence it.
