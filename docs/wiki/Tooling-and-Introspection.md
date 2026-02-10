# Tooling & Introspection

VECTORIA includes a suite of Python-based tools for inspecting execution traces. These tools are strictly observational.

## Available Tools

- **Analyzer**: Computes op execution order, kernel dispatch breakdown, and memory footprints.
- **Diff**: Asserts identical event sequences and dispatch decisions between traces for determinism verification.
- **Visualizer**: Generates SVG timelines and DOT graphs for architectural inspection.

## Usage

Traces exported from the `Runtime` or `Engine` can be processed directly:

```bash
python3 tools/trace/trace_analyzer.py trace.json
```

## Determinism

Diffing traces is the primary method for verifying intra-platform determinism across different runs.

> Tooling observes execution; it does not influence it.
