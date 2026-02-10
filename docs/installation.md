# Installation & Verification

VECTORIA can be installed as a Python package, a Swift package, or by linking prebuilt binaries directly.

## Python Installation

Install the Python package from source or wheel:

```bash
# From source directory
pip install .
```

This installs the `vectoria` module and the `vectoria-trace` CLI tool.

### Verifying Python Installation
```python
from vectoria import Graph, DType
from vectoria.runtime import Runtime

g = Graph()
# ... build graph ...
rt = Runtime()
rt.load_graph(g)
rt.execute()
print(rt.get_trace())
```

## Swift Installation

Add VECTORIA as a dependency in your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Sulkysubject37/VECTORIA.git", from: "1.3.1")
]
```

## Manual Binary Installation

Prebuilt binaries for macOS (ARM64) and Linux (x86_64) are available in the `release/` directory.

1. Copy `libvectoria.dylib` (macOS) or `libvectoria.so` (Linux) to your library path.
2. Link against the library using your preferred C++ compiler.

## Verifying Determinism

To verify that an installation is behaving deterministically:

1. Generate a trace from a known graph: `vectoria-trace viz my_trace.json my_viz`
2. Run the same graph again to generate `trace_new.json`.
3. Compare them: `vectoria-trace diff my_trace.json trace_new.json`

Identical traces confirm intra-platform determinism.
