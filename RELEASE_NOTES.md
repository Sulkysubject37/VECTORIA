# Release v1.3.1: Tooling & Distribution

**Standalone Introspection | Cross-Platform Packaging | Installation UX**

This release marks the completion of the Tooling and Distribution phase. It introduces a comprehensive suite for execution trace analysis, visualization, and cross-platform packaging, significantly reducing the friction for installation and numerical verification.

## 🔍 Tooling & Introspection

*   **Trace Analysis**: New standalone tools (`trace_analyzer.py`) to compute execution order, kernel breakdown, and memory footprints.
*   **Determinism Verification**: Introduced `trace_diff.py` for bitwise comparison of execution traces, enabling automated determinism checks.
*   **Visual Execution Flow**: Support for generating SVG timelines and DOT graphs from any VECTORIA trace.
*   **Trace CLI**: Unified `vectoria-trace` command-line utility for all introspection tasks.

## 📦 Distribution & Packaging

*   **Python Wheel**: Official pip-installable package supporting automated dependency management and CLI integration.
*   **Swift Package Manager**: Finalized SPM support with a new `VectoriaExample` executable target.
*   **Prebuilt Binaries**: Released certified `libvectoria.dylib` for macOS (ARM64) with cryptographic checksums.
*   **Portable Native Loading**: Dynamic library loading logic updated to support both packaged and developer environments.

## 🛠 CI/CD & Documentation

*   **Cross-Platform CI**: New validation workflows for Ubuntu and macOS to ensure packaging and tooling integrity.
*   **Comprehensive Docs**: New Installation, Getting Started, and Determinism Verification guides integrated into the Wiki and repository.
*   **Semantic Integrity**: No modifications to core kernels or execution semantics. This release observes the v1.3.0-stable freeze.

## 🚀 Getting Started

```bash
pip install vectoria
vectoria-trace --help
```
