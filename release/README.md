# VECTORIA Release Binaries

This directory contains prebuilt binaries for VECTORIA v1.3.2.

## Platforms

### macOS ARM64
- **Binary**: `macosx_arm64/libvectoria.dylib`
- **Compiler**: Apple clang version 17.0.0
- **Build Date**: 2026-02-11

### Linux x86_64
- **Status**: Available via CI/CD release pipeline.
- **Compiler**: GCC 11+ recommended.

## Verification
Verify the integrity of the binaries using the provided `checksum.txt` files:
```bash
shasum -a 256 -c checksum.txt
```
