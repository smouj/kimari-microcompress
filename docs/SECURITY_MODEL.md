# Security Model

## Threat Model

KMC operates on the assumption that the user controls the source data being packed and the output location. The primary security concerns are:

### 1. Path Traversal (Unpacking)

**Threat**: A malicious `.kmc` archive contains file paths like `../../etc/passwd` that, when unpacked, write outside the intended output directory.

**Mitigation**: The `unpack()` function in `archive.py` includes explicit path traversal protection:
- All relative paths are resolved against the output directory.
- The resolved path is checked to ensure it starts with the output directory prefix.
- Absolute paths and `..` components are rejected.
- This is implemented in the `_safe_path()` function.

### 2. Manifest Bomb (Denial of Service)

**Threat**: A malicious archive declares an extremely large manifest length, causing the reader to allocate excessive memory.

**Mitigation**: Manifest reading should enforce a reasonable upper bound on manifest size (e.g., 100 MB). Current implementation reads the manifest as declared; hardening this limit is a priority improvement.

### 3. Zip Bomb (Decompression Bomb)

**Threat**: A malicious archive contains blocks that decompress to a much larger size than expected, consuming disk space.

**Mitigation**: The manifest records the `original_size` for each block and file. Unpack operations should check that the total decompressed size matches expectations. Future versions should enforce explicit size limits.

### 4. Hash Collision (Integrity Bypass)

**Threat**: An attacker modifies block data and also modifies the manifest hash to match, bypassing integrity verification.

**Mitigation**: SHA-256 is currently considered collision-resistant. However, KMC does not provide authentication — it verifies integrity, not authenticity. For authentication, users should:
- Verify the archive's outer hash/signature via external means.
- Use HTTPS for archive transfer.
- Sign archives with a tool like GPG.

### 5. Pickle Deserialization (Model Files)

**Threat**: PyTorch `.bin` and `.pt` files may contain arbitrary pickle payloads that execute code on load.

**Mitigation**: KMC never deserializes model data — it only reads and writes raw bytes. The inspector module detects pickle-based formats but never unpickles them. Users should still be cautious when loading the unpacked model files with their respective frameworks.

## Security Improvements Planned

1. **Hardened `unpack()`**: Additional tests for path traversal edge cases (symlinks, Unicode normalization, Windows-specific attacks).
2. **Manifest size limit**: Enforce a configurable maximum manifest size.
3. **Decompressed size limit**: Enforce a maximum total decompressed size per archive.
4. **Archive signing**: Optional GPG or minisign signature verification.
5. **Sandboxed unpack**: Optional unpack in a restricted directory with OS-level sandboxing.

## Reporting Security Issues

If you discover a security vulnerability in KMC, please report it responsibly by opening a GitHub issue with the `security` label or contacting the maintainers directly.
