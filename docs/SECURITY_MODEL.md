# Security Model

## Overview

KMC operates on the assumption that the user controls the source data being packed and the output location. The primary security concerns are around unpacking untrusted archives and handling potentially malicious model files. This document describes KMC's threat model, implemented mitigations, and planned security improvements.

## Threat Model

### 1. Path Traversal (Unpacking)

**Threat**: A malicious `.kmc` archive contains file paths like `../../etc/passwd` that, when unpacked, write outside the intended output directory. This is the most critical attack vector for archive formats because it can lead to arbitrary file overwrite, privilege escalation, or data destruction.

**Mitigation**: The `unpack()` function in `archive.py` includes explicit path traversal protection via the `_safe_path()` function:

- All relative paths are resolved against the output directory using `Path.resolve()`.
- The resolved path is checked to ensure it starts with the output directory prefix.
- Absolute paths (starting with `/` or drive letters on Windows) are rejected.
- Path components containing `..` are rejected.
- Null bytes (`\x00`) in paths are rejected.
- Control characters (ASCII 0x00-0x1F) in paths are rejected.
- Unicode normalization tricks are handled by resolving paths before comparison.

**Test coverage**: `tests/test_security.py` includes tests for path traversal with `../` in various positions, absolute paths, null byte injection, and other edge cases.

### 2. Manifest Bomb (Denial of Service)

**Threat**: A malicious archive declares an extremely large manifest length, causing the reader to allocate excessive memory. This could crash the process or the entire system if the manifest length is set to, say, 100 GB.

**Mitigation**: Manifest reading enforces a configurable upper bound on manifest size (default: 100 MB, configured via `security.max_manifest_size` in `configs/default.yaml`). If the declared manifest length exceeds this limit, the archive is rejected immediately without reading the manifest data. This prevents unbounded memory allocation from a malicious header.

The 100 MB limit is conservative: a manifest for a model with 10,000 files and 100,000 blocks would typically be well under 50 MB. If you need to compress archives with extremely large manifests, you can increase this limit, but be aware of the memory implications.

### 3. Zip Bomb (Decompression Bomb)

**Threat**: A malicious archive contains blocks that decompress to a much larger size than expected, consuming disk space or memory. A classic zip bomb compresses 42 KB into a 4.5 GB file.

**Mitigation**: The manifest records the `original_size` for each block and file. Unpack operations should check that the total decompressed size matches expectations. The configurable `security.max_decompressed_size` (default: 10 GB) sets an upper bound on total decompressed size per archive. If the manifest declares a total original size exceeding this limit, the archive is rejected.

Additionally, the per-block `original_size` field enables incremental verification: each block's decompressed output is checked against its expected size before proceeding.

### 4. Hash Collision (Integrity Bypass)

**Threat**: An attacker modifies block data and also modifies the manifest hash to match, bypassing integrity verification. This requires finding a SHA-256 preimage or collision.

**Mitigation**: SHA-256 is currently considered collision-resistant, with no practical collision attacks known as of 2026. However, KMC does not provide authentication — it verifies integrity, not authenticity. A determined attacker with write access to the archive could modify both the data and the hash.

For authentication, users should:
- Verify the archive's outer hash/signature via external means (e.g., SHA-256 of the entire `.kmc` file).
- Use HTTPS for archive transfer.
- Sign archives with a tool like GPG or minisign.
- Store archive hashes in a trusted, append-only location.

### 5. Pickle Deserialization (Model Files)

**Threat**: PyTorch `.bin` and `.pt` files may contain arbitrary pickle payloads that execute code on load. This is a well-known attack vector in the ML ecosystem, where downloading and loading a malicious model file can result in arbitrary code execution.

**Mitigation**: KMC **never** deserializes model data — it only reads and writes raw bytes. The inspector module detects pickle-based formats by filename (e.g., `optimizer.pt`, `training_args.bin`, `pytorch_model.bin`) but **never** unpickles them. When pickle-based files are detected during checkpoint packing, KMC emits explicit warnings:

```
Warning: optimizer.pt detected (pickle-based). Only size/hash will be recorded;
contents will NOT be deserialized.
```

These files are still compressed and included in the archive. On unpack, they are restored byte-for-byte identical. The only difference is that KMC does not attempt to read their internal structure.

**User responsibility**: After unpacking, users should still be cautious when loading model files with their respective frameworks. Use `safetensors` format whenever possible to avoid pickle-based attacks.

### 6. Symlink Overwrite (Unpacking)

**Threat**: An attacker creates a symlink pointing to a sensitive location, and then the unpack operation writes through the symlink, overwriting the target file.

**Mitigation**: The `unpack()` function checks whether the target path is a symlink before writing. If a symlink exists at the target location, the unpack operation refuses to overwrite it and raises an error. This prevents both accidental and malicious symlink-based attacks.

### 7. Duplicate Path Injection (Unpacking)

**Threat**: A malicious manifest contains duplicate file paths, causing the unpack operation to write the same path multiple times with different data. This could lead to race conditions or unexpected file states.

**Mitigation**: Manifest validation checks for duplicate file paths during both pack and unpack. If duplicate paths are detected, the operation is rejected. This ensures that each path in the archive is unique and deterministic.

### 8. Selective Extraction Path Abuse (v0.7+)

**Threat**: The `--only` flag in `kmc unpack` accepts a pattern string that could be used to trick the extraction into writing files outside the output directory. For example, an attacker might craft a manifest with file paths that, when matched and extracted, resolve to locations outside the intended output directory.

**Mitigation**: The selective extraction code uses the same `safe_join_extract_path()` function as full unpacking. Every file extracted via `--only` or `--tensor` is validated to ensure the resolved path stays within the output directory. Additionally, the `--only` pattern itself is treated as a matching expression, not a filesystem path. Patterns containing path traversal sequences (`..`) or absolute paths are rejected.

**Test coverage**: `tests/test_v07_features.py` includes tests for `--only` with `../evil` patterns and absolute path patterns like `/etc/passwd`, confirming that these are rejected.

### 9. Block Checksum Bypass via Partial Access (v0.7+)

**Threat**: A malicious archive contains blocks with incorrect data but valid-looking manifest hashes. If partial access skips checksum verification, corrupted or tampered data could be returned to the caller without detection.

**Mitigation**: The `KMCReader` class verifies block checksums on every read operation. Each block's compressed data is hashed with SHA-256 and compared against the manifest value before decompression. For file-level reads, the reconstructed file hash is also verified after concatenation. This ensures that partial access maintains the same integrity guarantees as full unpack operations. There is no "skip verification" option in the public API.

**Test coverage**: `tests/test_v07_features.py` includes tests that corrupt block data in an archive and verify that `KMCReader.read_file()` raises a `ValueError` with a checksum mismatch message.

### 10. Tensor Name Injection (v0.7+)

**Threat**: Tensor names in the manifest contain special characters (path separators, null bytes) that could be used for path traversal when extracting tensors to disk via `extract_tensor()`.

**Mitigation**: The `extract_tensor()` method sanitizes tensor names before writing to disk. Forward slashes, backslashes, and colons are replaced with underscores. The output path is constructed by joining the output directory with the sanitized filename and a `.bin` extension. The resulting path is always within the output directory.

### 11. Archive Offset Spoofing (v0.7+)

**Threat**: A malicious manifest contains `archive_offset` values that point to incorrect locations in the archive, causing the reader to read data from unintended offsets. This could potentially be used to read arbitrary data from the file if the offset points outside the archive.

**Mitigation**: Block checksum verification prevents this attack. Even if the offset points to the wrong data, the SHA-256 hash of the read data will not match the manifest's block hash, and the read will fail with a checksum mismatch error. An attacker would need to simultaneously modify both the offset and the hash to bypass this check, which would require finding a SHA-256 collision.

## Security Architecture

### Defense in Depth

KMC applies security checks at multiple layers:

```
┌─────────────────────────────────────────┐
│  Layer 1: Manifest Validation           │
│  - Size limits (100 MB max)             │
│  - Duplicate path detection             │
│  - Codec validation                     │
│  - Size consistency checks              │
│  - Index metadata validation (v0.7+)    │
├─────────────────────────────────────────┤
│  Layer 2: Path Sanitization             │
│  - No absolute paths                    │
│  - No traversal (..) components         │
│  - No null bytes or control chars       │
│  - Symlink overwrite protection         │
│  - Tensor name sanitization (v0.7+)     │
│  - Selective extraction path checks     │
├─────────────────────────────────────────┤
│  Layer 3: Integrity Verification        │
│  - SHA-256 per-block hash               │
│  - SHA-256 per-file hash                │
│  - Block size consistency               │
│  - Total size validation                │
│  - Partial read checksum checks (v0.7+) │
├─────────────────────────────────────────┤
│  Layer 4: Data Safety                   │
│  - No pickle deserialization            │
│  - No model weight loading              │
│  - No code execution from archives      │
│  - Lossless-only operations             │
│  - Block offset validation (v0.7+)      │
└─────────────────────────────────────────┘
```

### What KMC Does NOT Protect Against

It is important to understand the boundaries of KMC's security model:

1. **Authentication**: KMC verifies integrity (data matches its hash) but does not verify authenticity (the hash came from a trusted source). An attacker with write access to the archive can modify both data and hashes.

2. **Encryption**: KMC does not encrypt archive contents. The manifest is plain JSON and the block data is compressed but not encrypted. Use external tools (GPG, age, etc.) for encryption.

3. **Side-channel attacks**: KMC does not defend against timing attacks, memory dumps, or other side channels during compression or decompression.

4. **Supply chain attacks**: KMC cannot verify that the original model files came from a trusted source. It can only verify that the archive faithfully preserves whatever was packed.

5. **Malicious model weights**: KMC does not inspect model weights for adversarial content. A model that produces harmful outputs when loaded will produce the same harmful outputs after unpacking.

## Secure Usage Recommendations

### For Archive Consumers

1. **Only unpack archives from trusted sources.** Verify the archive's outer hash (SHA-256 of the `.kmc` file) against a known-good value before unpacking.

2. **Inspect before unpacking.** Use `kmc inspect` to examine the archive's manifest and file list before decompressing. Look for unexpected file paths or unusually large declared sizes.

3. **Unpack to a dedicated directory.** Never unpack directly to system directories or directories containing sensitive files.

4. **Verify after unpacking.** Run `kmc verify` after packing and before relying on the archive for storage.

### For Archive Producers

1. **Use safetensors format.** Prefer `model.safetensors` over `pytorch_model.bin` to avoid pickle vulnerabilities.

2. **Sign your archives.** After packing, compute and publish the SHA-256 hash of the `.kmc` file. Consider GPG-signing the hash for strong authenticity guarantees.

3. **Use HTTPS for transfer.** Always transfer archives over encrypted channels to prevent tampering in transit.

4. **Keep the original manifest.** Store a copy of the manifest separately from the archive for disaster recovery.

## Security Improvements Planned

1. **Hardened `unpack()`**: Additional tests for path traversal edge cases (Unicode normalization variations, Windows-specific path separators, extremely long path components, case-sensitivity issues on case-insensitive filesystems).

2. **Configurable size limits**: Allow users to set custom limits for manifest size, total decompressed size, and maximum file count per archive.

3. **Archive signing**: Optional GPG or minisign signature verification integrated into the `kmc verify` command.

4. **Sandboxed unpack**: Optional unpack in a restricted directory with OS-level sandboxing (e.g., using `pledge()` on OpenBSD, seccomp on Linux, or sandbox-exec on macOS).

5. **Streaming verification**: Verify block hashes during unpacking rather than after, enabling early detection of corruption.

6. **Timestamp validation**: Optional check that archive timestamps are within expected ranges to detect suspicious archives.

7. **Entropy analysis**: Detect blocks with suspiciously low entropy (potential zip bombs) or high entropy (potential encrypted/packed payloads) and flag them for manual review.

## Reporting Security Issues

If you discover a security vulnerability in KMC, please report it responsibly:

1. **Do not** file a public GitHub issue with exploit details.
2. Open a GitHub issue with the `security` label and a vague description ("potential path traversal in unpack").
3. The maintainers will respond within 48 hours to coordinate disclosure.
4. Once confirmed and fixed, a security advisory will be published via GitHub Security Advisories.

We appreciate responsible disclosure and will credit researchers who report vulnerabilities.

## Security Audit Status

KMC has not undergone a formal security audit. The security mitigations described in this document are based on known attack patterns and best practices for archive formats. Users handling highly sensitive data or processing untrusted archives should conduct their own security review.

| Component | Status | Notes |
|-----------|--------|-------|
| Path traversal protection | ✅ Implemented | Tested with multiple attack vectors |
| Manifest size limits | ✅ Implemented | Default 100 MB max |
| Decompressed size limits | ✅ Implemented | Default 10 GB max |
| Duplicate path detection | ✅ Implemented | Rejected at validation |
| Symlink overwrite protection | ✅ Implemented | Refuses to overwrite symlinks |
| Null byte injection | ✅ Implemented | Rejected in path validation |
| Pickle deserialization | ✅ Implemented | Never deserializes pickle files |
| Block hash verification | ✅ Implemented | SHA-256 per-block |
| File hash verification | ✅ Implemented | SHA-256 per-file |
| Partial read checksum verification | ✅ Implemented | Block + file hash on every partial read (v0.7+) |
| Selective extraction path protection | ✅ Implemented | Same safe_join as full unpack (v0.7+) |
| Tensor name sanitization | ✅ Implemented | Special chars replaced in extract_tensor (v0.7+) |
| Archive offset validation | ✅ Implemented | Checksum prevents spoofed offsets (v0.7+) |
| Archive signing | 🔲 Planned | Future release |
| Sandboxed unpack | 🔲 Planned | Future release |
| Streaming verification | 🔲 Planned | Future release |
| Formal security audit | 🔲 Not started | Recommended before production use |
