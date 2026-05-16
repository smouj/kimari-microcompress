# KMC Format Specification

**Version:** 1  
**Status:** Experimental

## Overview

The `.kmc` format is a block-oriented archive format designed for reversible, verifiable, lossless compression of AI model files. It prioritizes integrity checking, human-readable metadata, and future extensibility.

## File Layout

```
┌──────────────────────────────────────────────┐
│ Header                                        │
│   Magic: "KMC\x00\x01\x00\x00\x00"  (8 B)   │
│   Manifest length: uint64 BE         (8 B)   │
├──────────────────────────────────────────────┤
│ Manifest                                      │
│   JSON document (UTF-8 encoded)     (variable)│
├──────────────────────────────────────────────┤
│ Block Data                                    │
│   Concatenated compressed blocks    (variable)│
└──────────────────────────────────────────────┘
```

All multi-byte integers are stored in **big-endian** byte order unless otherwise specified.

## Header

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 8 | Magic | `KMC\x00\x01\x00\x00\x00` — identifies the file as KMC format version 1 |
| 8 | 8 | Manifest Length | Length of the JSON manifest in bytes (uint64, big-endian) |

Total header size: **16 bytes**.

### Magic Number

The magic number encodes both the format identifier and the format version:
- Bytes 0-2: `"KMC"` — format identifier
- Byte 3: `\x00` — separator
- Bytes 4-5: `\x01\x00` — format version 1 (uint16 big-endian)
- Bytes 6-7: `\x00\x00` — reserved for future use

## Manifest

The manifest is a JSON document encoded in UTF-8. It describes all files, blocks, and compression parameters.

### Schema

```json
{
  "version": 1,
  "tool": "kimari-microcompress",
  "tool_version": "0.1.0",
  "created_at": "2025-01-01T00:00:00+00:00",
  "total_original_size": 0,
  "total_compressed_size": 0,
  "files": [
    {
      "path": "model.safetensors",
      "original_size": 1048576,
      "hash": "sha256:abcdef...",
      "block_size": 262144,
      "blocks": [
        {
          "index": 0,
          "offset": 1234,
          "compressed_size": 200000,
          "original_size": 262144,
          "codec": "zstd",
          "hash": "sha256:123456..."
        }
      ]
    }
  ]
}
```

### Fields

#### Top-Level

| Field | Type | Description |
|-------|------|-------------|
| `version` | integer | Format version (currently 1) |
| `tool` | string | Tool name that created the archive |
| `tool_version` | string | Version of the tool |
| `created_at` | string | ISO 8601 timestamp of archive creation |
| `total_original_size` | integer | Sum of all original file sizes in bytes |
| `total_compressed_size` | integer | Sum of all compressed block sizes in bytes |
| `files` | array | List of file entries |

#### File Entry

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Relative file path (POSIX format, forward slashes) |
| `original_size` | integer | Original file size in bytes |
| `hash` | string | SHA-256 hex digest of the original file |
| `block_size` | integer | Block size used for this file (bytes) |
| `blocks` | array | List of block entries in order |

#### Block Entry

| Field | Type | Description |
|-------|------|-------------|
| `index` | integer | Zero-based block index within the file |
| `offset` | integer | Absolute byte offset from the start of the archive |
| `compressed_size` | integer | Size of the compressed block data in bytes |
| `original_size` | integer | Size of the original (uncompressed) block data in bytes |
| `codec` | string | Codec used: `"zstd"`, `"zlib"`, or `"raw"` |
| `hash` | string | SHA-256 hex digest of the compressed block data |

## Block Data

Block data is stored immediately after the manifest, as a concatenation of all compressed blocks from all files. The order of blocks in the data section matches the order they appear in the manifest (files in order, blocks within each file in order).

Each block's absolute offset is recorded in the manifest, enabling random access to individual blocks without parsing preceding data.

### Block Size

The default block size is **256 KiB (262,144 bytes)**. The last block of a file may be smaller. Block size is configurable at pack time.

### Codec Selection

For each block, the best codec is chosen independently:
1. If `zstandard` is available, compress with zstd (level 3 by default).
2. Otherwise, compress with `zlib` (level 6 by default).
3. If the compressed output is not smaller than the original, store the block raw (`"raw"` codec).

This per-block codec selection ensures that:
- Already-compressed or random data is not expanded.
- Each block gets the most appropriate treatment.
- The archive never exceeds the original data size by more than a negligible overhead.

## Integrity Verification

Integrity is verified at two levels:

### Block Level
- Each block's compressed data is hashed with SHA-256.
- The hash is stored in the manifest's `blocks[].hash` field.
- Verification reads the compressed block, computes its hash, and compares it to the manifest value.
- This is fast and doesn't require decompression.

### File Level
- Each file's original (uncompressed) content is hashed with SHA-256.
- The hash is stored in the manifest's `files[].hash` field.
- Verification after unpacking computes the hash of the reconstructed file and compares it.
- This confirms byte-exact roundtrip integrity.

## Path Handling

- File paths in the manifest use **POSIX format** (forward slashes) for cross-platform compatibility.
- Paths are **relative** to the archive root (the source directory that was packed).
- Absolute paths are not allowed.
- Path traversal (`..`) components are not allowed.
- On unpack, paths are validated to ensure they resolve within the output directory (path traversal protection).

## Versioning

The format version is encoded in the magic number (bytes 4-5) and repeated in the manifest's `version` field. Future versions may add:
- New codecs.
- Additional metadata fields.
- Encrypted blocks.
- Block-level checksums beyond SHA-256.

Readers should check the magic number version and the manifest version for compatibility.

## Security Considerations

- **Path traversal**: The unpack operation validates all paths against directory traversal attacks. See [SECURITY_MODEL.md](SECURITY_MODEL.md).
- **Denial of service**: Manifest length is bounded; readers should reject manifests larger than a reasonable limit (e.g., 100 MB).
- **Data integrity**: SHA-256 hashes provide strong integrity guarantees against both accidental corruption and intentional tampering (though they do not provide authentication; see security model).
