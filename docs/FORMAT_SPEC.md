# KMC Format Specification

**Version:** 3  
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

### Manifest Schema (v3)

```json
{
  "version": 3,
  "tool": "kimari-microcompress",
  "tool_version": "0.4.0-alpha",
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
          "codec": "floatplane",
          "hash": "sha256:123456...",
          "codec_metadata": {
            "transform": "floatplane",
            "dtype": "BF16",
            "inner_codec": "zstd",
            "planes": ["sign", "exponent", "mantissa"],
            "n_elements": 131072
          },
          "tensor_name": "transformer.h.0.attn.c_attn.weight",
          "tensor_dtype": "BF16",
          "tensor_shape": [768, 2304]
        }
      ],
      "tensor_count": 1,
      "dtype_summary": ["BF16"],
      "tensor_entries": [
        {
          "name": "transformer.h.0.attn.c_attn.weight",
          "dtype": "BF16",
          "shape": [768, 2304],
          "byte_offset": 0,
          "byte_size": 3538944
        }
      ]
    }
  ]
}
```

### Manifest Schema (v1 — Legacy)

For reference, the original v1 manifest schema is shown below. V3 readers should handle v1 and v2 manifests gracefully by defaulting missing fields to empty/zero values.

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
| `version` | integer | Format version (currently 3) |
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
| `codec` | string | Codec used: `"zstd"`, `"zlib"`, `"raw"`, `"byteplane"`, or `"floatplane"` |
| `hash` | string | SHA-256 hex digest of the compressed block data |
| `codec_metadata` | object | Codec-specific parameters for decompression (v3, optional) |
| `tensor_name` | string | Name of the tensor this block belongs to (v3, optional) |
| `tensor_dtype` | string | Dtype of the tensor, e.g. `"BF16"`, `"FP16"`, `"FP32"` (v3, optional) |
| `tensor_shape` | array | Shape of the tensor as list of integers (v3, optional) |

## Block Data

Block data is stored immediately after the manifest, as a concatenation of all compressed blocks from all files. The order of blocks in the data section matches the order they appear in the manifest (files in order, blocks within each file in order).

Each block's absolute offset is recorded in the manifest, enabling random access to individual blocks without parsing preceding data.

### Block Size

The default block size is **256 KiB (262,144 bytes)**. The last block of a file may be smaller. Block size is configurable at pack time.

### Codec Selection

For each block, the best codec is chosen independently:
1. If `--codec auto` (default): The automatic selector determines a candidate chain based on `tensor_dtype` from the context, tries each candidate, verifies the roundtrip, and picks the smallest result.
2. If `--codec byteplane` or `--codec floatplane`: Only the specified codec is used.
3. If `zstandard` is available, compress with zstd (level 3 by default).
4. Otherwise, compress with `zlib` (level 6 by default).
5. If the compressed output is not smaller than the original, store the block raw (`"raw"` codec).

This per-block codec selection ensures that:
- Already-compressed or random data is not expanded.
- Each block gets the most appropriate treatment.
- The archive never exceeds the original data size by more than a negligible overhead.
- Tensor-aware codecs (BytePlane, FloatPlane) are automatically selected when they produce better results.

### `codec_metadata` Fields

The `codec_metadata` object stores parameters needed for lossless decompression that are specific to the codec used. It is present when the codec is `byteplane` or `floatplane` (v3 manifests), and empty/absent for `zstd`, `zlib`, and `raw` codecs.

**BytePlane `codec_metadata`:**

| Field | Type | Description |
|-------|------|-------------|
| `transform` | string | `"byteplane"` |
| `element_size` | integer | Bytes per element (2 for BF16/FP16, 4 for FP32) |
| `inner_codec` | string | Inner codec used: `"zstd"` or `"zlib"` |
| `_misaligned_tail` | integer | Number of tail bytes not fitting element_size (if present) |

**FloatPlane `codec_metadata`:**

| Field | Type | Description |
|-------|------|-------------|
| `transform` | string | `"floatplane"` or `"byteplane_fallback"` |
| `dtype` | string | Dtype used for separation: `"BF16"`, `"FP16"`, `"FP32"` |
| `inner_codec` | string | Inner codec used: `"zstd"` or `"zlib"` |
| `planes` | array | List of plane names: `["sign", "exponent", "mantissa"]` |
| `n_elements` | integer | Number of float elements separated |
| `fallback_reason` | string | Reason for falling back to byteplane (if applicable) |

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

The format version is encoded in the magic number (bytes 4-5) and repeated in the manifest's `version` field. Version history:

- **v1** (KMC v0.1–v0.2): Basic file/block/codec/hash manifest.
- **v2** (KMC v0.3): Adds optional tensor-aware entries (TensorEntry) for safetensors files.
- **v3** (KMC v0.4): Adds per-block `codec_metadata`, `tensor_name`, `tensor_dtype`, `tensor_shape` fields for tensor-aware codecs.

Future versions may add:
- New codecs.
- Additional metadata fields.
- Encrypted blocks.
- Block-level checksums beyond SHA-256.

### Backward Compatibility

- **v3 readers** can read v1 and v2 manifests. Missing `codec_metadata`, `tensor_name`, `tensor_dtype`, and `tensor_shape` fields default to empty/zero values.
- **v2 readers** can read v1 manifests. Missing tensor entries default to empty.
- **v1 readers** should not encounter v2/v3 manifests in normal operation, but they can safely ignore unknown fields in JSON.
- **v2/v3 readers encountering v1 manifests** will not have codec metadata available. They should use the legacy `compress_block`/`decompress_block` API for v1 blocks.
- **`byteplane` and `floatplane` codecs require v3 manifests** because their decompression requires `codec_metadata` (transform type, element_size, inner_codec). The legacy `decompress_block` function raises a `ValueError` for these codecs, directing users to the new archive API.

Readers should check the magic number version and the manifest version for compatibility.

## Security Considerations

- **Path traversal**: The unpack operation validates all paths against directory traversal attacks. See [SECURITY_MODEL.md](SECURITY_MODEL.md).
- **Denial of service**: Manifest length is bounded; readers should reject manifests larger than a reasonable limit (e.g., 100 MB).
- **Data integrity**: SHA-256 hashes provide strong integrity guarantees against both accidental corruption and intentional tampering (though they do not provide authentication; see security model).
