# KMC Format Specification

**Version:** 6
**Status:** Experimental

## Overview

The `.kmc` format is a block-oriented archive format designed for reversible, verifiable, lossless compression of AI model files. It prioritizes integrity checking, human-readable metadata, and future extensibility.

## File Layout

```
+----------------------------------------------+
| Header                                        |
|   Magic: "KMC\x00\x01\x00\x00\x00"  (8 B)   |
|   Manifest length: uint64 BE         (8 B)   |
+----------------------------------------------+
| Manifest                                      |
|   JSON document (UTF-8 encoded)     (variable)|
+----------------------------------------------+
| Block Data                                    |
|   Concatenated compressed blocks    (variable)|
+----------------------------------------------+
```

All multi-byte integers are stored in **big-endian** byte order unless otherwise specified.

## Header

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 8 | Magic | `KMC\x00\x01\x00\x00\x00` -- identifies the file as KMC format version 1 |
| 8 | 8 | Manifest Length | Length of the JSON manifest in bytes (uint64, big-endian) |

Total header size: **16 bytes**.

### Magic Number

The magic number encodes both the format identifier and the format version:
- Bytes 0-2: `"KMC"` -- format identifier
- Byte 3: `\x00` -- separator
- Bytes 4-5: `\x01\x00` -- format version 1 (uint16 big-endian)
- Bytes 6-7: `\x00\x00` -- reserved for future use

## Manifest

The manifest is a JSON document encoded in UTF-8. It describes all files, blocks, compression parameters, and artifact metadata.

### Manifest Schema (v6)

```json
{
  "version": 6,
  "tool": "kimari-microcompress",
  "tool_version": "0.7.0-alpha",
  "created_at": "2025-01-01T00:00:00+00:00",
  "total_original_size": 0,
  "total_compressed_size": 0,
  "artifact_type": "huggingface_model",
  "artifact_metadata": {},
  "format_metadata": {},
  "parallelism": {},
  "index": {
    "version": 1,
    "has_block_offsets": true,
    "has_file_index": true,
    "has_tensor_index": true
  },
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
          "tensor_shape": [768, 2304],
          "archive_offset": 5678
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

### Manifest Schema (v4)

```json
{
  "version": 4,
  "tool": "kimari-microcompress",
  "tool_version": "0.5.0-alpha",
  "created_at": "2025-01-01T00:00:00+00:00",
  "total_original_size": 0,
  "total_compressed_size": 0,
  "artifact_type": "huggingface_model",
  "artifact_metadata": {},
  "format_metadata": {},
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

### Manifest Schema (v1 -- Legacy)

For reference, the original v1 manifest schema is shown below. V4 readers should handle v1, v2, and v3 manifests gracefully by defaulting missing fields to empty/zero values.

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
| `version` | integer | Format version (currently 6) |
| `tool` | string | Tool name that created the archive |
| `tool_version` | string | Version of the tool |
| `created_at` | string | ISO 8601 timestamp of archive creation |
| `total_original_size` | integer | Sum of all original file sizes in bytes |
| `total_compressed_size` | integer | Sum of all compressed block sizes in bytes |
| `artifact_type` | string | Artifact classification (v4, optional). One of: `"huggingface_model"`, `"gguf_model"`, `"lora_adapter"`, `"training_checkpoint"`, `"unknown"` |
| `artifact_metadata` | object | Artifact-specific metadata (v4, optional). See artifact metadata schemas below. |
| `format_metadata` | object | Format-specific metadata (v4, optional). See format metadata schemas below. |
| `parallelism` | object | Parallelism metadata (v5, optional). Contains `created_with_jobs` and `deterministic_order`. |
| `index` | object | Partial access index metadata (v6, optional). See index metadata schema below. |
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
| `codec_metadata` | object | Codec-specific parameters for decompression (v3+, optional) |
| `tensor_name` | string | Name of the tensor this block belongs to (v3+, optional) |
| `tensor_dtype` | string | Dtype of the tensor, e.g. `"BF16"`, `"FP16"`, `"FP32"` (v3+, optional) |
| `tensor_shape` | array | Shape of the tensor as list of integers (v3+, optional) |
| `archive_offset` | integer | Physical byte offset of this block in the `.kmc` file (v6+, optional, 0 = not set) |

### Index Metadata Schema (v6)

The `index` field at the top level records the availability of partial access features. This field is added by KMC v0.7+ during packing and enables readers to quickly determine which partial access capabilities are available without scanning the entire manifest.

| Field | Type | Description |
|-------|------|-------------|
| `version` | integer | Index metadata version (currently 1) |
| `has_block_offsets` | boolean | Whether block entries have `archive_offset` values stored directly |
| `has_file_index` | boolean | Whether file-level index data is available |
| `has_tensor_index` | boolean | Whether tensor-level index data is available (requires `--tensor-aware` packing) |

When `has_block_offsets` is `true`, blocks can be accessed directly by seeking to their `archive_offset` without offset reconstruction. When `false` (older archives), the reader reconstructs offsets by computing cumulative block positions from the archive header.

When `has_tensor_index` is `false`, tensor-level partial access (`read_tensor`, `--tensor`) is unavailable for this archive, but file-level partial access (`read_file`, `--only`) still works.

### Artifact Metadata Schemas (v4)

The `artifact_metadata` field carries artifact-specific information. The schema depends on the `artifact_type`.

#### LoRA Adapter Metadata

When `artifact_type` is `"lora_adapter"`:

| Field | Type | Description |
|-------|------|-------------|
| `artifact_type` | string | `"lora_adapter"` |
| `base_model_name_or_path` | string | Base model reference from `adapter_config.json` (or `"unknown"`) |
| `peft_type` | string | PEFT type from config (e.g., `"LORA"`, or `"unknown"`) |
| `r` | integer | LoRA rank (if available from config or inferred from tensors) |
| `target_modules` | array | Target module names from config (e.g., `["q_proj", "v_proj"]`) |

#### Training Checkpoint Metadata

When `artifact_type` is `"training_checkpoint"`:

| Field | Type | Description |
|-------|------|-------------|
| `artifact_type` | string | `"training_checkpoint"` |
| `step` | integer | Training step number (inferred from directory name or global_step.json) |
| `has_optimizer_state` | boolean | Whether optimizer.pt was detected |
| `has_scheduler_state` | boolean | Whether scheduler.pt was detected |
| `has_rng_state` | boolean | Whether rng_state.pth was detected |
| `has_trainer_state` | boolean | Whether trainer_state.json was detected |

#### GGUF Model Metadata

When `artifact_type` is `"gguf_model"`, metadata is primarily in `format_metadata["gguf"]` (see below). The `artifact_metadata` field may be empty or absent.

### Format Metadata Schemas (v4)

The `format_metadata` field carries format-specific information parsed from the model files.

#### GGUF Format Metadata

When the archive contains GGUF files, `format_metadata["gguf"]` contains:

| Field | Type | Description |
|-------|------|-------------|
| `version` | integer | GGUF format version (1, 2, or 3) |
| `endianness` | string | `"little"` or `"big"` |
| `tensor_count` | integer | Number of tensors in the GGUF file |
| `metadata_kv_count` | integer | Number of metadata key-value pairs |
| `quantization_summary` | object | Dict mapping quantization type name to count (e.g., `{"Q4_K": 201, "F32": 1}`) |
| `tensor_names` | array | List of tensor names (up to 100) |
| `parse_warnings` | array | Warnings encountered during GGUF parsing (if any) |

#### Safetensors Format Metadata

When the archive contains safetensors files, `format_metadata["safetensors"]` contains:

| Field | Type | Description |
|-------|------|-------------|
| `is_sharded` | boolean | Whether the model is sharded |
| `shard_count` | integer | Number of shard files (if sharded) |
| `tensor_count` | integer | Number of tensors in the safetensors file |
| `dtypes` | array | List of dtype strings (e.g., `["F32", "BF16"]`) |

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

When `--gguf-aware` mode is enabled and a GGUF file contains quantized tensors:
- Blocks corresponding to quantized tensors skip floatplane and byteplane codecs.
- Only `zstd -> zlib -> raw` are attempted for quantized tensor blocks.
- Blocks corresponding to F32/F16/BF16 tensors use the normal candidate chain.

This per-block codec selection ensures that:
- Already-compressed or random data is not expanded.
- Each block gets the most appropriate treatment.
- The archive never exceeds the original data size by more than a negligible overhead.
- Tensor-aware codecs (BytePlane, FloatPlane) are automatically selected when they produce better results.
- Quantized GGUF tensors are not subjected to float-aware transforms that would not benefit them.

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
- This is fast and does not require decompression.

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

- **v1** (KMC v0.1-v0.2): Basic file/block/codec/hash manifest.
- **v2** (KMC v0.3): Adds optional tensor-aware entries (TensorEntry) for safetensors files.
- **v3** (KMC v0.4): Adds per-block `codec_metadata`, `tensor_name`, `tensor_dtype`, `tensor_shape` fields for tensor-aware codecs.
- **v4** (KMC v0.5): Adds `artifact_type`, `artifact_metadata`, `format_metadata` fields for artifact-aware workflows.
- **v5** (KMC v0.6): Adds `parallelism` field for tracking worker count and deterministic order guarantee.
- **v6** (KMC v0.7): Adds `index` field for partial access metadata and `archive_offset` field on block entries for direct block access.

Future versions may add:
- New codecs.
- Additional metadata fields.
- Encrypted blocks.
- Block-level checksums beyond SHA-256.

### Backward Compatibility

- **v4 readers** can read v1, v2, and v3 manifests. Missing `artifact_type`, `artifact_metadata`, and `format_metadata` fields default to `"unknown"`, `{}`, and `{}` respectively.
- **v5 readers** can read v1 through v4 manifests. Missing `parallelism` field defaults to `{}`.
- **v6 readers** can read v1 through v5 manifests. Missing `index` field defaults to `{}`. Missing `archive_offset` on block entries defaults to `0`; the reader reconstructs offsets from the archive layout when needed.
- **v3 readers** can read v1 and v2 manifests. Missing `codec_metadata`, `tensor_name`, `tensor_dtype`, and `tensor_shape` fields default to empty/zero values.
- **v2 readers** can read v1 manifests. Missing tensor entries default to empty.
- **v1 readers** should not encounter v2/v3/v4 manifests in normal operation, but they can safely ignore unknown fields in JSON.
- **v2/v3/v4 readers encountering v1 manifests** will not have codec metadata available. They should use the legacy `compress_block`/`decompress_block` API for v1 blocks.
- **`byteplane` and `floatplane` codecs require v3+ manifests** because their decompression requires `codec_metadata` (transform type, element_size, inner_codec). The legacy `decompress_block` function raises a `ValueError` for these codecs, directing users to the new archive API.
- **v4 `artifact_type` and `format_metadata` fields are ignored by v1/v2/v3 readers** because they are additional top-level fields that do not affect block decompression.
- **v6 `index` and `archive_offset` fields are ignored by v1/v2/v3/v4/v5 readers** because they are additive fields that do not affect block decompression or full unpack operations. Older readers simply do not use the partial access features.

Readers should check the magic number version and the manifest version for compatibility.

## Security Considerations

- **Path traversal**: The unpack operation validates all paths against directory traversal attacks. See [SECURITY_MODEL.md](SECURITY_MODEL.md).
- **Denial of service**: Manifest length is bounded; readers should reject manifests larger than a reasonable limit (e.g., 100 MB).
- **Data integrity**: SHA-256 hashes provide strong integrity guarantees against both accidental corruption and intentional tampering (though they do not provide authentication; see security model).
- **No pickle deserialization**: KMC never deserializes pickle-based files. Pickle files in checkpoints (optimizer.pt, training_args.bin, etc.) are compressed as raw bytes only.
