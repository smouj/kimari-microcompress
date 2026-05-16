# GGUF-Specific Codecs

> **Status**: Experimental (KMC v0.8.0-alpha)
> **Codec name**: `gguf_quant_block`
> **CLI flag**: `--gguf-aware`

## Overview

KMC v0.8.0-alpha introduces an experimental codec path specifically designed for GGUF files containing quantized tensor data. The `gguf_quant_block` codec (implemented in `kmc.codecs.gguf_quant`) applies conservative, format-aware compression strategies that respect the structure of quantized data — unlike the generic codec pipeline, which may inadvertently choose suboptimal transforms for quantized payloads.

The key insight behind the GGUF quantized codec is that **quantized tensor data has fundamentally different compression characteristics than floating-point data**. Quantized values (e.g., Q4_0, Q5_1, Q8_0, Q2_K through Q6_K) are stored as packed integer arrays with associated scale factors. These data patterns respond differently to compression than IEEE 754 floating-point values:

- **Quantized payloads** (packed 4-bit, 5-bit, or 8-bit integers) compress well with general-purpose algorithms like zstd and zlib due to the repetitive patterns in weight distributions.
- **Scale factors** (small metadata arrays associated with each block of quantized values) benefit from byte-plane separation, which exploits the regular structure of fixed-width elements.
- **Floating-point data** (BF16, FP16, FP32) is explicitly excluded from the `gguf_quant_block` codec because FloatPlane compression is more effective for FP data. Applying byte-plane separation to floating-point data would be counterproductive.

## How gguf_quant_block Works

### Compression Pipeline

When the `gguf_quant_block` codec is activated for a block, it tries a sequence of compression strategies and selects the one that produces the smallest output:

1. **zstd compression**: The first candidate tried, and typically the most effective for quantized payloads. Quantized weight data often contains significant redundancy — repeated quantization levels, similar weight distributions across channels — that zstd's LZ77 + entropy coding handles efficiently.

2. **zlib compression**: Tried as a fallback if zstd is not available or if zlib produces a smaller result (rare, but possible for very small blocks where zlib's simpler header overhead is advantageous).

3. **BytePlane separation**: Attempted for fixed-width quantized element types (Q4_0, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8). BytePlane separates the block data into individual byte planes (one plane per byte position within each element), which can improve compression when certain byte positions are more regular than others. This is particularly effective for **scale factor blocks** in quantized data, where the scale values tend to cluster around similar magnitudes.

4. **Raw fallback**: If none of the above strategies produce a smaller output than the original data, the block is stored uncompressed. This is the correct behavior — lossless compression that increases size would be wasteful.

### Codec Selection Logic

The `gguf_quant_block` codec includes a safety check that prevents it from being applied to floating-point data:

```python
# Safety check: don't apply to float dtypes
if context and context.dtype:
    dtype_upper = context.dtype.upper().strip()
    float_dtypes = {
        "BF16", "FP16", "F16", "FP32", "F32", "FP64", "F64",
        "BFLOAT16", "FLOAT16", "FLOAT32", "FLOAT64",
    }
    if dtype_upper in float_dtypes:
        # Fall back to raw — selector should have picked floatplane instead
        return CodecResult(payload=data, ..., metadata={"fallback": "raw", "reason": "float_dtype"})
```

This check ensures that even if `--gguf-aware` is enabled, floating-point tensors within a GGUF file (e.g., the output tensor in FP16) are not compressed with the quantized codec path. The codec selector should route these blocks to FloatPlane instead.

### Metadata Recording

Each compressed block records metadata about the compression process, enabling correct decompression:

```json
{
  "codec": "gguf_quant_block",
  "codec_metadata": {
    "gguf_aware": true,
    "inner_codec": "zstd",
    "candidates_tried": ["zstd", "zlib", "byteplane"]
  }
}
```

| Metadata Key | Description |
|---|---|
| `gguf_aware` | Always `true` when this codec is used |
| `inner_codec` | The actual compression method that produced the best result (`"zstd"`, `"zlib"`, `"byteplane"`, or absent for raw) |
| `candidates_tried` | List of codecs attempted (may include `":failed"` suffixes for errors) |
| `byteplane_meta` | Additional metadata from BytePlane (if byteplane was selected) |

### Decompression Pipeline

Decompression uses the `inner_codec` metadata to determine which decompressor to apply:

1. If `inner_codec` is `"raw"` or metadata is absent, the payload is returned as-is (no decompression needed).
2. If `inner_codec` is `"zstd"`, the `ZstdCodec.decompress()` method is called.
3. If `inner_codec` is `"zlib"`, the `ZlibCodec.decompress()` method is called.
4. If `inner_codec` is `"byteplane"`, the `BytePlaneCodec.decompress()` method is called.
5. Any other `inner_codec` value raises a `ValueError`.

The decompression context must include `codec_metadata` from the compression phase; otherwise, `GGUFQuantCodec.decompress()` raises a `ValueError`.

## When the Codec Is Selected

The `gguf_quant_block` codec is only selected under specific conditions:

### Activation via --gguf-aware

The `--gguf-aware` CLI flag sets `gguf_aware=True` in the `CodecContext` for blocks from GGUF files. When the codec selector encounters a block with `gguf_aware=True` and a quantized dtype, it routes the block to `GGUFQuantCodec` instead of the default candidate chain.

### Codec Context Requirements

For the GGUF quantized codec to be selected, the `CodecContext` must include:

- `gguf_aware=True` — Set by the `--gguf-aware` flag during packing
- A `dtype` that is a recognized quantized type (not a floating-point type)

If `gguf_aware` is not set, blocks from GGUF files are compressed using the standard codec selection pipeline (zstd/zlib/raw), which still produces valid compressed output but may not be optimal for quantized data patterns.

### Default Codec Candidates for GGUF

When `--gguf-aware` is **not** used, the codec selector's `get_candidates()` function returns `["zstd", "zlib", "raw"]` for GGUF files (detected by the `.gguf` file extension). This is the same as the default candidate list for unknown dtypes. The `gguf_quant_block` codec is not in this list because it requires explicit activation.

## What the Codec Does NOT Do

It is critical to understand the boundaries of the `gguf_quant_block` codec:

### No Data Modification or Reinterpretation

The codec **never modifies or reinterprets the data bytes**. It applies lossless compression transforms (zstd, zlib, byteplane) to the raw byte representation of the block data. The decompressed output is guaranteed to be bit-for-bit identical to the original input. There is no dequantization, requantization, or type conversion.

### No FloatPlane for Quantized Data

The `gguf_quant_block` codec explicitly **never** uses FloatPlane compression for quantized data. FloatPlane is designed for floating-point values (BF16, FP16, FP32) where byte-plane separation of IEEE 754 representations can exploit the shared exponent patterns. For quantized integer data, this transform would be counterproductive because quantized values do not have the same byte-level structure as floating-point numbers.

### No Assumptions About GGML Types Without Metadata

The codec does not make assumptions about the internal structure of GGML quantization types. It does not parse the quantized block format (e.g., the super-block structure of K-quants or the block structure of IQ types). It treats each block as opaque bytes and applies general-purpose compression. This conservative approach ensures correctness but may miss opportunities for format-specific optimizations in future versions.

## How to Activate

### CLI

```bash
# Enable GGUF-aware compression
kmc pack ./my-gguf-model.gguf ./model.kmc --gguf-aware

# Combine with tensor-aware mode for best results
kmc pack ./my-gguf-model.gguf ./model.kmc --gguf-aware --tensor-aware

# Combine with dedup for multi-file GGUF archives
kmc pack ./gguf-models/ ./models.kmc --gguf-aware --dedup
```

### Python API

```python
from kmc.archive import pack
from pathlib import Path

pack(
    Path("./my-gguf-model.gguf"),
    Path("./model.kmc"),
    gguf_aware=True,
    tensor_aware=True,
)
```

## Fallback Behavior

When `--gguf-aware` is enabled but the GGUF quantized codec encounters conditions where it cannot improve compression, it falls back gracefully:

| Condition | Behavior |
|---|---|
| Block dtype is floating-point | Returns raw data with `{"fallback": "raw", "reason": "float_dtype"}` |
| No inner codec reduces size | Returns raw data with `{"gguf_aware": True, "inner_codec": "raw"}` |
| zstd not available | Skips zstd candidate, tries zlib and byteplane |
| All codecs fail | Falls back to raw (guaranteed to succeed) |

The fallback chain ensures that `--gguf-aware` never produces larger archives than packing without the flag. In the worst case, it behaves identically to the default pipeline.

## Example: Packing a Quantized GGUF Model

```bash
# Pack a Q4_K_M quantized model
kmc pack ./llama-7b-Q4_K_M.gguf ./llama-7b-Q4_K_M.kmc --gguf-aware

# Inspect the result
kmc inspect ./llama-7b-Q4_K_M.kmc --compression

# Output includes:
#   Codecs: gguf_quant_block, zstd, raw
#   Compression summary:
#     gguf_quant_block: 245 blocks
#     zstd: 3 blocks
#     raw: 2 blocks
```

### Manifest Example

```json
{
  "files": [
    {
      "path": "llama-7b-Q4_K_M.gguf",
      "blocks": [
        {
          "index": 0,
          "codec": "gguf_quant_block",
          "original_size": 262144,
          "compressed_size": 198432,
          "codec_metadata": {
            "gguf_aware": true,
            "inner_codec": "zstd",
            "candidates_tried": ["zstd", "zlib"]
          }
        },
        {
          "index": 1,
          "codec": "gguf_quant_block",
          "original_size": 262144,
          "compressed_size": 262144,
          "codec_metadata": {
            "gguf_aware": true,
            "inner_codec": "raw",
            "candidates_tried": ["zstd:failed", "zlib", "byteplane"]
          }
        }
      ]
    }
  ]
}
```

In this example, block 0 compressed well with zstd (24% reduction), while block 1 was incompressible and stored raw (0% reduction — the correct behavior for incompressible data).
