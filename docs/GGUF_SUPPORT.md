# GGUF Support

This document describes KMC's GGUF parsing capabilities and the experimental `--gguf-aware` compression mode.

## Overview

GGUF (GPT-Generated Unified Format) is the standard binary format for quantized models used by llama.cpp. KMC v0.5 adds full GGUF tensor metadata parsing and an experimental GGUF-aware compression mode that adjusts codec selection based on the quantization types present in the file.

**Important disclaimers:**
- GGUF-aware compression is **experimental** and may change in future releases.
- KMC does **not** reduce inference VRAM. Compressed GGUF files must be fully decompressed before loading.
- Quantized tensors within GGUF files are already compressed; additional lossless compression yields limited benefit on those tensors.
- KMC is **lossless only**. It does not modify, requantize, or alter GGUF data in any way.

## GGUF Tensor Metadata Parser

### What It Parses

The GGUF parser (`src/kmc/formats/gguf.py`) reads the following information from GGUF v2/v3 files:

| Field | Description |
|-------|-------------|
| Magic | 4-byte identifier (`"GGUF"`) |
| Version | Format version (1, 2, or 3) |
| Endianness | `"little"` or `"big"`, determined from magic byte order |
| Tensor count | Number of tensors in the file |
| Metadata KV count | Number of metadata key-value pairs |
| Per-tensor metadata | Name, shape, GGML type, offset, estimated byte size |
| Quantization summary | Dict mapping quantization type name to tensor count |

### Supported GGML Types

| Type ID | Name | Description |
|---------|------|-------------|
| 0 | F32 | 32-bit float |
| 1 | F16 | 16-bit float |
| 2 | Q4_0 | 4-bit quantization (block 32) |
| 3 | Q4_1 | 4-bit quantization with min/max |
| 6 | Q5_0 | 5-bit quantization (block 32) |
| 7 | Q5_1 | 5-bit quantization with min/max |
| 8 | Q8_0 | 8-bit quantization (block 32) |
| 9 | Q8_1 | 8-bit quantization with min/max |
| 10 | Q2_K | 2-bit K-quant |
| 11 | Q3_K | 3-bit K-quant |
| 12 | Q4_K | 4-bit K-quant |
| 13 | Q5_K | 5-bit K-quant |
| 14 | Q6_K | 6-bit K-quant |
| 15 | Q8_K | 8-bit K-quant |
| 16-23 | IQ2/IQ3/IQ4 | Various integer quantization |
| 24 | I8 | 8-bit integer |
| 25 | I16 | 16-bit integer |
| 26 | I32 | 32-bit integer |
| 27 | F64 | 64-bit float |
| 29 | BF16 | Brain float 16 |

### Partial Parsing

If the GGUF file is malformed or truncated, the parser degrades gracefully:

- Partially parsed tensor metadata is returned with warnings.
- A message like `"Partially parsed tensor metadata: got 5/201 tensors"` is included.
- Parsing is capped at 100,000 tensors and 10,000 metadata KV pairs to prevent denial-of-service.

### Usage

#### Programmatic

```python
from kmc.formats.gguf import read_gguf_info, is_gguf_file

# Check if a file is GGUF
if is_gguf_file("./model.gguf"):
    # Parse full tensor metadata
    info = read_gguf_info("./model.gguf", parse_tensors=True)

    print(f"Version: {info.version}")
    print(f"Tensors: {info.tensor_count}")
    print(f"Endianness: {info.endianness}")

    # Quantization summary
    for qtype, count in info.quantization_summary.items():
        print(f"  {qtype}: {count} tensors")

    # Per-tensor details
    for t in info.tensors[:10]:
        print(f"  {t.name}: {t.ggml_type} {t.shape} ({t.estimated_size} bytes)")

    # Parse header only (faster, no tensor descriptors)
    info = read_gguf_info("./model.gguf", parse_tensors=False)
```

#### CLI

```bash
# Inspect a GGUF file with tensor details
kmc inspect ./model.gguf --gguf --tensors

# Inspect with JSON output
kmc inspect ./model.gguf --gguf --json

# Inspect without tensor details (faster)
kmc inspect ./model.gguf --gguf
```

#### Example Output

```
KMC GGUF Inspection

Detected type: GGUF model
Version: 3
Endianness: little
Tensors: 201
Metadata KV pairs: 19
File size: 4.37 GB

Quantization summary:
  BF16: 1
  F32: 1
  Q4_K: 199

Tensors (showing 10/201):
  token_embd.weight: Q4_K [32000, 4096] (79.69 MB)
  output_norm.weight: F32 [4096] (16.00 KB)
  output.weight: Q4_K [32000, 4096] (79.69 MB)
  blk.0.attn_norm.weight: F32 [4096] (16.00 KB)
  blk.0.attn_q.weight: Q4_K [4096, 4096] (19.50 MB)
  blk.0.attn_k.weight: Q4_K [4096, 4096] (19.50 MB)
  blk.0.attn_v.weight: Q4_K [4096, 4096] (19.50 MB)
  blk.0.attn_output.weight: Q4_K [4096, 4096] (19.50 MB)
  blk.0.ffn_norm.weight: F32 [4096] (16.00 KB)
  blk.0.ffn_gate.weight: Q4_K [4096, 11008] (52.38 MB)
  ... and 191 more tensors
```

## Experimental GGUF-Aware Compression

### How It Works

When `--gguf-aware` is enabled, the pack operation performs additional analysis on GGUF files:

1. **Parse GGUF tensor metadata**: Extract per-tensor name, shape, type, and offset.
2. **Build quantization summary**: Count tensors by quantization type.
3. **Record format metadata**: Store GGUF version, endianness, tensor count, and quantization summary in `format_metadata["gguf"]` in the manifest.
4. **Adjust codec selection**: For quantized tensor blocks, skip floatplane and byteplane codecs (which do not benefit quantized data) and use `zstd -> zlib -> raw` instead. For F32/F16/BF16 tensor blocks, use the normal codec candidate chain.

### Codec Selection Strategy

| Tensor Type | Codec Candidate Chain | Rationale |
|-------------|----------------------|-----------|
| F32, F16, BF16 | `floatplane -> byteplane -> zstd -> zlib -> raw` | Floating-point data benefits from float-aware transforms |
| Q4_K, Q5_0, Q8_0, etc. | `zstd -> zlib -> raw` | Already quantized; float-aware transforms add overhead without benefit |
| Unknown | `zstd -> zlib -> raw` | Conservative default |

### Usage

```bash
# Pack a GGUF file with GGUF-aware mode
kmc pack ./model.gguf ./model.kmc --gguf-aware

# Pack a directory containing GGUF files
kmc pack ./models/ ./models.kmc --gguf-aware

# Combine with tensor-aware mode
kmc pack ./model.gguf ./model.kmc --gguf-aware --tensor-aware

# Pack with a specific codec (overrides GGUF-aware selection)
kmc pack ./model.gguf ./model.kmc --gguf-aware --codec zstd
```

### Manifest Output

When `--gguf-aware` is used, the manifest includes GGUF-specific metadata:

```json
{
  "version": 4,
  "artifact_type": "gguf_model",
  "format_metadata": {
    "gguf": {
      "version": 3,
      "endianness": "little",
      "tensor_count": 201,
      "metadata_kv_count": 19,
      "quantization_summary": {
        "Q4_K": 199,
        "F32": 1,
        "BF16": 1
      },
      "tensor_names": [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_norm.weight",
        "..."
      ]
    }
  }
}
```

### What GGUF-Aware Does NOT Do

- It does **not** skip already-quantized blocks entirely (this is planned for a future release).
- It does **not** implement GGUF-specific compression algorithms.
- It does **not** reduce inference VRAM or change model loading behavior.
- It does **not** modify the GGUF data in any way.
- It does **not** guarantee improved compression ratios over non-GGUF-aware mode.

### When to Use GGUF-Aware Mode

Use `--gguf-aware` when:

- You are compressing GGUF files that contain a mix of quantized and floating-point tensors.
- You want GGUF format metadata (version, quantization summary) recorded in the manifest.
- You want the codec selector to automatically avoid float-aware transforms on quantized data.

It is safe to use `--gguf-aware` even when the source is not a GGUF file -- the flag only affects `.gguf` files.

## Limitations

1. **GGUF v1 tensor metadata parsing is not supported.** Only v2 and v3 files have the tensor info section. GGUF v1 files will still have their header parsed correctly.

2. **Partial parsing is possible.** Malformed or truncated GGUF files may produce partial results with warnings. The parser is designed to be robust rather than strict.

3. **No GGUF-specific compression algorithms.** The `--gguf-aware` flag only adjusts codec selection; it does not implement GGUF-specific compression strategies like skipping already-quantized blocks.

4. **Compression ratios on quantized GGUF files are limited.** Since quantized tensors are already compressed representations, additional lossless compression yields modest gains (typically 5-15%). The main benefit is on metadata, vocabulary, and F32/F16/BF16 tensor sections.

5. **Experimental status.** The `--gguf-aware` flag is experimental and its behavior may change in future releases.
