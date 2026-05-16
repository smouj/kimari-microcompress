# Architecture

## Design Principles

Kimari MicroCompress is built on a set of core principles that guide every design decision:

1. **Lossless by default**: Every compression operation must be perfectly reversible. There is no "lossy mode" — if a user needs quantization, that's a separate concern handled by tools like GGUF's quantization formats, not by KMC.

2. **Byte-exact verification**: SHA-256 hashes are computed at both the file level and the block level, ensuring that every byte of the original input can be verified after decompression. This dual-level hashing catches both large-scale corruption and subtle bit-flips.

3. **Codec flexibility**: The system supports multiple compression codecs (zstd, zlib, raw, byteplane, floatplane) and can automatically select the best one per-block based on tensor metadata. This means that blocks that don't benefit from compression are stored raw, floating-point blocks get tensor-aware codecs, and highly compressible blocks get the full benefit of zstd or zlib.

4. **Block-oriented design**: Files are split into fixed-size blocks (default 256 KiB) before compression. This enables future features like partial decompression (block-loading), parallel compression/decompression, and fine-grained integrity verification.

5. **Manifest-first metadata**: All metadata is stored in a single JSON manifest at the beginning of the archive. This allows tools to inspect the archive without decompressing any data, and makes the format human-readable and debuggable.

6. **Tensor-aware codecs**: BytePlane and FloatPlane codecs exploit the internal structure of floating-point data (byte positions, sign/exponent/mantissa bits) to improve compressibility before applying an inner codec (zstd or zlib). The automatic selector chooses the best codec per block based on dtype.

7. **Optional dependencies**: Features that require external packages (safetensors, zipnn) degrade gracefully when those packages are not installed. Core functionality never depends on optional packages.

## Module Structure

### `archive.py` — Core Operations

The `archive` module implements the three fundamental operations on `.kmc` archives:

- **`pack(source, output, tensor_aware=False)`**: Reads files from the source, splits them into blocks (optionally aligned to tensor boundaries), compresses each block, computes hashes, and writes the complete archive including the manifest. When `tensor_aware=True`, safetensors files are inspected for tensor metadata and block boundaries are adjusted to avoid splitting tensors across blocks where reasonable.
- **`unpack(archive, output_dir)`**: Reads the manifest, then for each file entry, reads and decompresses blocks in order, verifies hashes, and writes the reconstructed files. Includes path traversal protection.
- **`verify(archive)` / `verify_full(archive)`**: Reads the manifest and checks every block hash and file hash. `verify()` returns a list of errors; `verify_full()` returns a structured `VerificationReport`.

The archive format uses a simple sequential layout: magic bytes, manifest length, manifest, then block data. Offsets in the manifest point directly to block positions within the file, enabling random access to individual blocks.

### `codecs/` — Compression Codec Subpackage (v0.4+)

The codec subpackage is the home for all compression and transformation codecs in KMC. It replaces the flat `codecs.py` module with a structured, extensible architecture that supports tensor-aware codecs.

#### `codecs/base.py` — Protocol and Data Structures

Defines the `Codec` protocol and supporting types:

- **`CodecContext`**: A dataclass carrying tensor-aware hints — `dtype`, `shape`, `tensor_name`, `file_path`, `original_size`, and `block_index`. Codecs use context to make informed decisions (e.g., BytePlane uses `dtype` to determine `element_size`).
- **`CodecResult`**: A dataclass for compression/decompression output — `codec` name, `payload` bytes, `original_size`, `compressed_size`, and `metadata` dict. The `metadata` field stores codec-specific parameters (transform type, element_size, inner_codec) needed for lossless decompression.
- **`Codec` protocol**: Defines the `compress(data, *, context)` and `decompress(payload, *, context)` interface with a guaranteed lossless roundtrip: `decompress(compress(data)) == data`.

#### `codecs/byteplane.py` — BytePlane Codec

Lossless byte-plane separation for fixed-width numeric types (BF16/FP16/FP32).

**How it works:**
1. Determines `element_size` from `CodecContext.dtype` (2 for BF16/FP16, 4 for FP32).
2. Separates bytes by their position within each element: for FP32 data `[a0,b0,c0,d0,a1,b1,c1,d1,...]`, produces four planes `[a0,a1,...]`, `[b0,b1,...]`, `[c0,c1,...]`, `[d0,d1,...]`.
3. Concatenates all planes and compresses with an inner codec (zstd preferred, zlib fallback).
4. Misaligned tail bytes (data length not divisible by `element_size`) are stored separately.

**Why it helps:** Bytes at the same position within floating-point numbers tend to have similar patterns — sign bits cluster, exponent bytes cluster, mantissa bytes cluster. This makes the concatenated planes more compressible than interleaved data.

#### `codecs/floatplane.py` — FloatPlane Codec

Lossless sign/exponent/mantissa bit-level separation for FP16/BF16/FP32.

**How it works:**
1. Determines dtype from `CodecContext.dtype` and looks up the bit layout (e.g., BF16: 1 sign + 8 exponent + 7 mantissa bits).
2. Reads each element as an unsigned integer (no float conversion) and extracts sign, exponent, and mantissa bit fields.
3. Packs each component separately: sign bits are bit-packed (8 per byte), exponents and mantissas use minimal byte widths.
4. Compresses each plane independently with an inner codec (zstd or zlib).
5. Payload format: `[sign_len][sign_data][exp_len][exp_data][mantissa_len][mantissa_data][tail_len][tail]`.

**Fallback behavior:** If dtype is not provided or not supported, FloatPlane falls back to BytePlane internally and records `"transform": "byteplane_fallback"` in metadata.

**Why it helps:** Sign bits are often uniform (mostly positive weights), exponents cluster in a narrow range, and mantissa bits have varying entropy. Separating these components allows the inner codec to compress each more efficiently.

#### `codecs/registry.py` — Codec Registry

A central registry for all available codecs, providing:

- `register_codec(name, cls)`: Register a custom codec by name.
- `get_codec(name, **kwargs)`: Instantiate a codec by name with optional configuration.
- `list_codecs()`: List all registered codec names.
- `is_codec_available(name)`: Check if a codec's dependencies are installed.
- `available_codecs()`: List only codecs with dependencies met.

Currently registered codecs: `raw`, `zlib`, `zstd`, `byteplane`, `floatplane`.

#### `codecs/selector.py` — Automatic Codec Selector

Selects the best codec per block based on tensor metadata:

- **Candidate chains**: dtype-specific ordered lists of codecs to try:
  - BF16/FP16/FP32: `floatplane → byteplane → zstd → zlib → raw`
  - INT8/INT16/INT32/UINT*: `zstd → zlib → raw`
  - GGUF files: `zstd → zlib → raw`
  - Unknown dtype: `zstd → zlib → raw`
- **Selection process**: For each candidate, compress the data, verify the roundtrip (decompress matches original), and record the result. The smallest compressed result wins.
- **Forced codec**: The `--codec` CLI flag overrides the automatic selection, trying only the specified codec.
- **Fallback**: If no codec succeeds, raw passthrough is used.
- **`SelectionResult`**: Returns the best `CodecResult`, codec name, candidates tried, and roundtrip verification status.

#### `codecs/legacy.py` — Legacy Codec Interface

Preserves the original `CodecId` enum and `compress_block`/`decompress_block` functions used by v0.2/v0.3 archives. New code should use the codec subpackage directly. The legacy module raises a `ValueError` for `byteplane` and `floatplane` codecs, directing users to the new archive API that provides codec metadata.

#### `codecs/raw.py`, `codecs/zlib_codec.py`, `codecs/zstd_codec.py` — Standard Codecs

These implement the `Codec` protocol for passthrough, zlib, and zstd compression respectively. They are used both directly and as inner codecs by BytePlane and FloatPlane.

### `manifest.py` — Archive Metadata

The manifest uses Python dataclasses with a clear hierarchy: `KMCManifest` contains `FileEntry` objects, each of which contains `BlockEntry` objects and optionally `TensorEntry` objects. The manifest is serialized to JSON for human readability and forward compatibility.

Key design choices:
- POSIX-style paths are used for cross-platform compatibility.
- The manifest version field distinguishes between v1 (original), v2 (tensor-aware), and v3 (per-block codec metadata) formats.
- The `tool` and `tool_version` fields enable provenance tracking.
- `TensorEntry` records tensor name, dtype, shape, byte_offset, and byte_size for safetensors files.
- v3 `BlockEntry` adds `codec_metadata` (dict for codec-specific reconstruction parameters), `tensor_name`, `tensor_dtype`, and `tensor_shape` fields.
- v3 manifests are backward-compatible with v1/v2 readers (new fields default to empty/zero).

### `hashing.py` — Integrity Verification

The hashing module provides SHA-256 computation for bytes, files, and blocks. The file-level hash covers the complete original (uncompressed) file, while block-level hashes cover the compressed block data. This dual approach means:
- Block hashes can be verified without decompressing (fast).
- File hashes verify the complete reconstructed data (thorough).

### `formats/safetensors.py` — Safetensors Format Support

This dedicated module provides comprehensive safetensors support:

- **Header parsing**: Reads the 8-byte header length prefix and JSON header without loading tensor data.
- **Tensor metadata extraction**: For each tensor, extracts name, dtype, shape, byte_offset, and byte_size.
- **Shard detection**: Identifies files matching `model-NNNN-of-MMMM.safetensors` and checks for `model.safetensors.index.json`.
- **LoRA/PEFT detection**: Detects LoRA adapters by examining tensor names (lora_A, lora_B patterns) and naming conventions. Extracts rank and target modules.
- **Graceful degradation**: If the `safetensors` package is not installed, falls back to a pure-Python header parser that reads the JSON header directly from the file.
- **No weight loading**: No tensor data is ever loaded into memory. Only metadata is read.
- **No pickle usage**: The module never uses pickle or any other insecure deserialization method.

### `formats/gguf.py` — GGUF Format Support

This module provides minimal GGUF header parsing:

- **Magic detection**: Reads the 4-byte magic and determines endianness (little-endian or big-endian).
- **Version parsing**: Reads the GGUF format version (1, 2, or 3).
- **Header fields**: Extracts tensor_count and metadata_kv_count.
- **No full file loading**: Only the minimum bytes needed for the header are read.
- **No tensor metadata parsing**: Full tensor descriptor parsing is planned for a future release.

### `inspector.py` — AI Model Format Detection

The inspector module identifies AI model formats by examining file magic bytes and structure. It uses the dedicated format modules (`formats/safetensors.py`, `formats/gguf.py`) when available, with fallbacks for when they are not. The module also provides directory-level inspection that aggregates results across all files to detect model type, sharding, LoRA adapters, and tensor summaries.

### `tensor_inspector.py` — Legacy Safetensors Metadata

This module provides the original safetensors header parsing functionality. It remains for backward compatibility but the primary implementation has moved to `formats/safetensors.py`. New code should use `formats.safetensors.read_safetensors_info()` instead.

### `benchmark.py` — Performance Benchmarking

The benchmark module measures KMC performance and compares it against other tools:

- **Codec comparison**: Benchmarks raw, zlib, and zstd codecs on a 1 MB sample.
- **KMC pipeline**: Measures full pack, verify, and unpack timing.
- **ZipNN comparison**: Optionally compares against ZipNN on compatible files (safetensors, .bin).
- **Environment metadata**: Records Python version, OS, CPU, RAM, KMC version, and dependency versions for reproducibility.
- **Honest reporting**: No invented benchmarks or superiority claims. Results are measurements.

### `gguf.py` — Legacy GGUF Module

This module provides the original GGUF header parsing functionality. The primary implementation has moved to `formats/gguf.py`. It remains for backward compatibility.

## Data Flow

### Pack Operation

```
Source files → Read in blocks → Compress each block → Compute block hash
                                    ↓
                            Select best codec (zstd/zlib/raw)
                                    ↓
                    [Optional: Tensor-aware block boundaries]
                                    ↓
                            Build manifest with offsets and hashes
                            [Optional: Add tensor entries]
                                    ↓
                            Write: Magic + Manifest + Block data
```

### Unpack Operation

```
Read: Magic + Manifest + Block data
                ↓
        For each file entry:
            For each block:
                Seek to offset → Read compressed data
                    ↓
                Verify block hash (SHA-256)
                    ↓
                Decompress using codec from manifest
                    ↓
                Write to output file
                    ↓
            Verify file hash (SHA-256)
```

### Tensor-Aware Pack Operation

```
Source safetensors file
        ↓
    Read header (JSON)
        ↓
    Extract tensor metadata
    (name, dtype, shape, offsets, sizes)
        ↓
    Compute block boundaries
    aligned to tensor boundaries
        ↓
    For each block:
        Build CodecContext (dtype, shape, tensor_name)
            ↓
        Select codec via selector.py
        (dtype-based candidate chain)
            ↓
        Compress + verify roundtrip
            ↓
        Record codec_metadata in BlockEntry
        ↓
    Build manifest with
    TensorEntry + codec_metadata
        ↓
    Write archive
```

## Future Architecture Considerations

### Tensor-Specific Codecs (v0.4 — Completed)

The codec subpackage introduced in v0.4 provides the first tensor-aware codecs:

- **BytePlane**: Byte-plane separation for BF16/FP16/FP32 data.
- **FloatPlane**: Sign/exponent/mantissa bit-level separation for FP16/BF16/FP32.
- **Automatic selector**: dtype-based candidate chains with roundtrip verification.

Future codec improvements may include:
- **Tensor-specific dictionaries**: Build zstd dictionaries from tensors of the same shape and dtype across the model.
- **XOR delta encoding**: Encode differences between adjacent rows/columns of weight matrices.
- **Quantization-aware compression**: Exploit the structure of GGUF quantization levels.

### Block-Level Loading

The current design stores all blocks sequentially, but the manifest already contains per-block offsets and codec metadata. A future "block server" could serve individual blocks on demand, enabling:
- Loading specific layers of a model without downloading the entire file.
- Streaming decompression for large models.
- Memory-mapped access to specific tensor regions.
- **Runtime compressed loading**: Decompress blocks on-the-fly during inference (research phase).

> **Important:** Block-level loading does NOT reduce inference VRAM. The decompressed blocks still occupy the same memory. Runtime compressed loading (keeping blocks compressed in memory) is future research.

### Parallel Processing

The block-oriented design naturally supports parallel compression and decompression. Future versions could use Python's `concurrent.futures` or `multiprocessing` to process blocks in parallel, significantly improving throughput on multi-core systems.

### Format-Aware Compression

With the inspector and tensor_inspector modules, KMC can detect model formats and potentially apply format-specific optimizations:
- Aligning block boundaries with tensor boundaries in safetensors.
- Skipping already-compressed quantized data in GGUF.
- Applying delta compression for LoRA adapters relative to their base models.
