# Architecture

## Design Principles

Kimari MicroCompress is built on a set of core principles that guide every design decision:

1. **Lossless by default**: Every compression operation must be perfectly reversible. There is no "lossy mode" -- if a user needs quantization, that is a separate concern handled by tools like GGUF's quantization formats, not by KMC.

2. **Byte-exact verification**: SHA-256 hashes are computed at both the file level and the block level, ensuring that every byte of the original input can be verified after decompression. This dual-level hashing catches both large-scale corruption and subtle bit-flips.

3. **Codec flexibility**: The system supports multiple compression codecs (zstd, zlib, raw, byteplane, floatplane) and can automatically select the best one per-block based on tensor metadata. This means that blocks that do not benefit from compression are stored raw, floating-point blocks get tensor-aware codecs, and highly compressible blocks get the full benefit of zstd or zlib.

4. **Block-oriented design**: Files are split into fixed-size blocks (default 256 KiB) before compression. This enables future features like partial decompression (block-loading), parallel compression/decompression, and fine-grained integrity verification.

5. **Manifest-first metadata**: All metadata is stored in a single JSON manifest at the beginning of the archive. This allows tools to inspect the archive without decompressing any data, and makes the format human-readable and debuggable.

6. **Tensor-aware codecs**: BytePlane and FloatPlane codecs exploit the internal structure of floating-point data (byte positions, sign/exponent/mantissa bits) to improve compressibility before applying an inner codec (zstd or zlib). The automatic selector chooses the best codec per block based on dtype.

7. **Optional dependencies**: Features that require external packages (safetensors, zipnn) degrade gracefully when those packages are not installed. Core functionality never depends on optional packages.

8. **No pickle usage**: KMC never uses pickle for inspection or deserialization. Pickle-based files (optimizer.pt, training_args.bin, pytorch_model.bin) are detected by name and compressed as raw bytes only. Their contents are never loaded or inspected.

9. **Artifact-aware workflows**: KMC v0.5 introduces artifact-type detection and specialized workflows for LoRA adapters and training checkpoints. Each artifact type carries its own metadata schema in the manifest, enabling downstream tools to understand what was compressed without inspecting the data.

## Module Structure

### `archive.py` -- Core Operations

The `archive` module implements the three fundamental operations on `.kmc` archives:

- **`pack(source, output, tensor_aware, codec, gguf_aware, artifact_type, artifact_metadata)`**: Reads files from the source, splits them into blocks (optionally aligned to tensor boundaries), compresses each block, computes hashes, and writes the complete archive including the manifest. When `tensor_aware=True`, safetensors files are inspected for tensor metadata and block boundaries are adjusted to avoid splitting tensors across blocks where reasonable. When `gguf_aware=True`, GGUF files are parsed for tensor metadata and codec selection is adjusted for quantized tensors. The `artifact_type` and `artifact_metadata` parameters populate the v4 manifest fields.
- **`unpack(archive, output_dir)`**: Reads the manifest, then for each file entry, reads and decompresses blocks in order, verifies hashes, and writes the reconstructed files. Includes path traversal protection.
- **`verify(archive)` / `verify_full(archive)`**: Reads the manifest and checks every block hash and file hash. `verify()` returns a list of errors; `verify_full()` returns a structured `VerificationReport`.

The archive format uses a simple sequential layout: magic bytes, manifest length, manifest, then block data. Offsets in the manifest point directly to block positions within the file, enabling random access to individual blocks.

#### Artifact Auto-Detection

When `artifact_type` is not explicitly set, the `pack` function auto-detects it by examining the source directory:

1. If a LoRA adapter is detected (via `workflows.lora.detect_lora_adapter`), `artifact_type` is set to `"lora_adapter"`.
2. If a training checkpoint is detected (via `workflows.checkpoint.detect_checkpoint`), `artifact_type` is set to `"training_checkpoint"`.
3. If GGUF files are found, `artifact_type` is set to `"gguf_model"`.
4. If safetensors files are found, `artifact_type` is set to `"huggingface_model"`.
5. Otherwise, `artifact_type` is `"unknown"`.

Format metadata (safetensors, GGUF) is also auto-detected and recorded in `format_metadata`.

### `codecs/` -- Compression Codec Subpackage (v0.4+)

The codec subpackage is the home for all compression and transformation codecs in KMC. It replaces the flat `codecs.py` module with a structured, extensible architecture that supports tensor-aware codecs.

#### `codecs/base.py` -- Protocol and Data Structures

Defines the `Codec` protocol and supporting types:

- **`CodecContext`**: A dataclass carrying tensor-aware hints -- `dtype`, `shape`, `tensor_name`, `file_path`, `original_size`, and `block_index`. Codecs use context to make informed decisions (e.g., BytePlane uses `dtype` to determine `element_size`).
- **`CodecResult`**: A dataclass for compression/decompression output -- `codec` name, `payload` bytes, `original_size`, `compressed_size`, and `metadata` dict. The `metadata` field stores codec-specific parameters (transform type, element_size, inner_codec) needed for lossless decompression.
- **`Codec` protocol**: Defines the `compress(data, *, context)` and `decompress(payload, *, context)` interface with a guaranteed lossless roundtrip: `decompress(compress(data)) == data`.

#### `codecs/byteplane.py` -- BytePlane Codec

Lossless byte-plane separation for fixed-width numeric types (BF16/FP16/FP32).

**How it works:**
1. Determines `element_size` from `CodecContext.dtype` (2 for BF16/FP16, 4 for FP32).
2. Separates bytes by their position within each element: for FP32 data `[a0,b0,c0,d0,a1,b1,c1,d1,...]`, produces four planes `[a0,a1,...]`, `[b0,b1,...]`, `[c0,c1,...]`, `[d0,d1,...]`.
3. Concatenates all planes and compresses with an inner codec (zstd preferred, zlib fallback).
4. Misaligned tail bytes (data length not divisible by `element_size`) are stored separately.

**Why it helps:** Bytes at the same position within floating-point numbers tend to have similar patterns -- sign bits cluster, exponent bytes cluster, mantissa bytes cluster. This makes the concatenated planes more compressible than interleaved data.

#### `codecs/floatplane.py` -- FloatPlane Codec

Lossless sign/exponent/mantissa bit-level separation for FP16/BF16/FP32.

**How it works:**
1. Determines dtype from `CodecContext.dtype` and looks up the bit layout (e.g., BF16: 1 sign + 8 exponent + 7 mantissa bits).
2. Reads each element as an unsigned integer (no float conversion) and extracts sign, exponent, and mantissa bit fields.
3. Packs each component separately: sign bits are bit-packed (8 per byte), exponents and mantissas use minimal byte widths.
4. Compresses each plane independently with an inner codec (zstd or zlib).
5. Payload format: `[sign_len][sign_data][exp_len][exp_data][mantissa_len][mantissa_data][tail_len][tail]`.

**Fallback behavior:** If dtype is not provided or not supported, FloatPlane falls back to BytePlane internally and records `"transform": "byteplane_fallback"` in metadata.

**Why it helps:** Sign bits are often uniform (mostly positive weights), exponents cluster in a narrow range, and mantissa bits have varying entropy. Separating these components allows the inner codec to compress each more efficiently.

#### `codecs/registry.py` -- Codec Registry

A central registry for all available codecs, providing:

- `register_codec(name, cls)`: Register a custom codec by name.
- `get_codec(name, **kwargs)`: Instantiate a codec by name with optional configuration.
- `list_codecs()`: List all registered codec names.
- `is_codec_available(name)`: Check if a codec's dependencies are installed.
- `available_codecs()`: List only codecs with dependencies met.

Currently registered codecs: `raw`, `zlib`, `zstd`, `byteplane`, `floatplane`.

#### `codecs/selector.py` -- Automatic Codec Selector

Selects the best codec per block based on tensor metadata:

- **Candidate chains**: dtype-specific ordered lists of codecs to try:
  - BF16/FP16/FP32: `floatplane -> byteplane -> zstd -> zlib -> raw`
  - INT8/INT16/INT32/UINT*: `zstd -> zlib -> raw`
  - GGUF files: `zstd -> zlib -> raw`
  - Unknown dtype: `zstd -> zlib -> raw`
- **Selection process**: For each candidate, compress the data, verify the roundtrip (decompress matches original), and record the result. The smallest compressed result wins.
- **Forced codec**: The `--codec` CLI flag overrides the automatic selection, trying only the specified codec.
- **Fallback**: If no codec succeeds, raw passthrough is used.
- **`SelectionResult`**: Returns the best `CodecResult`, codec name, candidates tried, and roundtrip verification status.

#### `codecs/legacy.py` -- Legacy Codec Interface

Preserves the original `CodecId` enum and `compress_block`/`decompress_block` functions used by v0.2/v0.3 archives. New code should use the codec subpackage directly. The legacy module raises a `ValueError` for `byteplane` and `floatplane` codecs, directing users to the new archive API that provides codec metadata.

#### `codecs/raw.py`, `codecs/zlib_codec.py`, `codecs/zstd_codec.py` -- Standard Codecs

These implement the `Codec` protocol for passthrough, zlib, and zstd compression respectively. They are used both directly and as inner codecs by BytePlane and FloatPlane.

### `manifest.py` -- Archive Metadata

The manifest uses Python dataclasses with a clear hierarchy: `KMCManifest` contains `FileEntry` objects, each of which contains `BlockEntry` objects and optionally `TensorEntry` objects. The manifest is serialized to JSON for human readability and forward compatibility.

Key design choices:
- POSIX-style paths are used for cross-platform compatibility.
- The manifest version field distinguishes between v1 (original), v2 (tensor-aware), v3 (per-block codec metadata), and v4 (artifact type + format metadata) formats.
- The `tool` and `tool_version` fields enable provenance tracking.
- `TensorEntry` records tensor name, dtype, shape, byte_offset, and byte_size for safetensors files.
- v3 `BlockEntry` adds `codec_metadata` (dict for codec-specific reconstruction parameters), `tensor_name`, `tensor_dtype`, and `tensor_shape` fields.
- v4 `KMCManifest` adds `artifact_type` (string: `huggingface_model|gguf_model|lora_adapter|training_checkpoint|unknown`), `artifact_metadata` (dict with artifact-specific metadata such as LoRA rank), and `format_metadata` (dict with format-specific metadata such as GGUF version and quantization summary).
- v4 manifests are backward-compatible with v1/v2/v3 readers (new fields default to empty/zero/unknown).

### `hashing.py` -- Integrity Verification

The hashing module provides SHA-256 computation for bytes, files, and blocks. The file-level hash covers the complete original (uncompressed) file, while block-level hashes cover the compressed block data. This dual approach means:
- Block hashes can be verified without decompressing (fast).
- File hashes verify the complete reconstructed data (thorough).

### `formats/safetensors.py` -- Safetensors Format Support

This dedicated module provides comprehensive safetensors support:

- **Header parsing**: Reads the 8-byte header length prefix and JSON header without loading tensor data.
- **Tensor metadata extraction**: For each tensor, extracts name, dtype, shape, byte_offset, and byte_size.
- **Shard detection**: Identifies files matching `model-NNNN-of-MMMM.safetensors` and checks for `model.safetensors.index.json`.
- **LoRA/PEFT detection**: Detects LoRA adapters by examining tensor names (lora_A, lora_B patterns) and naming conventions. Extracts rank and target modules.
- **Graceful degradation**: If the `safetensors` package is not installed, falls back to a pure-Python header parser that reads the JSON header directly from the file.
- **No weight loading**: No tensor data is ever loaded into memory. Only metadata is read.
- **No pickle usage**: The module never uses pickle or any other insecure deserialization method.

### `formats/gguf.py` -- GGUF Format Support (v0.5+)

This module provides full GGUF header and tensor metadata parsing:

- **Magic detection**: Reads the 4-byte magic and determines endianness (little-endian or big-endian).
- **Version parsing**: Reads the GGUF format version (1, 2, or 3).
- **Header fields**: Extracts tensor_count and metadata_kv_count.
- **Tensor metadata parsing**: For GGUF v2/v3 files, reads the tensor info section after skipping metadata KV pairs. Extracts per-tensor name, shape, GGML type, offset, and estimated byte size.
- **Quantization summary**: Produces a dict mapping quantization type name (e.g., "Q4_K", "F32") to count of tensors using that type.
- **Type mapping**: Maps GGML type IDs (0-32) to human-readable names (F32, F16, Q4_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, BF16, etc.).
- **Size estimation**: Estimates tensor byte size from type and shape using known block sizes for quantized types.
- **Quantized type detection**: Provides `is_quantized_ggml_type()` to distinguish quantized types (Q4_K, Q5_0, etc.) from floating-point types (F32, F16, BF16).
- **Graceful partial parsing**: If tensor metadata parsing fails partway through, returns partial results with warnings rather than raising an exception.
- **No full file loading**: Only the minimum bytes needed for the header and tensor descriptors are read.
- **Sanity limits**: Tensor count is capped at 100,000 and metadata KV count at 10,000 to prevent denial-of-service on malformed files.

### `workflows/` -- Artifact-Specific Workflows (v0.5+)

The workflows subpackage provides dedicated detection and packing support for specific artifact types. Each workflow module follows the same pattern: detect the artifact type, extract metadata, build manifest fields, and delegate to `archive.pack()` for compression.

#### `workflows/lora.py` -- LoRA/PEFT Adapter Workflow

Provides dedicated support for LoRA/PEFT adapter directories:

- **`detect_lora_adapter(directory)`**: Scans a directory for LoRA adapter files. Looks for:
  - `adapter_model.safetensors` (primary)
  - Any `.safetensors` file with "adapter" in its name
  - Any `.safetensors` file containing LoRA tensor names (lora_A, lora_B)
  - `adapter_config.json` for PEFT configuration
  - Returns a `LoRAAdapterInfo` dataclass with: `is_lora`, `adapter_model_path`, `peft_type`, `lora_rank`, `target_modules`, `base_model_name_or_path`, and `warnings`.
- **`build_lora_manifest_metadata(adapter_info)`**: Builds the `artifact_metadata` dict for a LoRA adapter, containing: `artifact_type: "lora_adapter"`, `base_model_name_or_path`, `peft_type`, `r` (rank), and `target_modules`.

Strict rules:
- Never invents data. Missing fields default to `"unknown"`.
- Never uses pickle. Only `adapter_config.json` (JSON) and safetensors metadata are read.
- Never loads weights. Only metadata is inspected.

#### `workflows/checkpoint.py` -- Training Checkpoint Workflow

Provides dedicated support for Hugging Face training checkpoint directories:

- **`detect_checkpoint(directory)`**: Scans a directory for checkpoint files. Looks for:
  - Known checkpoint file patterns (`trainer_state.json`, `optimizer.pt`, `model.safetensors`, etc.)
  - Step number from directory name (e.g., `checkpoint-1000`) or `global_step.json`
  - Safetensors shard files
  - Returns a `CheckpointInfo` dataclass with: `is_checkpoint`, `step`, component flags (`has_trainer_state`, `has_optimizer_state`, etc.), `detected_files`, and `warnings`.
- **`build_checkpoint_manifest_metadata(ckpt_info)`**: Builds the `artifact_metadata` dict for a checkpoint, containing: `artifact_type: "training_checkpoint"`, `step`, and component flags.

Strict rules:
- Never uses pickle. Pickle-based files are detected by name only; their contents are never loaded.
- `optimizer.pt`, `training_args.bin`, `scheduler.pt`, `rng_state.pth`, `pytorch_model.bin` are compressed as raw bytes.
- Warnings are emitted when pickle-based files are detected, reminding users that only size/hash is recorded.

### `inspector.py` -- AI Model Format Detection

The inspector module identifies AI model formats by examining file magic bytes and structure. It uses the dedicated format modules (`formats/safetensors.py`, `formats/gguf.py`) when available, with fallbacks for when they are not. The module also provides directory-level inspection that aggregates results across all files to detect model type, sharding, LoRA adapters, and tensor summaries.

### `tensor_inspector.py` -- Legacy Safetensors Metadata

This module provides the original safetensors header parsing functionality. It remains for backward compatibility but the primary implementation has moved to `formats/safetensors.py`. New code should use `formats.safetensors.read_safetensors_info()` instead.

### `index/` -- Partial Access Indexes (v0.7+)

The index subpackage provides structured indexes that map between archive blocks, files, and tensors, enabling selective extraction without decompressing the entire archive. Indexes are built from the manifest on open and support efficient lookup by ID, path, or tensor name.

#### `index/block_index.py` -- Block Index

`BlockIndex` maps each compressed block to its physical location in the archive. Each entry is a `BlockLocation` dataclass containing the block ID, file path, tensor name, archive offset, compressed size, original size, codec, codec metadata, and block hash. The index supports lookup by block ID, by file path (returns all blocks for a file), and by tensor name (returns all blocks for a tensor). When the manifest lacks physical block offsets (common in older archives created before v0.7), the `from_manifest()` class method automatically reconstructs offsets by computing cumulative block positions from the archive header.

#### `index/file_index.py` -- File Index

`FileIndex` maps file paths to `FileLocation` objects containing the file's original size, SHA-256 hash, and the ordered list of block IDs that compose it. The index also provides pattern matching via `match_pattern()`, which supports fnmatch-style glob patterns (e.g., `*.json`, `tokenizer*`). This enables efficient file-level selective extraction by identifying which blocks need to be read for a given file or pattern.

#### `index/tensor_index.py` -- Tensor Index

`TensorIndex` maps tensor names to `TensorLocation` objects containing the tensor's dtype, shape, file path, and block IDs. This index is only populated for archives created with `--tensor-aware` mode, since tensor metadata is required to build the mapping. The `available` property returns whether tensor-level access is possible. The index merges information from both file-level `tensor_entries` (from safetensors metadata) and block-level `tensor_name`/`tensor_dtype`/`tensor_shape` fields (from codec context), preferring file-level entries for dtype and shape when both sources are available.

### `reader.py` -- KMCReader Partial Access API (v0.7+)

`KMCReader` is the primary Python API for partial access to `.kmc` archives. It opens an archive, reads the manifest, builds all three indexes (block, file, tensor), and provides methods for listing, reading, and extracting specific files or tensors without full decompression.

Key design decisions for `KMCReader`:

- **Context manager support.** `KMCReader` implements `__enter__`/`__exit__` for automatic resource cleanup. File handles are opened per-operation and closed between reads, avoiding the need for explicit handle management.
- **Per-operation file access.** Rather than keeping the archive file handle open for the lifetime of the reader, each read operation opens the file, seeks to the required offset, reads the block data, and closes the handle. This approach is safe for sequential access and avoids resource leaks if the reader is not properly closed.
- **Block checksum verification.** Every block read is verified against its SHA-256 hash from the manifest before decompression. File-level reads additionally verify the reconstructed file hash after concatenating all decompressed blocks. This ensures that partial reads maintain the same integrity guarantees as full unpack operations.
- **Graceful degradation for tensor access.** When a tensor has no dedicated block mapping (because it was not packed with tensor-aware block alignment), `read_tensor()` attempts to locate the tensor data within its parent file using the file-level `tensor_entries` metadata. This provides best-effort tensor access even when blocks are not individually aligned to tensor boundaries.

### `loaders/` -- Experimental Tensor Loaders (v0.7+)

The loaders subpackage provides experimental functionality for converting tensor bytes into native tensor objects (PyTorch or NumPy). It is optional and depends on external packages that are not required for core KMC functionality.

#### `loaders/safetensors_loader.py` -- Safetensors Tensor Loader

Provides two functions: `load_tensor_bytes()` returns raw bytes (no optional dependencies) and `load_tensor()` returns native tensor objects (requires PyTorch or NumPy). The loader handles dtype mapping from safetensors format strings (BF16, FP16, FP32, etc.) to framework-native dtypes. BF16 tensors require PyTorch since NumPy does not natively support bfloat16. This module is experimental and its API may change without notice.

### `benchmark.py` -- Performance Benchmarking

The benchmark module measures KMC performance and compares it against other tools:

- **Codec comparison**: Benchmarks raw, zlib, and zstd codecs on a 1 MB sample.
- **KMC pipeline**: Measures full pack, verify, and unpack timing.
- **ZipNN comparison**: Optionally compares against ZipNN on compatible files (safetensors, .bin).
- **Environment metadata**: Records Python version, OS, CPU, RAM, KMC version, and dependency versions for reproducibility.
- **Honest reporting**: No invented benchmarks or superiority claims. Results are measurements.

### `gguf.py` -- Legacy GGUF Module

This module provides the original GGUF header parsing functionality. The primary implementation has moved to `formats/gguf.py`. It remains for backward compatibility.

## Data Flow

### Pack Operation

```
Source files -> Read in blocks -> Compress each block -> Compute block hash
                                    |
                            Select best codec (zstd/zlib/raw)
                                    |
                    [Optional: Tensor-aware block boundaries]
                                    |
                            Build manifest with offsets and hashes
                            [Optional: Add tensor entries]
                            [Optional: Add artifact_type + artifact_metadata]
                            [Optional: Add format_metadata (GGUF, safetensors)]
                                    |
                            Write: Magic + Manifest + Block data
```

### Pack Operation with GGUF-Aware Mode

```
Source GGUF file
        |
    Read GGUF header + tensor metadata
    (parse_tensors=True)
        |
    Extract quantization summary
    (e.g., Q4_K: 201, F32: 1)
        |
    Record format_metadata["gguf"] in manifest
        |
    For each block:
        If GGUF tensor is quantized (Q4_K, Q5_0, etc.):
            Skip floatplane/byteplane
            Use zstd or zlib only
        If GGUF tensor is F32/F16/BF16:
            Use normal codec selection
            (floatplane -> byteplane -> zstd -> zlib -> raw)
        |
    Build manifest with artifact_type="gguf_model"
        |
    Write archive
```

### Pack-Lora Operation

```
Source LoRA adapter directory
        |
    detect_lora_adapter(directory)
        |
    Read adapter_config.json
    (peft_type, r, target_modules, base_model_name_or_path)
        |
    Read safetensors metadata
    (tensor names, dtypes, shapes, LoRA rank inference)
        |
    build_lora_manifest_metadata(adapter_info)
        |
    pack(source, output,
         tensor_aware=True,
         artifact_type="lora_adapter",
         artifact_metadata={...})
        |
    Write archive with v4 manifest
```

### Pack-Checkpoint Operation

```
Source checkpoint directory (e.g., checkpoint-1000/)
        |
    detect_checkpoint(directory)
        |
    Infer step from directory name or global_step.json
        |
    Detect components (optimizer, scheduler, RNG, etc.)
        |
    Emit warnings for pickle-based files
        |
    build_checkpoint_manifest_metadata(ckpt_info)
        |
    pack(source, output,
         tensor_aware=has_safetensors_model,
         artifact_type="training_checkpoint",
         artifact_metadata={...})
        |
    Write archive with v4 manifest
```

### Partial Access Operation (v0.7+)

```
Open .kmc archive
        |
    Read manifest
        |
    Build indexes:
    - BlockIndex (archive_offset -> BlockLocation)
    - FileIndex (path -> FileLocation + block_ids)
    - TensorIndex (name -> TensorLocation + block_ids)
        |
    read_file("config.json"):
        |
    FileIndex.get("config.json") -> FileLocation
        |
    For each block_id in FileLocation.block_ids:
        BlockIndex.get_by_id(block_id) -> BlockLocation
            |
        Seek to archive_offset in .kmc file
            |
        Read compressed_size bytes
            |
        Verify block hash (SHA-256)
            |
        Decompress using codec + codec_metadata
        |
    Concatenate decompressed blocks
        |
    Verify file hash (SHA-256)
        |
    Return bytes
```

### Unpack Operation

```
Read: Magic + Manifest + Block data
                |
        For each file entry:
            For each block:
                Seek to offset -> Read compressed data
                    |
                Verify block hash (SHA-256)
                    |
                Decompress using codec from manifest
                    |
                Write to output file
                    |
            Verify file hash (SHA-256)
```

### Tensor-Aware Pack Operation

```
Source safetensors file
        |
    Read header (JSON)
        |
    Extract tensor metadata
    (name, dtype, shape, offsets, sizes)
        |
    Compute block boundaries
    aligned to tensor boundaries
        |
    For each block:
        Build CodecContext (dtype, shape, tensor_name)
            |
        Select codec via selector.py
        (dtype-based candidate chain)
            |
        Compress + verify roundtrip
            |
        Record codec_metadata in BlockEntry
        |
    Build manifest with
    TensorEntry + codec_metadata
        |
    Write archive
```

## Future Architecture Considerations

### Tensor-Specific Codecs (v0.4 -- Completed)

The codec subpackage introduced in v0.4 provides the first tensor-aware codecs:

- **BytePlane**: Byte-plane separation for BF16/FP16/FP32 data.
- **FloatPlane**: Sign/exponent/mantissa bit-level separation for FP16/BF16/FP32.
- **Automatic selector**: dtype-based candidate chains with roundtrip verification.

Future codec improvements may include:
- **Tensor-specific dictionaries**: Build zstd dictionaries from tensors of the same shape and dtype across the model.
- **XOR delta encoding**: Encode differences between adjacent rows/columns of weight matrices.
- **Quantization-aware compression**: Exploit the structure of GGUF quantization levels.

### GGUF-Aware Compression (v0.5 -- Experimental)

The `--gguf-aware` flag enables experimental GGUF-specific compression:

- **Tensor metadata parsing**: Full tensor info extraction from GGUF v2/v3 files.
- **Quantization-aware codec selection**: Quantized tensors (Q4_K, Q5_0, etc.) skip float-aware transforms and use zstd/zlib directly.
- **Format metadata in manifest**: GGUF version, endianness, tensor count, and quantization summary are recorded in `format_metadata["gguf"]`.
- **Artifact type classification**: GGUF files are classified as `artifact_type: "gguf_model"`.

Future improvements:
- **Block-level GGUF compression**: Skip already-quantized blocks entirely (store raw), compress only metadata and vocabulary sections.
- **GGUF tensor-aligned blocks**: Align block boundaries to GGUF tensor boundaries for partial loading.

### Block-Level Loading (v0.7 -- Implemented)

Block-level loading is now implemented via the `KMCReader` class and the `index/` subpackage. The manifest v6 stores per-block `archive_offset` values that enable direct seeking to any block without scanning preceding data. The `KMCReader` API supports:

- Reading specific files without decompressing the entire archive (`read_file()`).
- Reading specific tensors by name without decompressing the entire archive (`read_tensor()`).
- Extracting individual files or tensors to disk (`extract_file()`, `extract_tensor()`).
- Listing archive contents without decompression (`list_files()`, `list_tensors()`).

For older archives without `archive_offset` values, the block index automatically reconstructs offsets from the archive layout. This provides partial access support for all KMC archives, not just those created with v0.7.

> **Important:** Block-level loading does NOT reduce inference VRAM. The decompressed blocks still occupy the same memory. Runtime compressed loading (keeping blocks compressed in memory) is future research.

### Runtime Integration (Future)

The current partial access API provides raw data access but does not integrate with ML frameworks. Future work includes:
- Integration with Hugging Face `from_pretrained()` loading.
- Integration with llama.cpp GGUF loading.
- Block server for remote block fetching (HTTP/gRPC).
- Runtime compressed loading (research phase — keeping blocks compressed in memory).

### Parallel Processing

The block-oriented design naturally supports parallel compression and decompression. Future versions could use Python's `concurrent.futures` or `multiprocessing` to process blocks in parallel, significantly improving throughput on multi-core systems.

### Format-Aware Compression

With the inspector and tensor_inspector modules, KMC can detect model formats and potentially apply format-specific optimizations:
- Aligning block boundaries with tensor boundaries in safetensors.
- Skipping already-compressed quantized data in GGUF.
- Applying delta compression for LoRA adapters relative to their base models.
