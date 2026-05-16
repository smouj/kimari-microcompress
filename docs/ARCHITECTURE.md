# Architecture

## Design Principles

Kimari MicroCompress is built on a set of core principles that guide every design decision:

1. **Lossless by default**: Every compression operation must be perfectly reversible. There is no "lossy mode" — if a user needs quantization, that's a separate concern handled by tools like GGUF's quantization formats, not by KMC.

2. **Byte-exact verification**: SHA-256 hashes are computed at both the file level and the block level, ensuring that every byte of the original input can be verified after decompression. This dual-level hashing catches both large-scale corruption and subtle bit-flips.

3. **Codec flexibility**: The system supports multiple compression codecs (zstd, zlib, raw) and selects the best one per-block. This means that blocks that don't benefit from compression are stored raw, while highly compressible blocks get the full benefit of zstd or zlib.

4. **Block-oriented design**: Files are split into fixed-size blocks (default 256 KiB) before compression. This enables future features like partial decompression (block-loading), parallel compression/decompression, and fine-grained integrity verification.

5. **Manifest-first metadata**: All metadata is stored in a single JSON manifest at the beginning of the archive. This allows tools to inspect the archive without decompressing any data, and makes the format human-readable and debuggable.

6. **Tensor-aware extension**: When `--tensor-aware` mode is enabled, block boundaries are aligned to tensor boundaries in safetensors files, and tensor metadata is recorded in the manifest. This is a structural preparation for future tensor-specific codecs.

7. **Optional dependencies**: Features that require external packages (safetensors, zipnn) degrade gracefully when those packages are not installed. Core functionality never depends on optional packages.

## Module Structure

### `archive.py` — Core Operations

The `archive` module implements the three fundamental operations on `.kmc` archives:

- **`pack(source, output, tensor_aware=False)`**: Reads files from the source, splits them into blocks (optionally aligned to tensor boundaries), compresses each block, computes hashes, and writes the complete archive including the manifest. When `tensor_aware=True`, safetensors files are inspected for tensor metadata and block boundaries are adjusted to avoid splitting tensors across blocks where reasonable.
- **`unpack(archive, output_dir)`**: Reads the manifest, then for each file entry, reads and decompresses blocks in order, verifies hashes, and writes the reconstructed files. Includes path traversal protection.
- **`verify(archive)` / `verify_full(archive)`**: Reads the manifest and checks every block hash and file hash. `verify()` returns a list of errors; `verify_full()` returns a structured `VerificationReport`.

The archive format uses a simple sequential layout: magic bytes, manifest length, manifest, then block data. Offsets in the manifest point directly to block positions within the file, enabling random access to individual blocks.

### `codecs.py` — Compression Codecs

The codec system is designed around a `Codec` protocol with `compress()` and `decompress()` methods. Each codec returns a `CodecResult` containing the compressed/decompressed data, the codec identifier, and size information.

The `compress_block()` function implements the codec selection logic: it tries the best available codec (zstd if installed, otherwise zlib), and if the compressed output isn't smaller than the input, it falls back to the raw codec. This ensures that already-compressed or random data isn't expanded by the compression attempt.

Codec identifiers are stored as strings in the manifest (`"zstd"`, `"zlib"`, `"raw"`), allowing future codecs to be added without breaking backward compatibility.

### `manifest.py` — Archive Metadata

The manifest uses Python dataclasses with a clear hierarchy: `KMCManifest` contains `FileEntry` objects, each of which contains `BlockEntry` objects and optionally `TensorEntry` objects. The manifest is serialized to JSON for human readability and forward compatibility.

Key design choices:
- POSIX-style paths are used for cross-platform compatibility.
- The manifest version field distinguishes between v1 (original) and v2 (tensor-aware) formats.
- The `tool` and `tool_version` fields enable provenance tracking.
- `TensorEntry` records tensor name, dtype, shape, byte_offset, and byte_size for safetensors files.
- v2 manifests are backward-compatible with v1 readers (tensor fields default to empty/zero).

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
    Compress each block
        ↓
    Build manifest with
    TensorEntry records
        ↓
    Write archive
```

## Future Architecture Considerations

### Tensor-Specific Codecs (v0.4)

With tensor metadata available in the manifest, future versions can implement codecs that exploit the structure of specific data types:

- **BF16/FP16 separation**: Split weights into sign, exponent, and mantissa components, compress each separately. Exponents tend to cluster, mantissas have different entropy patterns.
- **Per-dtype block selection**: Use different codec parameters for BF16 blocks vs FP32 blocks vs INT8 blocks.
- **Tensor-specific dictionaries**: Build zstd dictionaries from tensors of the same shape and dtype across the model.

### Block-Level Loading

The current design stores all blocks sequentially, but the manifest already contains per-block offsets. A future "block server" could serve individual blocks on demand, enabling:
- Loading specific layers of a model without downloading the entire file.
- Streaming decompression for large models.
- Memory-mapped access to specific tensor regions.

### Parallel Processing

The block-oriented design naturally supports parallel compression and decompression. Future versions could use Python's `concurrent.futures` or `multiprocessing` to process blocks in parallel, significantly improving throughput on multi-core systems.

### Format-Aware Compression

With the inspector and tensor_inspector modules, KMC can detect model formats and potentially apply format-specific optimizations:
- Aligning block boundaries with tensor boundaries in safetensors.
- Skipping already-compressed quantized data in GGUF.
- Applying delta compression for LoRA adapters relative to their base models.
