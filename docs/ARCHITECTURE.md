# Architecture

## Design Principles

Kimari MicroCompress is built on a set of core principles that guide every design decision:

1. **Lossless by default**: Every compression operation must be perfectly reversible. There is no "lossy mode" — if a user needs quantization, that's a separate concern handled by tools like GGUF's quantization formats, not by KMC.

2. **Byte-exact verification**: SHA-256 hashes are computed at both the file level and the block level, ensuring that every byte of the original input can be verified after decompression. This dual-level hashing catches both large-scale corruption and subtle bit-flips.

3. **Codec flexibility**: The system supports multiple compression codecs (zstd, zlib, raw) and selects the best one per-block. This means that blocks that don't benefit from compression are stored raw, while highly compressible blocks get the full benefit of zstd or zlib.

4. **Block-oriented design**: Files are split into fixed-size blocks (default 256 KiB) before compression. This enables future features like partial decompression (block-loading), parallel compression/decompression, and fine-grained integrity verification.

5. **Manifest-first metadata**: All metadata is stored in a single JSON manifest at the beginning of the archive. This allows tools to inspect the archive without decompressing any data, and makes the format human-readable and debuggable.

## Module Structure

### `archive.py` — Core Operations

The `archive` module implements the three fundamental operations on `.kmc` archives:

- **`pack(source, output)`**: Reads files from the source, splits them into blocks, compresses each block, computes hashes, and writes the complete archive including the manifest.
- **`unpack(archive, output_dir)`**: Reads the manifest, then for each file entry, reads and decompresses blocks in order, verifies hashes, and writes the reconstructed files. Includes path traversal protection.
- **`verify(archive)`**: Reads the manifest and checks every block hash without decompressing. This is fast and sufficient to confirm archive integrity.

The archive format uses a simple sequential layout: magic bytes, manifest length, manifest, then block data. Offsets in the manifest point directly to block positions within the file, enabling random access to individual blocks.

### `codecs.py` — Compression Codecs

The codec system is designed around a `Codec` protocol with `compress()` and `decompress()` methods. Each codec returns a `CodecResult` containing the compressed/decompressed data, the codec identifier, and size information.

The `compress_block()` function implements the codec selection logic: it tries the best available codec (zstd if installed, otherwise zlib), and if the compressed output isn't smaller than the input, it falls back to the raw codec. This ensures that already-compressed or random data isn't expanded by the compression attempt.

Codec identifiers are stored as strings in the manifest (`"zstd"`, `"zlib"`, `"raw"`), allowing future codecs to be added without breaking backward compatibility.

### `manifest.py` — Archive Metadata

The manifest uses Python dataclasses with a clear hierarchy: `KMCManifest` contains `FileEntry` objects, each of which contains `BlockEntry` objects. The manifest is serialized to JSON for human readability and forward compatibility.

Key design choices:
- POSIX-style paths are used for cross-platform compatibility.
- The manifest version field allows future format changes to be handled gracefully.
- The `tool` and `tool_version` fields enable provenance tracking.

### `hashing.py` — Integrity Verification

The hashing module provides SHA-256 computation for bytes, files, and blocks. The file-level hash covers the complete original (uncompressed) file, while block-level hashes cover the compressed block data. This dual approach means:
- Block hashes can be verified without decompressing (fast).
- File hashes verify the complete reconstructed data (thorough).

### `inspector.py` — AI Model Format Detection

The inspector module identifies AI model formats by examining file magic bytes and structure:
- **safetensors**: Detected by the 8-byte header length prefix followed by valid JSON.
- **GGUF**: Detected by the `0x46475547` magic number.
- **PyTorch .bin**: Detected by pickle protocol magic bytes.
- **.pt/.ckpt**: Identified by file extension with additional validation.

This detection enables KMC to apply format-aware optimizations in the future.

### `tensor_inspector.py` — safetensors Metadata

This module parses safetensors headers to extract tensor names, dtypes, shapes, and data offsets. This information is useful for:
- Estimating compression potential per-tensor.
- Understanding model structure without loading it.
- Future block-loading features that need to know tensor boundaries.

### `gguf.py` — GGUF Format Support

The GGUF module provides header parsing and format validation. Full integration (reading tensor data, block-level compression) is planned for a future release, as GGUF files are already typically stored in quantized formats where additional compression may have limited benefit.

## Data Flow

### Pack Operation

```
Source files → Read in blocks → Compress each block → Compute block hash
                                    ↓
                            Select best codec (zstd/zlib/raw)
                                    ↓
                            Build manifest with offsets and hashes
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

## Future Architecture Considerations

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
