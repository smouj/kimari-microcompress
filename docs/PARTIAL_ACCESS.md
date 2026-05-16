# Partial Access

## Overview

KMC v0.7.0-alpha introduces partial access features that allow reading specific files and tensors from a `.kmc` archive without decompressing the entire contents. This is a significant capability improvement over previous versions, which required full unpacking to access any data within an archive. Partial access enables workflows like inspecting a single config file, extracting one tensor for inspection, or loading specific model components on demand.

**Important warnings:**

- **KMC does NOT perform compressed inference.** Partial access decompresses the requested data before returning it. Tensors are fully decompressed in memory.
- **Partial tensor loading returns bytes.** The `read_tensor` method returns raw bytes. To convert these bytes into native tensor objects (PyTorch or NumPy), use the experimental safetensors loader documented in [EXPERIMENTAL_LOADERS.md](EXPERIMENTAL_LOADERS.md).
- **Tensor extraction depends on metadata captured during packing.** Archives created without `--tensor-aware` mode will not have tensor-level indexes and can only support file-level partial access.
- **Older archives may support file-level partial access but not tensor-level access.** The index module can reconstruct block offsets from older manifests, but tensor indexes require tensor metadata that was only recorded with `--tensor-aware` mode.

## How Partial Access Works

### Archive Layout

A `.kmc` archive stores all metadata in a JSON manifest at the beginning of the file, followed by concatenated compressed blocks. The manifest contains:

1. **File entries** with paths, sizes, hashes, and lists of blocks.
2. **Block entries** with offsets, codec information, and optional tensor associations.
3. **Tensor entries** (when packed with `--tensor-aware`) that map tensor names to byte ranges.

Starting with manifest v6 (KMC v0.7), each block entry includes an `archive_offset` field that records the physical byte offset of the compressed block data within the archive file. This enables direct seeking to any block without scanning preceding data.

### Index Architecture

When a `KMCReader` opens an archive, it builds three indexes in memory:

1. **BlockIndex** -- Maps each compressed block to its physical location in the archive, including archive offset, compressed size, codec, codec metadata, and associated file/tensor names. The block index enables efficient random access to individual blocks.

2. **FileIndex** -- Maps file paths to their metadata (size, SHA-256 hash) and the ordered list of block IDs that compose the file. When you request a specific file, the file index identifies the relevant blocks and the block index provides their physical locations.

3. **TensorIndex** -- Maps tensor names to their metadata (dtype, shape, file path) and the ordered list of block IDs that contain the tensor's data. This index is only populated for archives created with `--tensor-aware` mode. For archives without tensor metadata, the tensor index will be empty and tensor-level operations will fail gracefully.

### Offset Reconstruction

For archives created with older KMC versions (before v0.7), block entries do not include the `archive_offset` field. In this case, the `BlockIndex.from_manifest()` method automatically reconstructs offsets by computing the cumulative position of each block starting from the end of the manifest. This reconstruction is transparent to the caller and enables partial access even on archives that were not created with v0.7.

The reconstruction algorithm works as follows: it reads the manifest to determine the total header size (magic bytes + manifest length field + manifest bytes), then iterates through all files and blocks in manifest order, computing each block's start offset as the sum of all preceding block compressed sizes plus the header size.

### Data Flow for Partial Reads

When you call `reader.read_file("config.json")`, the following sequence occurs:

1. The file index is consulted to find the `FileLocation` for `config.json`, which includes the list of block IDs.
2. For each block ID, the block index provides the `BlockLocation` with the physical archive offset.
3. The reader seeks directly to each block's offset in the archive file, reads the compressed data, verifies the block checksum (SHA-256), and decompresses using the recorded codec and codec metadata.
4. The decompressed blocks are concatenated and the file's SHA-256 hash is verified against the manifest value.
5. The verified file bytes are returned.

This process reads only the blocks needed for the requested file, skipping all other data in the archive. For a small config file in a multi-gigabyte model archive, this can avoid decompressing gigabytes of unrelated data.

## Index Module

The index module (`src/kmc/index/`) provides three index classes that enable efficient partial access:

### BlockIndex

`BlockIndex` maps block IDs to `BlockLocation` objects, which contain the physical byte offset, compressed size, original size, codec, codec metadata, and block checksum. The index supports lookup by block ID, by file path (returns all blocks for a file), and by tensor name (returns all blocks for a tensor).

```python
from kmc.index import BlockIndex

# Built automatically by KMCReader
index = reader.block_index

# Look up a specific block
block = index.get_by_id(0)

# Get all blocks for a file
blocks = index.get_blocks_for_file("model.safetensors")

# Get all blocks for a tensor
blocks = index.get_blocks_for_tensor("transformer.h.0.attn.weight")
```

### FileIndex

`FileIndex` maps file paths to `FileLocation` objects, which contain the file's original size, SHA-256 hash, and the ordered list of block IDs that compose the file. The index also supports pattern matching using fnmatch-style glob patterns.

```python
from kmc.index import FileIndex

# Built automatically by KMCReader
index = reader.file_index

# Look up a file
loc = index.get("config.json")

# Match files by pattern
json_files = index.match_pattern("*.json")
```

### TensorIndex

`TensorIndex` maps tensor names to `TensorLocation` objects, which contain the tensor's dtype, shape, file path, and the list of block IDs. This index is only populated for archives created with `--tensor-aware` mode.

```python
from kmc.index import TensorIndex

# Built automatically by KMCReader
index = reader.tensor_index

# Check if tensor-level access is available
if index.available:
    tensors = index.list_tensors()
    loc = index.get("transformer.h.0.attn.weight")
```

## Manifest v6 Index Metadata

Manifest v6 adds an `index` field at the top level that records the availability of partial access features:

```json
{
  "version": 6,
  "index": {
    "version": 1,
    "has_block_offsets": true,
    "has_file_index": true,
    "has_tensor_index": false
  }
}
```

The `has_block_offsets` field indicates whether blocks have physical archive offsets stored directly in the manifest. When `true`, no offset reconstruction is needed. The `has_file_index` and `has_tensor_index` fields indicate the presence of file-level and tensor-level index data respectively.

## Performance Considerations

Partial access is most beneficial when you need a small subset of data from a large archive. The performance characteristics are:

- **Archive open time** is dominated by manifest parsing and index construction. For very large manifests, this can take a noticeable amount of time, but it is still much faster than full decompression.
- **Single file read** requires reading and decompressing only the blocks that compose that file. For small files (config, tokenizer), this is nearly instant even in multi-GB archives.
- **Tensor read** requires reading and decompressing only the blocks that contain the tensor. For tensor-aware archives, blocks are often aligned to tensor boundaries, making this efficient.
- **Block checksum verification** adds a small overhead to each read but guarantees data integrity.

Use `kmc bench --partial-access` to measure partial access performance on your specific archives and hardware. The benchmark reports archive open time, single-file read time, and tensor read time.

## Limitations

1. **No streaming partial reads.** The current implementation reads entire blocks into memory before decompression. For very large individual blocks, this requires enough memory to hold both the compressed and decompressed data.

2. **File-level reads require all blocks.** Even for partial file reads via `read_file_range`, the current implementation reads all blocks composing the file and then slices the result. Future versions may optimize this by reading only the overlapping blocks.

3. **Tensor access requires --tensor-aware packing.** Archives created without `--tensor-aware` mode do not have the tensor metadata needed for tensor-level partial access. You can still use file-level partial access on these archives.

4. **No concurrent reads.** The `KMCReader` class is not thread-safe for concurrent reads. Each read operation opens the archive file, seeks to the block offset, reads, and closes. For concurrent access, create separate `KMCReader` instances per thread.

5. **Block decompression is not lazy.** All blocks for a requested file or tensor are decompressed immediately. There is no caching of decompressed blocks between calls.

## See Also

- [KMC_READER_API.md](KMC_READER_API.md) -- Complete Python API reference for KMCReader
- [SELECTIVE_EXTRACTION.md](SELECTIVE_EXTRACTION.md) -- CLI guide for selective extraction
- [EXPERIMENTAL_LOADERS.md](EXPERIMENTAL_LOADERS.md) -- Experimental safetensors tensor loader
- [FORMAT_SPEC.md](FORMAT_SPEC.md) -- Manifest v6 format specification
