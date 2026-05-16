# KMCReader Python API Reference

## Overview

`KMCReader` is the primary Python API for partial access to `.kmc` archives. It provides read-only access to archive contents without requiring full decompression. You can list files and tensors, read specific files or tensors, extract items to disk, and inspect archive metadata. The reader builds in-memory indexes on open, enabling efficient partial reads that touch only the blocks needed for the requested data.

**Important:** KMC does NOT perform compressed inference. All data is fully decompressed before being returned. The `read_tensor` method returns raw bytes; to obtain native tensor objects, use the experimental safetensors loader.

## Installation

KMCReader is part of the core `kmc` package and requires no additional dependencies beyond what KMC itself needs:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from kmc.reader import KMCReader

# Open an archive (use as context manager for automatic cleanup)
with KMCReader("model.kmc") as reader:
    # List contents
    files = reader.list_files()
    tensors = reader.list_tensors()

    # Read a specific file
    config_data = reader.read_file("config.json")

    # Read a specific tensor (returns raw bytes)
    weight_bytes = reader.read_tensor("model.layers.0.mlp.down_proj.weight")

    # Extract a file to disk
    reader.extract_file("config.json", "./output/")

    # Extract a tensor to disk
    reader.extract_tensor("model.layers.0.mlp.down_proj.weight", "./output/")
```

## Class Reference

### KMCReader

```python
class KMCReader:
    """Read-only partial-access interface for .kmc archives."""
```

#### Constructor

```python
KMCReader(archive_path: str | Path)
```

Opens a `.kmc` archive for reading. Reads the manifest, builds block, file, and tensor indexes, and prepares the reader for partial access operations.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `archive_path` | `str \| Path` | Path to the `.kmc` archive file |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | The archive file does not exist |
| `ValueError` | The file is not a valid `.kmc` archive |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `archive_path` | `Path` | Resolved absolute path to the archive |
| `manifest` | `KMCManifest` | The archive's manifest |
| `block_index` | `BlockIndex` | Index of all blocks with physical locations |
| `file_index` | `FileIndex` | Index of all files with block mappings |
| `tensor_index` | `TensorIndex` | Index of all tensors (may be empty) |

#### Context Manager

`KMCReader` supports the context manager protocol for automatic resource cleanup:

```python
with KMCReader("model.kmc") as reader:
    data = reader.read_file("config.json")
# File handles are automatically closed
```

#### close()

```python
reader.close() -> None
```

Closes any open file handles. Called automatically when used as a context manager. It is safe to call this method multiple times.

---

### Listing APIs

#### list_files()

```python
reader.list_files() -> list[str]
```

Returns a list of all file paths in the archive. Paths use POSIX format (forward slashes) and are relative to the archive root. The order matches the manifest's file entry order.

**Returns:** List of relative file path strings.

**Example:**

```python
with KMCReader("model.kmc") as reader:
    for path in reader.list_files():
        print(path)
    # Output:
    # config.json
    # tokenizer.json
    # model.safetensors
```

#### list_tensors()

```python
reader.list_tensors() -> list[str]
```

Returns a list of all tensor names in the archive. Returns an empty list if the archive was created without `--tensor-aware` mode, since tensor metadata is required to build the tensor index.

**Returns:** List of tensor name strings.

**Example:**

```python
with KMCReader("model.kmc") as reader:
    tensors = reader.list_tensors()
    if not tensors:
        print("No tensor index available (archive not created with --tensor-aware)")
    else:
        for name in tensors:
            print(name)
```

#### get_manifest()

```python
reader.get_manifest() -> KMCManifest
```

Returns the archive's manifest object, which contains complete metadata about the archive including version, file entries, block entries, artifact type, and format metadata. This is the same object stored in the `manifest` attribute.

**Returns:** `KMCManifest` instance.

#### get_file_info()

```python
reader.get_file_info(path: str) -> FileLocation | None
```

Returns metadata for a specific file, or `None` if the file is not in the archive. The `FileLocation` object contains the file's path, original size, SHA-256 hash, and the ordered list of block IDs that compose the file.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Relative file path |

**Returns:** `FileLocation` or `None`.

**Example:**

```python
with KMCReader("model.kmc") as reader:
    info = reader.get_file_info("model.safetensors")
    if info:
        print(f"Size: {info.size} bytes")
        print(f"SHA-256: {info.sha256}")
        print(f"Blocks: {len(info.block_ids)}")
```

#### get_tensor_info()

```python
reader.get_tensor_info(name: str) -> TensorLocation | None
```

Returns metadata for a specific tensor, or `None` if the tensor is not in the archive. The `TensorLocation` object contains the tensor's name, dtype, shape, file path, and the list of block IDs that contain the tensor's data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tensor name |

**Returns:** `TensorLocation` or `None`.

**Example:**

```python
with KMCReader("model.kmc") as reader:
    info = reader.get_tensor_info("transformer.h.0.attn.c_attn.weight")
    if info:
        print(f"Dtype: {info.dtype}")
        print(f"Shape: {info.shape}")
        print(f"File: {info.file_path}")
    else:
        print("Tensor not found or archive lacks tensor metadata")
```

---

### Reading APIs

#### read_file()

```python
reader.read_file(path: str) -> bytes
```

Reads and returns the full contents of a file from the archive. Only the blocks needed for this file are read and decompressed. The file's SHA-256 hash is verified after reconstruction, ensuring byte-exact integrity.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Relative path of the file to read |

**Returns:** The file's uncompressed contents as `bytes`.

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | The file is not in the archive |
| `ValueError` | The reconstructed file hash does not match the manifest |

**Example:**

```python
with KMCReader("model.kmc") as reader:
    config = reader.read_file("config.json")
    print(config.decode("utf-8"))
```

#### read_file_range()

```python
reader.read_file_range(path: str, offset: int, length: int) -> bytes
```

Reads a byte range from a file in the archive. This is useful when you only need a portion of a file, such as reading a safetensors header without loading the entire file. The current implementation reads the full file and slices the result; future versions may optimize this to read only the overlapping blocks.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Relative path of the file |
| `offset` | `int` | Byte offset to start reading from (must be non-negative) |
| `length` | `int` | Number of bytes to read (must be non-negative) |

**Returns:** The requested byte range as `bytes`. Returns empty bytes if offset is beyond the file's end.

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | The file is not in the archive |
| `ValueError` | Offset or length is negative |

**Example:**

```python
with KMCReader("model.kmc") as reader:
    # Read the first 100 bytes of a safetensors file (header length prefix)
    header_prefix = reader.read_file_range("model.safetensors", 0, 100)
```

#### read_tensor()

```python
reader.read_tensor(name: str) -> bytes
```

Reads and returns the raw bytes of a tensor from the archive. Only the blocks that belong to the requested tensor are read and decompressed. Block checksums are verified during the read.

**Important:** This returns raw bytes, not a PyTorch or NumPy tensor. To convert to a native tensor object, use the experimental safetensors loader (`kmc.loaders.safetensors_loader.load_tensor()`).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of the tensor |

**Returns:** The tensor's raw bytes.

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | The tensor is not in the archive |
| `ValueError` | Block checksums do not match or tensor data cannot be located |

**Example:**

```python
with KMCReader("model.kmc") as reader:
    weight_bytes = reader.read_tensor("transformer.h.0.attn.c_attn.weight")
    print(f"Read {len(weight_bytes)} bytes")

    # Convert to a PyTorch tensor using the experimental loader
    from kmc.loaders.safetensors_loader import load_tensor
    # Note: load_tensor() opens its own reader; for efficiency in a
    # long-running application, use the reader directly with manual conversion.
```

---

### Extraction APIs

#### extract_file()

```python
reader.extract_file(path: str, output_dir: str | Path) -> Path
```

Extracts a single file from the archive to disk. The file is written to the specified output directory using its relative path from the archive. Path traversal protection is applied to ensure the output path remains within the output directory.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Relative path of the file to extract |
| `output_dir` | `str \| Path` | Directory to extract the file into |

**Returns:** `Path` to the extracted file on disk.

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | The file is not in the archive |
| `ExtractionError` | The resolved path would escape the output directory |

**Example:**

```python
with KMCReader("model.kmc") as reader:
    out_path = reader.extract_file("config.json", "./extracted/")
    print(f"Extracted to: {out_path}")
```

#### extract_tensor()

```python
reader.extract_tensor(name: str, output_dir: str | Path) -> Path
```

Extracts a tensor's raw bytes to disk. The tensor data is written to a file named after the tensor (with `.bin` extension) in the output directory. Special characters in tensor names (slashes, backslashes, colons) are replaced with underscores to create safe filenames.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of the tensor |
| `output_dir` | `str \| Path` | Directory to extract the tensor into |

**Returns:** `Path` to the extracted tensor file.

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | The tensor is not in the archive |

**Example:**

```python
with KMCReader("model.kmc") as reader:
    out_path = reader.extract_tensor(
        "transformer.h.0.attn.c_attn.weight",
        "./tensors/"
    )
    print(f"Extracted to: {out_path}")
    # Output: ./tensors/transformer.h.0.attn.c_attn.weight.bin
```

---

## Data Classes

### FileLocation

```python
@dataclass
class FileLocation:
    path: str           # Relative file path (POSIX format)
    size: int           # Original size in bytes
    sha256: str         # SHA-256 hex digest of the original file
    block_ids: list[int] # Ordered list of block IDs composing this file
```

### TensorLocation

```python
@dataclass
class TensorLocation:
    name: str                   # Tensor name
    file_path: str              # Relative path of the containing file
    dtype: str | None           # Data type (e.g., 'BF16', 'FP32')
    shape: list[int] | None     # Shape as list of integers
    block_ids: list[int]        # Ordered list of block IDs
```

### BlockLocation

```python
@dataclass
class BlockLocation:
    block_id: int               # Unique block identifier
    file_path: str              # File this block belongs to
    tensor_name: str | None     # Tensor this block belongs to (if any)
    archive_offset: int         # Physical byte offset in the .kmc file
    compressed_size: int        # Compressed block size in bytes
    original_size: int          # Original block size in bytes
    codec: str                  # Codec name
    codec_metadata: dict        # Codec-specific decompression parameters
    block_hash: str             # SHA-256 of the compressed block data
```

## Error Handling

All `KMCReader` methods raise specific exceptions that can be caught individually:

```python
from kmc.reader import KMCReader

try:
    with KMCReader("model.kmc") as reader:
        data = reader.read_file("nonexistent.json")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Integrity check failed: {e}")
```

The `ValueError` exceptions from `read_file` and `read_tensor` indicate data integrity problems: either a block checksum mismatch during decompression or a file-level hash mismatch after reconstruction. These errors suggest archive corruption and should be taken seriously.

## Thread Safety

`KMCReader` is not thread-safe for concurrent reads. Each read operation opens the archive file, seeks to the required offset, reads data, and closes the file handle. For concurrent access from multiple threads, create a separate `KMCReader` instance per thread. Since the index data structures are read-only after construction, creating multiple readers on the same archive is safe and efficient (each reader will parse its own manifest and build its own indexes).

## See Also

- [PARTIAL_ACCESS.md](PARTIAL_ACCESS.md) -- Overview of partial access features and architecture
- [SELECTIVE_EXTRACTION.md](SELECTIVE_EXTRACTION.md) -- CLI guide for selective extraction
- [EXPERIMENTAL_LOADERS.md](EXPERIMENTAL_LOADERS.md) -- Safetensors tensor-byte loader documentation
