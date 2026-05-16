# Experimental Loaders

## Overview

KMC v0.7.0-alpha includes experimental tensor loading functionality that can convert raw tensor bytes from `.kmc` archives into native tensor objects (PyTorch tensors or NumPy arrays). This module is located at `src/kmc/loaders/safetensors_loader.py` and is the recommended way to obtain usable tensor objects from KMC archives.

**This module is experimental.** The API may change without notice between versions. It is not covered by the stability guarantees that apply to the core KMC pack/unpack/verify operations.

**Critical warnings:**

- **KMC does NOT perform compressed inference.** Tensors are fully decompressed in memory before being returned. Loading a tensor from a KMC archive does not reduce VRAM usage during inference.
- **Partial tensor loading returns bytes by default.** The `read_tensor` method on `KMCReader` returns raw `bytes`. To obtain native tensor objects, you must use the loader functions documented here.
- **Tensor extraction depends on tensor metadata.** Archives must be created with `--tensor-aware` mode for the loader to have access to dtype and shape information. Without this metadata, conversion to native tensors is not possible.
- **Optional dependencies are required for native tensor objects.** The `load_tensor()` function requires either PyTorch or NumPy to be installed. If neither is available, it raises an `ImportError` with instructions.

## API Reference

### load_tensor_bytes()

```python
from kmc.loaders.safetensors_loader import load_tensor_bytes

def load_tensor_bytes(archive_path: str | Path, tensor_name: str) -> bytes
```

Loads the raw bytes of a tensor from a `.kmc` archive. This is the primary API for tensor-byte loading when you do not need native tensor objects. It returns the decompressed bytes of the requested tensor without requiring any optional dependencies (no PyTorch or NumPy needed).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `archive_path` | `str \| Path` | Path to the `.kmc` archive file |
| `tensor_name` | `str` | Name of the tensor to load |

**Returns:** Raw bytes of the tensor data.

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | The archive or tensor is not found |
| `ValueError` | The tensor cannot be located or decompressed |

**Example:**

```python
from kmc.loaders.safetensors_loader import load_tensor_bytes

# Load raw tensor bytes (no optional dependencies needed)
weight_bytes = load_tensor_bytes("model.kmc", "transformer.h.0.attn.c_attn.weight")
print(f"Loaded {len(weight_bytes)} bytes of tensor data")
```

This function opens a `KMCReader` internally, reads the tensor, and closes the reader. If you need to read multiple tensors from the same archive, it is more efficient to use `KMCReader` directly to avoid repeated manifest parsing and index construction.

### load_tensor()

```python
from kmc.loaders.safetensors_loader import load_tensor

def load_tensor(archive_path: str | Path, tensor_name: str) -> Any
```

Loads a tensor from a `.kmc` archive as a native tensor object. Attempts to return a `torch.Tensor` if PyTorch is installed, or a `numpy.ndarray` if NumPy is installed. If neither is available, raises an `ImportError` suggesting `load_tensor_bytes()` or installing the required extras.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `archive_path` | `str \| Path` | Path to the `.kmc` archive file |
| `tensor_name` | `str` | Name of the tensor to load |

**Returns:** `torch.Tensor` or `numpy.ndarray` depending on available libraries.

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | No tensor library (PyTorch or NumPy) is installed |
| `FileNotFoundError` | The archive or tensor is not found |
| `ValueError` | The tensor has no dtype metadata or cannot be constructed |

**Requirements:**

1. The archive must have been created with `--tensor-aware` mode so that dtype and shape metadata is available.
2. Either PyTorch or NumPy must be installed for tensor construction.
3. The safetensors library is recommended for accurate header parsing but not strictly required.

**Example:**

```python
from kmc.loaders.safetensors_loader import load_tensor

# Load as native tensor (requires PyTorch or NumPy)
weight = load_tensor("model.kmc", "transformer.h.0.attn.c_attn.weight")
print(f"Shape: {weight.shape}, Dtype: {weight.dtype}")

# Use with PyTorch
import torch
if isinstance(weight, torch.Tensor):
    weight = weight.cuda()  # Move to GPU
```

## Dtype Mapping

The loader maps safetensors dtype strings to native framework dtypes. The following table shows all supported dtypes and their mappings:

### PyTorch Dtype Mapping

| Safetensors Dtype | PyTorch Dtype | Element Size |
|-------------------|---------------|-------------|
| `BOOL` | `torch.bool` | 1 byte |
| `U8` | `torch.uint8` | 1 byte |
| `I8` | `torch.int8` | 1 byte |
| `I16` | `torch.int16` | 2 bytes |
| `I32` | `torch.int32` | 4 bytes |
| `I64` | `torch.int64` | 8 bytes |
| `F16` | `torch.float16` | 2 bytes |
| `BF16` | `torch.bfloat16` | 2 bytes |
| `F32` | `torch.float32` | 4 bytes |
| `F64` | `torch.float64` | 8 bytes |

### NumPy Dtype Mapping

| Safetensors Dtype | NumPy Dtype | Element Size | Notes |
|-------------------|-------------|-------------|-------|
| `BOOL` | `bool` | 1 byte | |
| `U8` | `uint8` | 1 byte | |
| `I8` | `int8` | 1 byte | |
| `I16` | `int16` | 2 bytes | |
| `I32` | `int32` | 4 bytes | |
| `I64` | `int64` | 8 bytes | |
| `F16` | `float16` | 2 bytes | |
| `BF16` | N/A | 2 bytes | Not natively supported by NumPy |
| `F32` | `float32` | 4 bytes | |
| `F64` | `float64` | 8 bytes | |

**Important:** NumPy does not natively support the BF16 (bfloat16) dtype. If you attempt to load a BF16 tensor with NumPy, the loader raises a `ValueError` suggesting you use PyTorch instead. This is a fundamental limitation of NumPy, not a bug in the KMC loader.

## Internal Functions

The following internal functions are used by the loader but are documented here for completeness. They are not part of the public API and may change without notice.

### _bytes_to_torch_tensor()

```python
def _bytes_to_torch_tensor(data: bytes, dtype_str: str, shape: list[int] | None) -> Any
```

Converts raw bytes to a PyTorch tensor using `torch.frombuffer()`. The bytes are wrapped in a `bytearray` (required by PyTorch's frombuffer for writable tensors), assigned the appropriate dtype, and reshaped to the specified shape if provided.

### _bytes_to_numpy_array()

```python
def _bytes_to_numpy_array(data: bytes, dtype_str: str, shape: list[int] | None) -> Any
```

Converts raw bytes to a NumPy array using `np.frombuffer()`. The bytes are interpreted with the appropriate NumPy dtype and reshaped to the specified shape if provided. Raises `ValueError` for BF16 tensors since NumPy does not support bfloat16.

## Usage Patterns

### Loading Multiple Tensors Efficiently

If you need to load multiple tensors from the same archive, use `KMCReader` directly instead of calling `load_tensor()` repeatedly. Each call to `load_tensor()` opens a new `KMCReader`, parses the manifest, and builds indexes, which is wasteful when loading multiple tensors:

```python
from kmc.reader import KMCReader
from kmc.loaders.safetensors_loader import _bytes_to_torch_tensor

# Efficient: open once, read many
with KMCReader("model.kmc") as reader:
    for tensor_name in reader.list_tensors():
        info = reader.get_tensor_info(tensor_name)
        if info and info.dtype:
            raw = reader.read_tensor(tensor_name)
            tensor = _bytes_to_torch_tensor(raw, info.dtype, info.shape)
            # Process tensor...
```

Note that using internal functions like `_bytes_to_torch_tensor` is not guaranteed to be stable between versions. For production code, implement your own byte-to-tensor conversion using the dtype mapping tables above.

### Loading Tensors for Model Inspection

```python
from kmc.loaders.safetensors_loader import load_tensor, load_tensor_bytes

# Quick inspection: just check the size
bytes_data = load_tensor_bytes("model.kmc", "model.layers.0.mlp.down_proj.weight")
print(f"Weight size: {len(bytes_data)} bytes")

# Detailed inspection: check statistics
import numpy as np
weight = load_tensor("model.kmc", "model.layers.0.mlp.down_proj.weight")
if isinstance(weight, np.ndarray):
    print(f"Mean: {weight.mean()}, Std: {weight.std()}")
    print(f"Min: {weight.min()}, Max: {weight.max()}")
```

### Handling Missing Dependencies Gracefully

```python
from kmc.loaders.safetensors_loader import load_tensor, load_tensor_bytes

try:
    tensor = load_tensor("model.kmc", "transformer.h.0.attn.weight")
except ImportError:
    # Fallback: get raw bytes and handle conversion yourself
    raw = load_tensor_bytes("model.kmc", "transformer.h.0.attn.weight")
    print(f"No tensor library available. Raw bytes: {len(raw)}")
    # You could implement your own conversion here, or install torch/numpy
```

## Limitations

1. **No BF16 support in NumPy.** NumPy does not natively support the bfloat16 dtype. BF16 tensors can only be loaded with PyTorch. If you attempt to load a BF16 tensor with NumPy as the only available backend, you will get a `ValueError`.

2. **No GGUF tensor loading.** The current loader only supports safetensors-format tensor metadata. GGUF tensor types (Q4_K, Q5_0, etc.) are not supported because their quantized representations cannot be directly converted to standard PyTorch or NumPy dtypes.

3. **No memory mapping.** The loader reads entire tensor data into memory before conversion. For very large tensors, this requires enough RAM to hold both the compressed blocks and the decompressed tensor data simultaneously.

4. **No lazy loading.** All requested tensors are fully decompressed and converted immediately. There is no mechanism for lazy evaluation or deferred conversion.

5. **Experimental status.** This module may undergo API changes between versions. The internal functions (`_bytes_to_torch_tensor`, `_bytes_to_numpy_array`) are particularly likely to change. For production use, consider implementing your own conversion logic based on the dtype mapping tables.

6. **Single-threaded conversion.** Tensor byte-to-object conversion is single-threaded. Loading many tensors or very large tensors may be slow. For bulk loading, consider using the full unpack operation instead.

## See Also

- [PARTIAL_ACCESS.md](PARTIAL_ACCESS.md) -- Overview of partial access features
- [KMC_READER_API.md](KMC_READER_API.md) -- Python API for reading archives
- [SELECTIVE_EXTRACTION.md](SELECTIVE_EXTRACTION.md) -- CLI selective extraction guide
