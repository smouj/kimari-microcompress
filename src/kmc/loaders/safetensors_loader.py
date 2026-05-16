"""Experimental safetensors tensor loader for KMC archives.

This module provides experimental functionality for loading tensor bytes
from .kmc archives that contain safetensors files. It can optionally
convert tensor bytes to PyTorch or NumPy arrays if the respective
libraries are installed.

WARNING: This module is experimental. API may change without notice.
KMC does not perform compressed inference — tensors are decompressed
before being returned.

Optional dependencies:
    - safetensors: For reading safetensors header metadata.
    - torch: For converting bytes to torch.Tensor objects.
    - numpy: For converting bytes to numpy.ndarray objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_tensor_bytes(archive_path: str | Path, tensor_name: str) -> bytes:
    """Load the raw bytes of a tensor from a .kmc archive.

    This is the primary API for tensor-byte loading. It returns the
    decompressed bytes of the requested tensor without requiring any
    optional dependencies.

    Args:
        archive_path: Path to the .kmc archive file.
        tensor_name: Name of the tensor to load.

    Returns:
        Raw bytes of the tensor data.

    Raises:
        FileNotFoundError: If the archive or tensor is not found.
        ValueError: If the tensor cannot be located or decompressed.
    """
    from ..reader import KMCReader

    with KMCReader(archive_path) as reader:
        return reader.read_tensor(tensor_name)


def load_tensor(archive_path: str | Path, tensor_name: str) -> Any:
    """Load a tensor from a .kmc archive as a native tensor object.

    Attempts to return a torch.Tensor if PyTorch is installed, or a
    numpy.ndarray if NumPy is installed. If neither is available,
    raises an error suggesting load_tensor_bytes() or installing extras.

    This function requires:
    1. The .kmc archive to have been created with --tensor-aware mode.
    2. The safetensors library for header parsing.
    3. torch or numpy for tensor construction.

    Args:
        archive_path: Path to the .kmc archive file.
        tensor_name: Name of the tensor to load.

    Returns:
        torch.Tensor or numpy.ndarray depending on available libraries.

    Raises:
        ImportError: If no tensor library is installed.
        FileNotFoundError: If the archive or tensor is not found.
        ValueError: If the tensor cannot be constructed.
    """
    from ..reader import KMCReader

    with KMCReader(archive_path) as reader:
        tensor_info = reader.get_tensor_info(tensor_name)
        if tensor_info is None:
            raise FileNotFoundError(f"Tensor not found in archive: {tensor_name!r}")

        if tensor_info.dtype is None:
            raise ValueError(
                f"Tensor {tensor_name!r} has no dtype metadata. "
                "The archive may not have been created with --tensor-aware mode."
            )

        raw_bytes = reader.read_tensor(tensor_name)

    # Try PyTorch first
    try:
        import torch  # noqa: F401

        return _bytes_to_torch_tensor(raw_bytes, tensor_info.dtype, tensor_info.shape)
    except ImportError:
        pass

    # Try NumPy
    try:
        import numpy as np  # noqa: F401

        return _bytes_to_numpy_array(raw_bytes, tensor_info.dtype, tensor_info.shape)
    except ImportError:
        pass

    raise ImportError(
        "No tensor library available. Install PyTorch (`pip install torch`) "
        "or NumPy (`pip install numpy`), or use load_tensor_bytes() to get "
        "raw bytes without conversion."
    )


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

_SAFETENSORS_TO_TORCH_DTYPE = {
    "BOOL": "torch.bool",
    "U8": "torch.uint8",
    "I8": "torch.int8",
    "I16": "torch.int16",
    "I32": "torch.int32",
    "I64": "torch.int64",
    "F16": "torch.float16",
    "BF16": "torch.bfloat16",
    "F32": "torch.float32",
    "F64": "torch.float64",
}

_SAFETENSORS_TO_NUMPY_DTYPE = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": None,  # NumPy doesn't natively support BF16
    "F32": "float32",
    "F64": "float64",
}

_SAFETENSORS_DTYPE_SIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


def _bytes_to_torch_tensor(
    data: bytes,
    dtype_str: str,
    shape: list[int] | None,
) -> Any:
    """Convert raw bytes to a PyTorch tensor.

    Args:
        data: Raw tensor bytes.
        dtype_str: Safetensors dtype string (e.g., 'BF16', 'F32').
        shape: Tensor shape as a list of integers.

    Returns:
        torch.Tensor instance.
    """
    import torch

    torch_dtype_name = _SAFETENSORS_TO_TORCH_DTYPE.get(dtype_str)
    if torch_dtype_name is None:
        raise ValueError(f"Unsupported safetensors dtype for torch: {dtype_str!r}")

    # Map string to actual torch dtype
    dtype_map = {
        "torch.bool": torch.bool,
        "torch.uint8": torch.uint8,
        "torch.int8": torch.int8,
        "torch.int16": torch.int16,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }

    torch_dtype = dtype_map.get(torch_dtype_name)
    if torch_dtype is None:
        raise ValueError(f"Cannot map dtype {torch_dtype_name!r} to torch")

    tensor = torch.frombuffer(bytearray(data), dtype=torch_dtype)
    if shape:
        tensor = tensor.reshape(shape)

    return tensor


def _bytes_to_numpy_array(
    data: bytes,
    dtype_str: str,
    shape: list[int] | None,
) -> Any:
    """Convert raw bytes to a NumPy array.

    Args:
        data: Raw tensor bytes.
        dtype_str: Safetensors dtype string (e.g., 'BF16', 'F32').
        shape: Tensor shape as a list of integers.

    Returns:
        numpy.ndarray instance.
    """
    import numpy as np

    np_dtype_str = _SAFETENSORS_TO_NUMPY_DTYPE.get(dtype_str)
    if np_dtype_str is None:
        raise ValueError(
            f"Unsupported safetensors dtype for numpy: {dtype_str!r}. "
            "BF16 is not natively supported by NumPy. Use PyTorch instead."
        )

    arr = np.frombuffer(data, dtype=np.dtype(np_dtype_str))
    if shape:
        arr = arr.reshape(shape)

    return arr
