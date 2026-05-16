"""Tensor-aware inspector: extract metadata from safetensors and similar formats."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TensorInfo:
    """Metadata about a single tensor in a model file."""

    name: str
    dtype: str
    shape: list[int]
    byte_offset: int
    byte_size: int


@dataclass
class SafetensorsMeta:
    """Parsed metadata from a safetensors file."""

    header_size: int
    tensors: list[TensorInfo]
    total_params: int
    total_bytes: int


def parse_safetensors_header(path: Path) -> SafetensorsMeta:
    """Parse the header of a safetensors file.

    safetensors format:
        - First 8 bytes: header length (little-endian uint64)
        - Next header_length bytes: JSON header
        - JSON header maps tensor names to {dtype, shape, data_offsets}
        - Special "__metadata__" key for user metadata

    Args:
        path: Path to the safetensors file.

    Returns:
        SafetensorsMeta with parsed tensor information.
    """
    path = Path(path)

    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("File too small for safetensors header")
        header_len = struct.unpack("<Q", header_len_bytes)[0]

        header_data = f.read(header_len)
        if len(header_data) < header_len:
            raise ValueError("Truncated safetensors header")

    header = json.loads(header_data.decode("utf-8"))

    tensors: list[TensorInfo] = []
    total_params = 0
    total_bytes = 0

    for name, info in header.items():
        if name == "__metadata__":
            continue

        dtype = info.get("dtype", "unknown")
        shape = info.get("shape", [])
        data_offsets = info.get("data_offsets", [0, 0])

        if len(data_offsets) >= 2:
            byte_offset = data_offsets[0]
            byte_size = data_offsets[1] - data_offsets[0]
        else:
            byte_offset = 0
            byte_size = 0

        # Estimate parameter count from shape
        param_count = 1
        for dim in shape:
            param_count *= dim

        total_params += param_count
        total_bytes += byte_size

        tensors.append(TensorInfo(
            name=name,
            dtype=dtype,
            shape=shape,
            byte_offset=byte_offset,
            byte_size=byte_size,
        ))

    return SafetensorsMeta(
        header_size=8 + header_len,
        tensors=tensors,
        total_params=total_params,
        total_bytes=total_bytes,
    )


def get_tensor_summary(path: Path) -> dict:
    """Get a summary of tensors in a safetensors file.

    Returns a dict with:
        - num_tensors: number of tensors
        - total_params: total parameter count
        - total_bytes: total tensor data size
        - dtypes: set of unique dtypes
        - largest_tensor: name of the largest tensor
    """
    meta = parse_safetensors_header(path)

    dtypes = set(t.dtype for t in meta.tensors)
    largest = max(meta.tensors, key=lambda t: t.byte_size) if meta.tensors else None

    return {
        "num_tensors": len(meta.tensors),
        "total_params": meta.total_params,
        "total_bytes": meta.total_bytes,
        "dtypes": sorted(dtypes),
        "largest_tensor": largest.name if largest else None,
    }
