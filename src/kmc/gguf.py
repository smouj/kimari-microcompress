"""GGUF format support — future integration for llama.cpp models.

This module provides the foundation for reading GGUF file metadata.
Full block-level compression integration is planned for a future release.

GGUF format reference: https://www.mintlify.com/ggml-org/llama.cpp/concepts/gguf-format
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


GGUF_MAGIC = 0x46475547  # "GGUF"


@dataclass
class GGUFHeader:
    """Parsed GGUF file header."""

    version: int
    tensor_count: int
    metadata_kv_count: int


@dataclass
class GGUFTensorInfo:
    """Information about a tensor in a GGUF file."""

    name: str
    n_dimensions: int
    dimensions: list[int]
    type_id: int
    offset: int


def read_gguf_header(path: Path) -> GGUFHeader:
    """Read the header of a GGUF file.

    GGUF header layout:
        - 4 bytes: magic (0x46475547)
        - 4 bytes: version (uint32 LE)
        - 8 bytes: tensor_count (uint64 LE)
        - 8 bytes: metadata_kv_count (uint64 LE)

    Args:
        path: Path to the GGUF file.

    Returns:
        GGUFHeader with parsed information.
    """
    path = Path(path)

    with open(path, "rb") as f:
        magic_bytes = f.read(4)
        if len(magic_bytes) < 4:
            raise ValueError("File too small for GGUF header")
        magic = struct.unpack("<I", magic_bytes)[0]

        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})")

        version_bytes = f.read(4)
        if len(version_bytes) < 4:
            raise ValueError("Truncated GGUF version")
        version = struct.unpack("<I", version_bytes)[0]

        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        tensor_count_bytes = f.read(8)
        if len(tensor_count_bytes) < 8:
            raise ValueError("Truncated tensor count")
        tensor_count = struct.unpack("<Q", tensor_count_bytes)[0]

        metadata_kv_count_bytes = f.read(8)
        if len(metadata_kv_count_bytes) < 8:
            raise ValueError("Truncated metadata KV count")
        metadata_kv_count = struct.unpack("<Q", metadata_kv_count_bytes)[0]

    return GGUFHeader(
        version=version,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
    )


def is_gguf(path: Path) -> bool:
    """Check if a file is a valid GGUF file."""
    try:
        header = read_gguf_header(path)
        return header.version in (1, 2, 3)
    except (ValueError, OSError):
        return False
