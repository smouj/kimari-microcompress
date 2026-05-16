"""GGUF format parser: minimal safe header reading.

Reads GGUF file metadata (magic, version, endianness, tensor count,
metadata KV count, file size) without loading the full file into memory.
Does NOT attempt to parse all tensor metadata or KV pairs yet.

GGUF format reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GGUF_MAGIC_LE = 0x46475547  # "GGUF" read as little-endian uint32
GGUF_MAGIC_BE = 0x47554647  # "GGUF" read as big-endian uint32 (indicates big-endian file)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GGUFInfo:
    """Parsed GGUF file header information.

    Attributes:
        available: Whether GGUF parsing is available (always True for basic parsing).
        magic: The raw magic bytes as a string (e.g., "GGUF").
        version: GGUF format version (1, 2, or 3).
        endianness: Either "little" or "big", determined from magic byte order.
        tensor_count: Number of tensors in the file (from header).
        metadata_kv_count: Number of metadata key-value pairs (from header).
        file_size: Total file size in bytes.
        header_size: Size of the header portion that was read (24 bytes for v2/v3).
        tensor_metadata_implemented: Always False for now (planned for future).
    """

    available: bool = True
    magic: str = "GGUF"
    version: int = 0
    endianness: str = "little"
    tensor_count: int = 0
    metadata_kv_count: int = 0
    file_size: int = 0
    header_size: int = 0
    tensor_metadata_implemented: bool = False


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------


def read_gguf_info(path: Path) -> GGUFInfo:
    """Read minimal GGUF header information safely.

    Opens the file in binary mode, reads only the header bytes needed,
    and returns a GGUFInfo structure. Does NOT load the entire file
    into memory. Does NOT parse tensor metadata or KV pairs.

    Args:
        path: Path to the GGUF file.

    Returns:
        GGUFInfo with parsed header information.

    Raises:
        ValueError: If the file is not a valid GGUF file.
        OSError: If the file cannot be read.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    file_size = path.stat().st_size

    # GGUF header layout (minimum 24 bytes for v2/v3):
    #   4 bytes: magic ("GGUF")
    #   4 bytes: version (uint32)
    #   8 bytes: tensor_count (uint64)
    #   8 bytes: metadata_kv_count (uint64)
    MIN_HEADER_SIZE = 24

    with open(path, "rb") as f:
        # Read magic
        magic_bytes = f.read(4)
        if len(magic_bytes) < 4:
            raise ValueError("File too small for GGUF header")

        # Determine endianness from magic
        magic_le = struct.unpack("<I", magic_bytes)[0]
        magic_be = struct.unpack(">I", magic_bytes)[0]

        if magic_le == GGUF_MAGIC_LE:
            endianness = "little"
            fmt_prefix = "<"
        elif magic_be == GGUF_MAGIC_LE:
            # If we read "GGUF" as big-endian, the file is big-endian
            endianness = "big"
            fmt_prefix = ">"
        else:
            raise ValueError(
                f"Invalid GGUF magic: 0x{magic_le:08X} (expected 0x{GGUF_MAGIC_LE:08X})"
            )

        # Read version
        version_bytes = f.read(4)
        if len(version_bytes) < 4:
            raise ValueError("Truncated GGUF version")
        version = struct.unpack(f"{fmt_prefix}I", version_bytes)[0]

        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        # Read tensor_count and metadata_kv_count
        tensor_count = 0
        metadata_kv_count = 0

        if version >= 2:
            tc_bytes = f.read(8)
            if len(tc_bytes) >= 8:
                tensor_count = struct.unpack(f"{fmt_prefix}Q", tc_bytes)[0]

            kv_bytes = f.read(8)
            if len(kv_bytes) >= 8:
                metadata_kv_count = struct.unpack(f"{fmt_prefix}Q", kv_bytes)[0]
        elif version == 1:
            # v1 uses uint32 for counts
            tc_bytes = f.read(4)
            if len(tc_bytes) >= 4:
                tensor_count = struct.unpack(f"{fmt_prefix}I", tc_bytes)[0]

            kv_bytes = f.read(4)
            if len(kv_bytes) >= 4:
                metadata_kv_count = struct.unpack(f"{fmt_prefix}I", kv_bytes)[0]

    header_size = MIN_HEADER_SIZE if version >= 2 else 16

    return GGUFInfo(
        available=True,
        magic="GGUF",
        version=version,
        endianness=endianness,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
        file_size=file_size,
        header_size=header_size,
        tensor_metadata_implemented=False,
    )


def is_gguf_file(path: Path) -> bool:
    """Check if a file is a valid GGUF file.

    This is a lightweight check that only reads the magic bytes.

    Args:
        path: Path to the file to check.

    Returns:
        True if the file appears to be a valid GGUF file.
    """
    try:
        with open(path, "rb") as f:
            magic_bytes = f.read(4)
            if len(magic_bytes) < 4:
                return False
            magic_le = struct.unpack("<I", magic_bytes)[0]
            magic_be = struct.unpack(">I", magic_bytes)[0]
            return magic_le == GGUF_MAGIC_LE or magic_be == GGUF_MAGIC_LE
    except (OSError, struct.error):
        return False
