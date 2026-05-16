"""Block-based file readers for streaming compression."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def iter_file_blocks(path: Path, block_size: int = 256 * 1024) -> Iterator[bytes]:
    """Iterate over blocks of a file without loading it entirely into memory.

    Each iteration yields one block of up to ``block_size`` bytes.
    The last block may be smaller.

    Args:
        path: Path to the file to read.
        block_size: Maximum size of each block in bytes (default 256 KiB).

    Yields:
        Bytes objects of up to ``block_size`` bytes each.
    """
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            yield chunk


def read_block_at(path: Path, offset: int, size: int) -> bytes:
    """Read a specific block from a file at the given offset.

    Seeks to ``offset`` and reads ``size`` bytes. Returns fewer bytes
    if the file ends before ``size`` bytes are available.

    Args:
        path: Path to the file to read.
        offset: Byte offset to start reading from.
        size: Number of bytes to read.

    Returns:
        The bytes read from the file.
    """
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)
