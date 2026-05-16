"""Streaming I/O for large files — block-based reading, writing, and hashing."""

from __future__ import annotations

from .readers import iter_file_blocks, read_block_at
from .streaming import sha256_stream
from .writers import write_blocks, write_blocks_from_iter

__all__ = [
    "iter_file_blocks",
    "read_block_at",
    "sha256_stream",
    "write_blocks",
    "write_blocks_from_iter",
]
