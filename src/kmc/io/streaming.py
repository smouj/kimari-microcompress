"""Streaming hash computation for large files without loading them fully into memory."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_stream(path: Path, block_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file using streaming reads.

    This avoids loading the entire file into memory by reading in chunks.

    Args:
        path: Path to the file to hash.
        block_size: Size of each read chunk in bytes (default 1 MiB).

    Returns:
        Hex digest of the SHA-256 hash.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
