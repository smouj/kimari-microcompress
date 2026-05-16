"""Hashing utilities for integrity verification (SHA-256 per file, per block)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, block_size: int = 256 * 1024) -> str:
    """Compute SHA-256 hex digest of a file, reading in blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_block(data: bytes) -> str:
    """Compute SHA-256 hex digest of a single block (alias for clarity)."""
    return sha256_bytes(data)
