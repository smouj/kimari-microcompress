"""Block, file, and tensor indexes for partial access to .kmc archives.

This module provides structured indexes that map between archive blocks,
files, and tensors, enabling selective extraction without decompressing
the entire archive. Indexes can be built from an existing manifest or
reconstructed by scanning the archive when physical offsets are missing.

Index construction is backward-compatible with manifests from v0.2 through
v0.6. When a manifest lacks physical block offsets (common in older
archives), the index builder scans the .kmc file to reconstruct them.
"""

from __future__ import annotations

from .block_index import BlockIndex, BlockLocation
from .file_index import FileIndex, FileLocation
from .tensor_index import TensorIndex, TensorLocation

__all__ = [
    "BlockIndex",
    "BlockLocation",
    "FileIndex",
    "FileLocation",
    "TensorIndex",
    "TensorLocation",
]
