"""Block fingerprinting for cross-file deduplication.

Computes SHA-256 fingerprints of original (uncompressed) block data
to identify identical blocks across files within a .kmc archive.
Only exact byte-for-byte matches are considered duplicates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..manifest import KMCManifest


@dataclass
class BlockFingerprint:
    """Fingerprint for a single block's original data.

    Attributes:
        block_file_index: Index of the file in manifest.files.
        block_index: Index of the block within its file.
        original_size: Size of the original (uncompressed) block data.
        sha256: SHA-256 hash of the original block data.
    """

    block_file_index: int
    block_index: int
    original_size: int
    sha256: str


def fingerprint_block_data(data: bytes) -> str:
    """Compute SHA-256 fingerprint of raw block data.

    Args:
        data: Original (uncompressed) block data.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(data).hexdigest()


def compute_block_fingerprint(
    original_data: bytes,
    file_index: int,
    block_index: int,
) -> BlockFingerprint:
    """Compute a fingerprint for a block's original data.

    Args:
        original_data: Original (uncompressed) block data.
        file_index: Index of the file in the manifest.
        block_index: Index of the block within its file.

    Returns:
        BlockFingerprint with the SHA-256 hash and metadata.
    """
    return BlockFingerprint(
        block_file_index=file_index,
        block_index=block_index,
        original_size=len(original_data),
        sha256=fingerprint_block_data(original_data),
    )


def fingerprint_manifest_blocks(manifest: KMCManifest) -> list[BlockFingerprint]:
    """Compute fingerprints for all blocks in a manifest.

    Note: This requires access to the original data. It's used during
    packing when original data is available.

    Args:
        manifest: KMCManifest with file and block entries.

    Returns:
        List of BlockFingerprint for each block.
    """
    fingerprints: list[BlockFingerprint] = []
    for file_idx, file_entry in enumerate(manifest.files):
        for block in file_entry.blocks:
            # We can't compute original data hash from manifest alone
            # The actual fingerprinting happens during pack when we have data
            fingerprints.append(
                BlockFingerprint(
                    block_file_index=file_idx,
                    block_index=block.index,
                    original_size=block.original_size,
                    sha256="",  # Must be filled with actual data hash
                )
            )
    return fingerprints
