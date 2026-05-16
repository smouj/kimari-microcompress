"""Similarity metrics for delta compression planning.

Provides functions to compare blocks and files to determine whether
delta compression would be beneficial. Similarity is based on SHA-256
exact match at the block level and size proximity at the file level.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..manifest import FileEntry


@dataclass
class SimilarityResult:
    """Result of a similarity comparison.

    Attributes:
        identical: Whether the compared items are byte-for-byte identical.
        similarity_ratio: 1.0 for identical, 0.0 for completely different.
        matching_blocks: Number of blocks that match exactly.
        total_blocks: Total number of blocks compared.
        size_difference: Absolute size difference in bytes.
    """

    identical: bool = False
    similarity_ratio: float = 0.0
    matching_blocks: int = 0
    total_blocks: int = 0
    size_difference: int = 0


def block_similarity(block_data_a: bytes, block_data_b: bytes) -> float:
    """Compute similarity between two blocks.

    Returns 1.0 for identical blocks, 0.0 for completely different.
    Uses SHA-256 hash comparison for exact match detection.

    Args:
        block_data_a: First block's original data.
        block_data_b: Second block's original data.

    Returns:
        1.0 if identical, 0.0 otherwise (no partial similarity in v0.8).
    """
    if len(block_data_a) != len(block_data_b):
        return 0.0
    hash_a = hashlib.sha256(block_data_a).hexdigest()
    hash_b = hashlib.sha256(block_data_b).hexdigest()
    return 1.0 if hash_a == hash_b else 0.0


def file_similarity(
    file_a: FileEntry,
    file_b: FileEntry,
    block_data_a: dict[int, bytes] | None = None,
    block_data_b: dict[int, bytes] | None = None,
) -> SimilarityResult:
    """Compute similarity between two file entries.

    Compares files based on size and (if block data is available)
    block-level SHA-256 matching.

    Args:
        file_a: First file entry.
        file_b: Second file entry.
        block_data_a: Optional mapping of block index to original data for file_a.
        block_data_b: Optional mapping of block index to original data for file_b.

    Returns:
        SimilarityResult with match statistics.
    """
    size_diff = abs(file_a.original_size - file_b.original_size)

    # Quick check: if sizes differ significantly, not similar
    if file_a.original_size > 0 and file_b.original_size > 0:
        size_ratio = min(file_a.original_size, file_b.original_size) / max(
            file_a.original_size, file_b.original_size
        )
        if size_ratio < 0.5:
            return SimilarityResult(
                identical=False,
                similarity_ratio=0.0,
                matching_blocks=0,
                total_blocks=max(len(file_a.blocks), len(file_b.blocks)),
                size_difference=size_diff,
            )

    # Without block data, can only compare sizes
    if block_data_a is None or block_data_b is None:
        ratio = 1.0 - (size_diff / max(file_a.original_size, file_b.original_size, 1))
        return SimilarityResult(
            identical=file_a.original_size == file_b.original_size and file_a.hash == file_b.hash,
            similarity_ratio=ratio,
            matching_blocks=0,
            total_blocks=0,
            size_difference=size_diff,
        )

    # With block data, compare block-by-block
    total = max(len(file_a.blocks), len(file_b.blocks))
    matching = 0

    for i in range(min(len(file_a.blocks), len(file_b.blocks))):
        data_a = block_data_a.get(i)
        data_b = block_data_b.get(i)
        if data_a is not None and data_b is not None:
            if block_similarity(data_a, data_b) == 1.0:
                matching += 1

    ratio = matching / total if total > 0 else 0.0
    identical = (
        matching == total
        and len(file_a.blocks) == len(file_b.blocks)
        and file_a.hash == file_b.hash
    )

    return SimilarityResult(
        identical=identical,
        similarity_ratio=ratio,
        matching_blocks=matching,
        total_blocks=total,
        size_difference=size_diff,
    )
