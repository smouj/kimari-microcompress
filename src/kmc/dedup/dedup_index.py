"""Deduplication index: tracks which blocks are duplicates.

The DedupIndex maps block fingerprints to their canonical location,
allowing duplicate blocks to reference the original instead of storing
redundant data. Only exact SHA-256 matches are considered duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .block_fingerprint import fingerprint_block_data


@dataclass
class DedupEntry:
    """A single deduplication mapping.

    Attributes:
        canonical_block_id: The global block ID of the unique block.
        canonical_file_index: File index containing the unique block.
        canonical_block_index: Block index within the canonical file.
        duplicate_block_ids: List of global block IDs that reference
            this canonical block.
        fingerprint: SHA-256 fingerprint of the original data.
        original_size: Size of the original block data.
    """

    canonical_block_id: int
    canonical_file_index: int
    canonical_block_index: int
    duplicate_block_ids: list[int] = field(default_factory=list)
    fingerprint: str = ""
    original_size: int = 0

    @property
    def total_references(self) -> int:
        """Total number of blocks referencing this entry (including canonical)."""
        return 1 + len(self.duplicate_block_ids)


class DedupIndex:
    """Index mapping block fingerprints to their canonical locations.

    When multiple blocks have identical original data, only the first
    occurrence (canonical) is stored. Subsequent occurrences reference
    the canonical block via dedup_ref.
    """

    def __init__(self) -> None:
        self._entries: dict[str, DedupEntry] = {}  # sha256 -> DedupEntry
        self._block_to_entry: dict[int, str] = {}  # global_block_id -> sha256
        self._total_blocks: int = 0
        self._unique_blocks: int = 0
        self._deduplicated_blocks: int = 0
        self._saved_bytes: int = 0

    def add_block(
        self,
        global_block_id: int,
        file_index: int,
        block_index: int,
        original_data: bytes,
    ) -> bool:
        """Add a block to the dedup index.

        If the block's data matches an existing entry, it's marked as
        a duplicate. Otherwise, it becomes a new canonical entry.

        Args:
            global_block_id: Global block ID across all files.
            file_index: File index in the manifest.
            block_index: Block index within the file.
            original_data: Original (uncompressed) block data.

        Returns:
            True if this block is a duplicate, False if it's unique.
        """
        fp = fingerprint_block_data(original_data)
        self._total_blocks += 1
        self._block_to_entry[global_block_id] = fp

        if fp in self._entries:
            # Duplicate found
            entry = self._entries[fp]
            entry.duplicate_block_ids.append(global_block_id)
            self._deduplicated_blocks += 1
            self._saved_bytes += len(original_data)
            return True

        # New unique block
        self._entries[fp] = DedupEntry(
            canonical_block_id=global_block_id,
            canonical_file_index=file_index,
            canonical_block_index=block_index,
            fingerprint=fp,
            original_size=len(original_data),
        )
        self._unique_blocks += 1
        return False

    def is_duplicate(self, global_block_id: int) -> bool:
        """Check if a block is a duplicate."""
        fp = self._block_to_entry.get(global_block_id)
        if fp is None:
            return False
        entry = self._entries.get(fp)
        if entry is None:
            return False
        return global_block_id != entry.canonical_block_id

    def get_canonical(self, global_block_id: int) -> int | None:
        """Get the canonical block ID for a (possibly duplicate) block.

        Returns the same block_id if it's the canonical block, or the
        canonical block's ID if it's a duplicate. Returns None if not found.
        """
        fp = self._block_to_entry.get(global_block_id)
        if fp is None:
            return None
        entry = self._entries.get(fp)
        if entry is None:
            return None
        return entry.canonical_block_id

    def get_entry(self, global_block_id: int) -> DedupEntry | None:
        """Get the DedupEntry for a block."""
        fp = self._block_to_entry.get(global_block_id)
        if fp is None:
            return None
        return self._entries.get(fp)

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def unique_blocks(self) -> int:
        return self._unique_blocks

    @property
    def deduplicated_blocks(self) -> int:
        return self._deduplicated_blocks

    @property
    def saved_bytes(self) -> int:
        return self._saved_bytes

    @property
    def entries(self) -> list[DedupEntry]:
        return list(self._entries.values())

    def to_manifest_dict(self) -> dict:
        """Serialize dedup stats for manifest.deduplication field."""
        return {
            "enabled": True,
            "fingerprint": "sha256",
            "unique_blocks": self._unique_blocks,
            "deduplicated_blocks": self._deduplicated_blocks,
            "saved_bytes": self._saved_bytes,
        }
