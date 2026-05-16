"""Deduplication planner: analyzes blocks and creates dedup plans.

The DedupPlanner processes blocks during packing, identifies duplicates,
and creates a DedupPlan that records which blocks reference which canonical
blocks. The plan is used during archive writing to skip writing duplicate
block data and instead record dedup_ref in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..manifest import KMCManifest
from .dedup_index import DedupIndex


@dataclass
class DedupRef:
    """A reference from a duplicate block to its canonical block.

    Attributes:
        duplicate_block_id: Global block ID of the duplicate.
        canonical_block_id: Global block ID of the unique (canonical) block.
    """

    duplicate_block_id: int
    canonical_block_id: int


@dataclass
class DedupPlan:
    """Result of deduplication planning.

    Attributes:
        enabled: Whether deduplication is active.
        dedup_refs: Mapping from duplicate block_id to canonical block_id.
        unique_block_ids: Set of block IDs that should be written to the archive.
        total_blocks: Total number of blocks analyzed.
        unique_blocks: Number of unique blocks.
        deduplicated_blocks: Number of blocks that are duplicates.
        saved_bytes: Estimated bytes saved (original size of deduplicated blocks).
    """

    enabled: bool = True
    dedup_refs: dict[int, int] = field(default_factory=dict)
    unique_block_ids: set[int] = field(default_factory=set)
    total_blocks: int = 0
    unique_blocks: int = 0
    deduplicated_blocks: int = 0
    saved_bytes: int = 0

    def is_duplicate(self, block_id: int) -> bool:
        """Check if a block ID is a duplicate."""
        return block_id in self.dedup_refs

    def get_canonical(self, block_id: int) -> int | None:
        """Get canonical block ID for a duplicate."""
        return self.dedup_refs.get(block_id)

    def should_write_block(self, block_id: int) -> bool:
        """Whether this block should be written to the archive.

        Canonical blocks are written; duplicate blocks are skipped
        (their data is referenced from the canonical block).
        """
        return block_id in self.unique_block_ids


class DedupPlanner:
    """Plans deduplication for a set of blocks.

    Usage::

        planner = DedupPlanner()
        planner.add_block(0, original_data_0)
        planner.add_block(1, original_data_1)
        plan = planner.create_plan()
    """

    def __init__(self) -> None:
        self._index = DedupIndex()
        self._block_data: dict[int, bytes] = {}  # block_id -> original_data

    def add_block(self, global_block_id: int, original_data: bytes) -> bool:
        """Add a block for dedup analysis.

        Args:
            global_block_id: Global block ID.
            original_data: Original (uncompressed) block data.

        Returns:
            True if this block is a duplicate of a previous one.
        """
        # We use file_index=0 and block_index=global_block_id as placeholder
        # The actual file/block indices don't matter for dedup detection
        self._block_data[global_block_id] = original_data
        return self._index.add_block(global_block_id, 0, global_block_id, original_data)

    def create_plan(self) -> DedupPlan:
        """Create a dedup plan from the analyzed blocks.

        Returns:
            DedupPlan with dedup_refs, unique_block_ids, and statistics.
        """
        dedup_refs: dict[int, int] = {}
        unique_block_ids: set[int] = set()

        for entry in self._index.entries:
            unique_block_ids.add(entry.canonical_block_id)
            for dup_id in entry.duplicate_block_ids:
                dedup_refs[dup_id] = entry.canonical_block_id

        return DedupPlan(
            enabled=True,
            dedup_refs=dedup_refs,
            unique_block_ids=unique_block_ids,
            total_blocks=self._index.total_blocks,
            unique_blocks=self._index.unique_blocks,
            deduplicated_blocks=self._index.deduplicated_blocks,
            saved_bytes=self._index.saved_bytes,
        )

    @classmethod
    def from_manifest_and_data(
        cls,
        manifest: KMCManifest,
        block_data_map: dict[int, bytes],
    ) -> tuple[DedupPlan, DedupIndex]:
        """Create a dedup plan from a manifest and block data.

        Args:
            manifest: KMCManifest with file and block entries.
            block_data_map: Mapping from global block ID to original data.

        Returns:
            Tuple of (DedupPlan, DedupIndex).
        """
        planner = cls()
        global_block_id = 0
        for file_entry in manifest.files:
            for _block in file_entry.blocks:
                data = block_data_map.get(global_block_id)
                if data is not None:
                    planner.add_block(global_block_id, data)
                global_block_id += 1
        return planner.create_plan(), planner._index
