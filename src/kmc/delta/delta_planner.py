"""Delta planner: creates delta compression plans by comparing archives.

The DeltaPlanner compares blocks of a new archive against a base archive
and creates a DeltaPlan that records which blocks are changed vs. referenced.
This is used during pack with --delta-base.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..archive import read_manifest_from_archive
from ..manifest import KMCManifest
from .delta_codec import DeltaBlock, DeltaCodec


@dataclass
class DeltaPlan:
    """Plan for creating a delta archive.

    Attributes:
        enabled: Whether delta compression is active.
        base_archive_path: Path to the base .kmc archive.
        base_archive_sha256: SHA-256 of the base archive file.
        delta_blocks: List of DeltaBlock entries.
        changed_block_ids: Set of block IDs that must be stored.
        referenced_block_ids: Set of block IDs that reference the base.
        total_blocks: Total number of blocks in the new archive.
        changed_blocks: Number of changed blocks.
        referenced_blocks: Number of blocks referenced from base.
    """

    enabled: bool = True
    base_archive_path: str = ""
    base_archive_sha256: str = ""
    delta_blocks: list[DeltaBlock] = field(default_factory=list)
    changed_block_ids: set[int] = field(default_factory=set)
    referenced_block_ids: set[int] = field(default_factory=set)
    total_blocks: int = 0
    changed_blocks: int = 0
    referenced_blocks: int = 0

    def is_changed(self, block_id: int) -> bool:
        """Check if a block must be stored (changed)."""
        return block_id in self.changed_block_ids

    def is_referenced(self, block_id: int) -> bool:
        """Check if a block references the base."""
        return block_id in self.referenced_block_ids

    def should_store_block(self, block_id: int) -> bool:
        """Whether this block's data should be written to the delta archive.

        Changed blocks are stored; referenced blocks are skipped
        (reconstructed from base during unpack).
        """
        return self.is_changed(block_id)

    def to_manifest_dict(self) -> dict:
        """Serialize delta plan for manifest.delta field."""
        return {
            "enabled": True,
            "base_archive_sha256": self.base_archive_sha256,
            "base_archive_path_hint": self.base_archive_path,
            "mode": "experimental",
            "total_blocks": self.total_blocks,
            "changed_blocks": self.changed_blocks,
            "referenced_blocks": self.referenced_blocks,
        }


class DeltaPlanner:
    """Plans delta compression by comparing new data against a base archive.

    Usage:
        planner = DeltaPlanner(base_archive_path="checkpoint-1000.kmc")
        planner.add_block(0, new_data_0, "model.safetensors", 0)
        planner.add_block(1, new_data_1, "model.safetensors", 1)
        plan = planner.create_plan()
    """

    def __init__(self, base_archive_path: str | Path) -> None:
        """Initialize with the base archive.

        Args:
            base_archive_path: Path to the base .kmc archive.

        Raises:
            FileNotFoundError: If the base archive doesn't exist.
            ValueError: If the base archive is not a valid .kmc file.
        """
        self._base_path = Path(base_archive_path)
        self._codec = DeltaCodec()
        self._base_manifest: KMCManifest | None = None
        self._base_block_hashes: dict[int, str] = {}

        if self._base_path.exists():
            self._base_manifest, _ = read_manifest_from_archive(self._base_path)
            self._build_base_index()

    def _build_base_index(self) -> None:
        """Build an index of base archive block hashes.

        Maps global block IDs to the SHA-256 of their original (uncompressed) data.
        Note: The manifest stores hash of compressed data, not original.
        For delta comparison, we need original data hashes, which are computed
        during pack when original data is available.
        """
        # We store the file-level hashes and block-level metadata.
        # Actual block-level original data hashing happens during pack.
        pass

    def add_block(
        self,
        global_block_id: int,
        original_data: bytes,
        file_path: str = "",
        block_index: int | None = None,
    ) -> DeltaBlock:
        """Add a block and compare against base.

        Args:
            global_block_id: Global block ID.
            original_data: Original (uncompressed) block data.
            file_path: Relative file path this block belongs to.
            block_index: Block index within the file.

        Returns:
            DeltaBlock indicating changed/referenced status.
        """
        if self._base_manifest is not None:
            for file_entry in self._base_manifest.files:
                if file_entry.path == file_path:
                    if block_index is not None and block_index < len(file_entry.blocks):
                        # Compare size first (quick check)
                        base_block = file_entry.blocks[block_index]
                        if base_block.original_size == len(original_data):
                            # Size matches — but we can't compare original data hash
                            # from manifest alone. The caller should provide base data
                            # for accurate comparison.
                            pass

        # For v0.8, we use a simpler approach: the caller provides base data
        # and we do exact matching. If no base data available, treat as changed.
        delta_block = DeltaBlock(
            block_id=global_block_id,
            is_changed=True,  # Default to changed
        )

        self._codec._delta_blocks.append(delta_block)  # noqa: SLF001
        return delta_block

    def add_block_with_base_data(
        self,
        global_block_id: int,
        original_data: bytes,
        base_original_data: bytes | None,
        base_file_path: str = "",
        base_block_index: int | None = None,
    ) -> DeltaBlock:
        """Add a block with explicit base data comparison.

        This is the preferred method when base block data is available.

        Args:
            global_block_id: Global block ID.
            original_data: Original data of the new block.
            base_original_data: Original data of the corresponding base block.
            base_file_path: File path in the base archive.
            base_block_index: Block index in the base archive.

        Returns:
            DeltaBlock indicating changed/referenced status.
        """
        if base_original_data is not None and original_data == base_original_data:
            delta_block = DeltaBlock(
                block_id=global_block_id,
                is_changed=False,
                base_block_id=global_block_id,
                base_file_path=base_file_path,
                base_block_index=base_block_index,
            )
        else:
            delta_block = DeltaBlock(
                block_id=global_block_id,
                is_changed=True,
            )

        self._codec._delta_blocks.append(delta_block)  # noqa: SLF001
        return delta_block

    def create_plan(self) -> DeltaPlan:
        """Create a delta plan from the analyzed blocks.

        Returns:
            DeltaPlan with changed/referenced block IDs and statistics.
        """
        base_sha256 = ""
        if self._base_path.exists():
            base_sha256 = hashlib.sha256(self._base_path.read_bytes()).hexdigest()

        changed_ids: set[int] = set()
        referenced_ids: set[int] = set()

        for db in self._codec.delta_blocks:
            if db.is_changed:
                changed_ids.add(db.block_id)
            else:
                referenced_ids.add(db.block_id)

        return DeltaPlan(
            enabled=True,
            base_archive_path=str(self._base_path),
            base_archive_sha256=base_sha256,
            delta_blocks=self._codec.delta_blocks,
            changed_block_ids=changed_ids,
            referenced_block_ids=referenced_ids,
            total_blocks=self._codec.total_blocks,
            changed_blocks=self._codec.changed_count,
            referenced_blocks=self._codec.referenced_count,
        )
