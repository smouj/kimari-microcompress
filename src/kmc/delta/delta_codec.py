"""Delta codec: block-level delta encoding between archives.

The DeltaCodec handles the simple block-level delta strategy:
- Blocks identical to the base are referenced (not stored).
- Changed blocks are stored normally.
- No complex binary diff (xdelta/rsync) in v0.8 — that's future work.

This is intentionally simple: compare blocks by SHA-256 hash and
either reference or store them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeltaBlock:
    """A single block in a delta archive.

    Attributes:
        block_id: Global block ID.
        is_changed: Whether this block differs from the base.
        base_block_id: Block ID in the base archive (if unchanged).
        base_file_path: File path in the base archive.
        base_block_index: Block index in the base archive's file.
    """

    block_id: int
    is_changed: bool = True
    base_block_id: int | None = None
    base_file_path: str = ""
    base_block_index: int | None = None


class DeltaCodec:
    """Simple block-level delta codec.

    Compares blocks between a new archive and a base archive by
    SHA-256 hash. Identical blocks are referenced; changed blocks
    are stored normally.

    This codec does NOT implement xdelta-style binary diffing.
    It only references or stores entire blocks.
    """

    def __init__(self) -> None:
        self._delta_blocks: list[DeltaBlock] = []

    def compare_block(
        self,
        block_id: int,
        block_hash: str,
        base_hashes: dict[int, str],
        base_file_path: str = "",
        base_block_index: int | None = None,
    ) -> DeltaBlock:
        """Compare a block against base archive blocks.

        Args:
            block_id: Global block ID of the new block.
            block_hash: SHA-256 hash of the new block's original data.
            base_hashes: Mapping of base block_id -> SHA-256 hash.
            base_file_path: File path in the base archive for reference.
            base_block_index: Block index in the base file.

        Returns:
            DeltaBlock indicating whether this is a changed or unchanged block.
        """
        # Check if any base block has the same hash
        for base_id, base_hash in base_hashes.items():
            if block_hash == base_hash:
                delta_block = DeltaBlock(
                    block_id=block_id,
                    is_changed=False,
                    base_block_id=base_id,
                    base_file_path=base_file_path,
                    base_block_index=base_block_index,
                )
                self._delta_blocks.append(delta_block)
                return delta_block

        # Block is changed
        delta_block = DeltaBlock(block_id=block_id, is_changed=True)
        self._delta_blocks.append(delta_block)
        return delta_block

    @property
    def delta_blocks(self) -> list[DeltaBlock]:
        return list(self._delta_blocks)

    @property
    def changed_blocks(self) -> list[DeltaBlock]:
        return [db for db in self._delta_blocks if db.is_changed]

    @property
    def referenced_blocks(self) -> list[DeltaBlock]:
        return [db for db in self._delta_blocks if not db.is_changed]

    @property
    def total_blocks(self) -> int:
        return len(self._delta_blocks)

    @property
    def changed_count(self) -> int:
        return len(self.changed_blocks)

    @property
    def referenced_count(self) -> int:
        return len(self.referenced_blocks)
