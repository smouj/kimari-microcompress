"""Block index: maps each compressed block to its physical location in the archive.

BlockIndex provides fast lookup of block data by block ID, enabling
partial reads without scanning the entire manifest. For archives created
with v0.7+, block offsets are stored directly in the manifest. For older
archives, offsets are reconstructed by scanning the archive file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..archive import KMC_MAGIC_LEN, MANIFEST_LEN_SIZE


@dataclass
class BlockLocation:
    """Physical location and metadata for a single compressed block.

    Attributes:
        block_id: Unique identifier for this block (index within its file).
        file_path: Relative path of the file this block belongs to.
        tensor_name: Name of the tensor this block belongs to (if tensor-aware).
        archive_offset: Byte offset of the compressed block data in the .kmc file.
        compressed_size: Size of the compressed block data in bytes.
        original_size: Size of the original (uncompressed) block data in bytes.
        codec: Codec used to compress this block.
        codec_metadata: Codec-specific parameters for decompression.
        block_hash: SHA-256 hash of the compressed block data.
    """

    block_id: int
    file_path: str
    tensor_name: str | None
    archive_offset: int
    compressed_size: int
    original_size: int
    codec: str
    codec_metadata: dict[str, Any] = field(default_factory=dict)
    block_hash: str = ""


class BlockIndex:
    """Index of all blocks in a .kmc archive, optimized for partial access.

    Supports lookup by block ID, by file path, and by tensor name.
    The index can be built from a manifest (fast) or by scanning the
    archive (for older manifests without physical offsets).
    """

    def __init__(self) -> None:
        self._blocks: list[BlockLocation] = []
        self._by_file: dict[str, list[BlockLocation]] = {}
        self._by_tensor: dict[str, list[BlockLocation]] = {}
        self._by_id: dict[int, BlockLocation] = {}

    def add(self, block: BlockLocation) -> None:
        """Add a block location to the index."""
        self._blocks.append(block)
        self._by_id[block.block_id] = block

        if block.file_path not in self._by_file:
            self._by_file[block.file_path] = []
        self._by_file[block.file_path].append(block)

        if block.tensor_name:
            if block.tensor_name not in self._by_tensor:
                self._by_tensor[block.tensor_name] = []
            self._by_tensor[block.tensor_name].append(block)

    def get_by_id(self, block_id: int) -> BlockLocation | None:
        """Look up a block by its ID."""
        return self._by_id.get(block_id)

    def get_blocks_for_file(self, file_path: str) -> list[BlockLocation]:
        """Get all blocks for a given file path, in order."""
        return sorted(self._by_file.get(file_path, []), key=lambda b: b.block_id)

    def get_blocks_for_tensor(self, tensor_name: str) -> list[BlockLocation]:
        """Get all blocks for a given tensor name, in order."""
        return sorted(self._by_tensor.get(tensor_name, []), key=lambda b: b.block_id)

    @property
    def all_blocks(self) -> list[BlockLocation]:
        """Return all indexed blocks."""
        return list(self._blocks)

    @property
    def total_blocks(self) -> int:
        """Total number of indexed blocks."""
        return len(self._blocks)

    @classmethod
    def from_manifest(cls, manifest: object, archive_path: Path) -> BlockIndex:
        """Build a BlockIndex from a KMCManifest.

        If the manifest contains valid archive offsets for all blocks,
        the index is built directly. Otherwise, offsets are reconstructed
        by scanning the archive file.

        Args:
            manifest: KMCManifest instance.
            archive_path: Path to the .kmc archive file.

        Returns:
            Populated BlockIndex.
        """
        index = cls()
        needs_offset_reconstruction = False

        for file_entry in manifest.files:  # type: ignore[attr-defined]
            for block in file_entry.blocks:  # type: ignore[attr-defined]
                offset = block.offset  # type: ignore[attr-defined]
                if offset <= 0:
                    needs_offset_reconstruction = True
                    break
            if needs_offset_reconstruction:
                break

        if needs_offset_reconstruction:
            offsets = _reconstruct_offsets(manifest, archive_path)
        else:
            offsets = None

        global_block_id = 0
        for file_entry in manifest.files:  # type: ignore[attr-defined]
            for block in file_entry.blocks:  # type: ignore[attr-defined]
                if offsets and block.index in offsets.get(file_entry.path, {}):
                    archive_offset = offsets[file_entry.path][block.index]
                elif block.offset > 0:  # type: ignore[attr-defined]
                    archive_offset = block.offset  # type: ignore[attr-defined]
                else:
                    archive_offset = 0

                tensor_name = block.tensor_name if block.tensor_name else None  # type: ignore[attr-defined]

                loc = BlockLocation(
                    block_id=global_block_id,
                    file_path=file_entry.path,  # type: ignore[attr-defined]
                    tensor_name=tensor_name,
                    archive_offset=archive_offset,
                    compressed_size=block.compressed_size,  # type: ignore[attr-defined]
                    original_size=block.original_size,  # type: ignore[attr-defined]
                    codec=block.codec,  # type: ignore[attr-defined]
                    codec_metadata=block.codec_metadata,  # type: ignore[attr-defined]
                    block_hash=block.hash,  # type: ignore[attr-defined]
                )
                index.add(loc)
                global_block_id += 1

        return index


def _reconstruct_offsets(
    manifest: object,
    archive_path: Path,
) -> dict[str, dict[int, int]]:
    """Reconstruct block offsets by reading the archive header.

    Computes the start of block data from the magic + manifest length +
    manifest bytes, then iterates blocks in order to compute cumulative
    offsets.

    Args:
        manifest: KMCManifest instance.
        archive_path: Path to the .kmc archive file.

    Returns:
        Dict mapping file_path -> {block_index: archive_offset}.
    """
    manifest_bytes = manifest.to_bytes()  # type: ignore[attr-defined]
    data_start = KMC_MAGIC_LEN + MANIFEST_LEN_SIZE + len(manifest_bytes)

    offsets: dict[str, dict[int, int]] = {}
    current_offset = data_start

    for file_entry in manifest.files:  # type: ignore[attr-defined]
        offsets[file_entry.path] = {}  # type: ignore[attr-defined]
        for block in file_entry.blocks:  # type: ignore[attr-defined]
            offsets[file_entry.path][block.index] = current_offset  # type: ignore[attr-defined]
            current_offset += block.compressed_size  # type: ignore[attr-defined]

    return offsets
