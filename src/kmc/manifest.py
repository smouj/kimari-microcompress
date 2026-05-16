"""KMC manifest: JSON metadata describing files, blocks, codecs, hashes and tensors.

The manifest is the central metadata structure for a .kmc archive. It contains
file entries with block information, integrity hashes, and optional tensor-level
metadata for format-aware compression.

Format version history:
    - v1 (KMC v0.1-v0.2): Basic file/block/codec/hash manifest.
    - v2 (KMC v0.3): Adds optional tensor-aware entries for safetensors files.
        - TensorEntry records tensor names, dtypes, shapes, byte ranges.
        - FileEntry gains optional tensor_count, dtype_summary, tensor_entries.
        - Backward-compatible: v1 manifests read without errors in v2 readers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

KMC_MANIFEST_VERSION = 2  # v0.3-alpha with tensor-aware support


@dataclass
class BlockEntry:
    """A single compressed block within a file."""

    index: int
    offset: int
    compressed_size: int
    original_size: int
    codec: str  # CodecId value
    hash: str  # SHA-256 of compressed data


@dataclass
class TensorEntry:
    """A single tensor within a safetensors file (tensor-aware manifest extension).

    This entry records the tensor's position and type so that block boundaries
    can be aligned to tensor boundaries when using --tensor-aware mode.
    """

    name: str
    dtype: str
    shape: list[int]
    byte_offset: int  # Offset within the file (after safetensors header)
    byte_size: int  # Size of the tensor data in bytes


@dataclass
class FileEntry:
    """A single file within the archive.

    Tensor-aware fields (tensor_count, dtype_summary, tensor_entries) are
    optional and only populated when --tensor-aware mode is used and the
    file is a safetensors file with readable metadata.
    """

    path: str  # Relative path inside the archive (POSIX)
    original_size: int
    hash: str  # SHA-256 of the original uncompressed file
    block_size: int  # Block size used (bytes)
    blocks: list[BlockEntry] = field(default_factory=list)
    # Tensor-aware fields (optional, v0.3+)
    tensor_count: int = 0
    dtype_summary: list[str] = field(default_factory=list)
    tensor_entries: list[TensorEntry] = field(default_factory=list)


@dataclass
class KMCManifest:
    """Top-level manifest for a .kmc archive.

    The format_version field distinguishes between manifest versions:
        - 1: Original format (KMC v0.1-v0.2)
        - 2: Tensor-aware format (KMC v0.3+)
    Version 2 manifests are backward-compatible: readers that only understand
    version 1 can safely ignore the tensor-aware fields.
    """

    version: int = KMC_MANIFEST_VERSION
    tool: str = "kimari-microcompress"
    tool_version: str = "0.3.0-alpha"
    created_at: str = ""
    total_original_size: int = 0
    total_compressed_size: int = 0
    files: list[FileEntry] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> KMCManifest:
        """Deserialize manifest from JSON string.

        Handles both v1 (no tensor fields) and v2 (tensor-aware) manifests.
        Missing fields default to empty/zero values.
        """
        data = json.loads(raw)
        files = []
        for f in data.get("files", []):
            blocks = [BlockEntry(**b) for b in f.get("blocks", [])]
            tensor_entries = [TensorEntry(**t) for t in f.get("tensor_entries", [])]
            files.append(
                FileEntry(
                    path=f.get("path", ""),
                    original_size=f.get("original_size", 0),
                    hash=f.get("hash", ""),
                    block_size=f.get("block_size", 262144),
                    blocks=blocks,
                    tensor_count=f.get("tensor_count", 0),
                    dtype_summary=f.get("dtype_summary", []),
                    tensor_entries=tensor_entries,
                )
            )
        return cls(
            version=data.get("version", 1),  # Default to v1 for old manifests
            tool=data.get("tool", "kimari-microcompress"),
            tool_version=data.get("tool_version", "0.1.0"),
            created_at=data.get("created_at", ""),
            total_original_size=data.get("total_original_size", 0),
            total_compressed_size=data.get("total_compressed_size", 0),
            files=files,
        )

    def to_bytes(self) -> bytes:
        """Serialize manifest to UTF-8 bytes."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> KMCManifest:
        """Deserialize manifest from UTF-8 bytes."""
        return cls.from_json(raw.decode("utf-8"))
