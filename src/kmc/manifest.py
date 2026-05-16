"""KMC manifest: JSON metadata describing files, blocks, codecs and hashes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


KMC_MANIFEST_VERSION = 1


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
class FileEntry:
    """A single file within the archive."""

    path: str  # Relative path inside the archive (POSIX)
    original_size: int
    hash: str  # SHA-256 of the original uncompressed file
    block_size: int  # Block size used (bytes)
    blocks: list[BlockEntry] = field(default_factory=list)


@dataclass
class KMCManifest:
    """Top-level manifest for a .kmc archive."""

    version: int = KMC_MANIFEST_VERSION
    tool: str = "kimari-microcompress"
    tool_version: str = "0.1.0"
    created_at: str = ""
    total_original_size: int = 0
    total_compressed_size: int = 0
    files: list[FileEntry] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> KMCManifest:
        """Deserialize manifest from JSON string."""
        data = json.loads(raw)
        files = []
        for f in data.get("files", []):
            blocks = [BlockEntry(**b) for b in f.get("blocks", [])]
            files.append(
                FileEntry(**{k: v for k, v in f.items() if k != "blocks"}, blocks=blocks)
            )
        return cls(
            version=data.get("version", KMC_MANIFEST_VERSION),
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
