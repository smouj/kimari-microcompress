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
    - v3 (KMC v0.4): Adds per-block codec_metadata for tensor-aware codecs.
        - BlockEntry gains optional codec_metadata, tensor_name, tensor_dtype,
          tensor_shape fields for lossless codec reconstruction.
        - Backward-compatible: v1/v2 manifests read without errors in v3 readers.
    - v4 (KMC v0.5): Adds artifact_type and format-level metadata.
        - KMCManifest gains artifact_type, artifact_metadata, format_metadata.
        - artifact_type: "huggingface_model"|"gguf_model"|"lora_adapter"|
          "training_checkpoint"|"unknown"
        - artifact_metadata: Dict with artifact-specific metadata (e.g., LoRA rank).
        - format_metadata: Dict with format-specific metadata (e.g., GGUF info).
        - Backward-compatible: v1/v2/v3 manifests read without errors in v4 readers.
    - v5 (KMC v0.6): Adds parallelism and streaming metadata.
        - KMCManifest gains parallelism field recording worker count and deterministic
          order guarantee.
        - Backward-compatible: v1/v2/v3/v4 manifests read without errors in v5 readers.
    - v6 (KMC v0.7): Adds index metadata for partial access.
        - KMCManifest gains index field recording availability of block, file, and
          tensor indexes.
        - BlockEntry gains archive_offset field for direct block access.
        - Backward-compatible: v1/v2/v3/v4/v5 manifests read without errors in v6 readers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

KMC_MANIFEST_VERSION = 6  # v0.7-alpha with index metadata for partial access


@dataclass
class BlockEntry:
    """A single compressed block within a file.

    v0.4 additions:
        - codec_metadata: Dict with codec-specific parameters needed for
          decompression (e.g., transform type, element_size, inner_codec).
        - tensor_name: Name of the tensor this block belongs to (if known).
        - tensor_dtype: Dtype of the tensor (e.g., 'BF16', 'FP16', 'FP32').
        - tensor_shape: Shape of the tensor this block belongs to.
    """

    index: int
    offset: int
    compressed_size: int
    original_size: int
    codec: str  # CodecId value
    hash: str  # SHA-256 of compressed data
    # v0.4 fields (optional, for tensor-aware codecs)
    codec_metadata: dict = field(default_factory=dict)
    tensor_name: str = ""
    tensor_dtype: str = ""
    tensor_shape: list[int] = field(default_factory=list)
    # v0.7 field (optional, for partial access)
    archive_offset: int = 0  # Physical byte offset in the .kmc file (0 = not set)


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
        - 2: Tensor-aware format (KMC v0.3)
        - 3: Per-block codec metadata (KMC v0.4)
        - 4: Artifact type and format metadata (KMC v0.5+)
    Version 4 manifests are backward-compatible: readers that only understand
    version 1/2/3 can safely ignore the artifact and format fields.
    """

    version: int = KMC_MANIFEST_VERSION
    tool: str = "kimari-microcompress"
    tool_version: str = "0.7.0-alpha"
    created_at: str = ""
    total_original_size: int = 0
    total_compressed_size: int = 0
    files: list[FileEntry] = field(default_factory=list)
    # v0.5 fields (optional, for artifact-aware workflows)
    artifact_type: str = "unknown"  # huggingface_model|gguf_model|lora_adapter|...
    artifact_metadata: dict = field(default_factory=dict)
    format_metadata: dict = field(default_factory=dict)
    # v0.6 fields (optional, for parallelism tracking)
    parallelism: dict = field(default_factory=dict)  # v0.6: parallelism metadata
    # v0.7 fields (optional, for partial access index)
    index: dict = field(default_factory=dict)  # v0.7: index metadata for partial access

    def to_json(self) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> KMCManifest:
        """Deserialize manifest from JSON string.

        Handles v1, v2, v3, and v4 manifests. Missing fields default
        to empty/zero values for backward compatibility.
        """
        data = json.loads(raw)
        files = []
        for f in data.get("files", []):
            blocks = []
            for b in f.get("blocks", []):
                blocks.append(
                    BlockEntry(
                        index=b.get("index", 0),
                        offset=b.get("offset", 0),
                        compressed_size=b.get("compressed_size", 0),
                        original_size=b.get("original_size", 0),
                        codec=b.get("codec", "raw"),
                        hash=b.get("hash", ""),
                        codec_metadata=b.get("codec_metadata", {}),
                        tensor_name=b.get("tensor_name", ""),
                        tensor_dtype=b.get("tensor_dtype", ""),
                        tensor_shape=b.get("tensor_shape", []),
                        archive_offset=b.get("archive_offset", 0),
                    )
                )
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
            artifact_type=data.get("artifact_type", "unknown"),
            artifact_metadata=data.get("artifact_metadata", {}),
            format_metadata=data.get("format_metadata", {}),
            parallelism=data.get("parallelism", {}),
            index=data.get("index", {}),
        )

    def to_bytes(self) -> bytes:
        """Serialize manifest to UTF-8 bytes."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> KMCManifest:
        """Deserialize manifest from UTF-8 bytes."""
        return cls.from_json(raw.decode("utf-8"))
