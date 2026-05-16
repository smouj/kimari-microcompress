"""Core archive operations: pack, unpack, verify.

KMC file format layout:
    [Magic: 8 bytes "KMC\\x00\\x01\\x00\\x00\\x00"]
    [Manifest length: 8 bytes, big-endian uint64]
    [Manifest: JSON, UTF-8 encoded]
    [Block data: concatenated compressed blocks]

v0.4 additions:
    - Per-block codec metadata stored in manifest
    - --codec flag supports: auto, byteplane, floatplane, zstd, zlib, raw
    - Tensor-aware mode now applies dtype-specific codecs

v0.5 additions:
    - GGUF-aware compression mode (--gguf-aware)
    - Artifact type and metadata in manifest
    - LoRA and checkpoint workflow support
    - Format-specific metadata (safetensors, GGUF) in manifest
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .codecs.base import CodecContext
from .codecs.selector import select_codec
from .codecs.zstd_codec import is_zstd_available
from .hashing import sha256_block, sha256_file
from .manifest import BlockEntry, FileEntry, KMCManifest, TensorEntry

KMC_MAGIC = b"KMC\x00\x01\x00\x00\x00"
KMC_MAGIC_LEN = 8
MANIFEST_LEN_FMT = ">Q"  # 8-byte big-endian unsigned
MANIFEST_LEN_SIZE = struct.calcsize(MANIFEST_LEN_FMT)
KMC_FORMAT_VERSION = 1  # Archive binary format version (manifest has its own version)

DEFAULT_BLOCK_SIZE = 256 * 1024  # 256 KiB
MAX_MANIFEST_SIZE = 100 * 1024 * 1024  # 100 MB safety limit


# ---------------------------------------------------------------------------
# Safe path extraction
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when a path or manifest entry fails safety checks during unpack."""


def safe_join_extract_path(output_dir: Path, relative_path: str) -> Path:
    """Safely join an output directory with a relative path from the manifest.

    Guarantees that the resulting path is strictly within ``output_dir``.
    Raises ``ExtractionError`` for any unsafe path component.
    """
    if "\x00" in relative_path:
        raise ExtractionError(f"Null byte in path: {relative_path!r}")
    if any(ord(c) < 0x20 for c in relative_path):
        raise ExtractionError(f"Control character in path: {relative_path!r}")

    stripped = relative_path.strip()
    if not stripped:
        raise ExtractionError("Empty or whitespace-only path in manifest")

    if relative_path.startswith("/"):
        raise ExtractionError(f"Absolute path not allowed: {relative_path!r}")
    if len(relative_path) >= 2 and relative_path[1] == ":":
        raise ExtractionError(f"Windows-style absolute path not allowed: {relative_path!r}")

    if "//" in relative_path:
        raise ExtractionError(f"Empty path component (consecutive slashes): {relative_path!r}")

    parts = PurePosixPath(relative_path).parts
    if ".." in parts:
        raise ExtractionError(f"Path traversal ('..') not allowed: {relative_path!r}")

    output_dir = output_dir.resolve()
    candidate = (output_dir / relative_path).resolve()

    try:
        common = os.path.commonpath([str(output_dir), str(candidate)])
        if common != str(output_dir):
            raise ExtractionError(
                f"Path escapes output directory: {relative_path!r} "
                f"resolves to {candidate!r} (outside {output_dir!r})"
            )
    except ValueError:
        raise ExtractionError(f"Path escapes output directory: {relative_path!r}")

    return candidate


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


@dataclass
class VerificationReport:
    """Structured verification report for a .kmc archive."""

    archive_path: str
    format_version: int = 0
    tool: str = ""
    tool_version: str = ""
    created_at: str = ""
    total_files: int = 0
    total_blocks: int = 0
    compressed_size: int = 0
    restored_size: int = 0
    integrity: str = "OK"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "KMC Verification Report",
            "",
            f"Archive: {self.archive_path}",
            f"Format: {self.tool} v{self.format_version}",
            f"Files: {self.total_files}",
            f"Blocks: {self.total_blocks}",
            f"Compressed size: {self._fmt_size(self.compressed_size)}",
            f"Restored size: {self._fmt_size(self.restored_size)}",
        ]
        if self.compressed_size > 0 and self.restored_size > 0:
            ratio = self.compressed_size / self.restored_size
            lines.append(f"Compression ratio: {ratio:.2%}")
        lines.append(f"Integrity: {self.integrity}")
        if self.errors:
            for err in self.errors:
                lines.append(f"  Error: {err}")
        if self.warnings:
            for warn in self.warnings:
                lines.append(f"  Warning: {warn}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_size(n: int) -> str:
        if n >= 1024 * 1024 * 1024:
            return f"{n / (1024 * 1024 * 1024):.2f} GB"
        if n >= 1024 * 1024:
            return f"{n / (1024 * 1024):.2f} MB"
        if n >= 1024:
            return f"{n / 1024:.2f} KB"
        return f"{n} bytes"


def validate_manifest(manifest: KMCManifest) -> list[str]:
    """Validate a manifest for structural correctness and security."""
    errors: list[str] = []
    supported_codecs = {"zstd", "zlib", "raw", "byteplane", "floatplane"}
    seen_paths: set[str] = set()

    for file_entry in manifest.files:
        if file_entry.path in seen_paths:
            errors.append(f"Duplicate path in manifest: {file_entry.path!r}")
        seen_paths.add(file_entry.path)

        try:
            safe_join_extract_path(Path("/tmp/kmc_check"), file_entry.path)
        except ExtractionError as e:
            errors.append(f"Unsafe path in manifest: {file_entry.path!r} — {e}")

        if file_entry.original_size < 0:
            errors.append(
                f"Negative original_size for '{file_entry.path}': {file_entry.original_size}"
            )

        if file_entry.original_size == 0 and file_entry.blocks:
            errors.append(
                f"File '{file_entry.path}' has zero original_size but "
                f"{len(file_entry.blocks)} block(s)"
            )

        block_indices: set[int] = set()
        total_block_original = 0
        for block in file_entry.blocks:
            if block.index in block_indices:
                errors.append(f"Duplicate block index {block.index} in '{file_entry.path}'")
            block_indices.add(block.index)

            if block.codec not in supported_codecs:
                errors.append(
                    f"Unsupported codec '{block.codec}' in block "
                    f"{block.index} of '{file_entry.path}'"
                )

            if block.compressed_size < 0:
                errors.append(
                    f"Negative compressed_size in block {block.index} of '{file_entry.path}'"
                )
            if block.original_size < 0:
                errors.append(
                    f"Negative original_size in block {block.index} of '{file_entry.path}'"
                )
            total_block_original += block.original_size

        if file_entry.original_size > 0 and total_block_original != file_entry.original_size:
            errors.append(
                f"Block size mismatch for '{file_entry.path}': "
                f"file original_size={file_entry.original_size}, "
                f"sum of block original_sizes={total_block_original}"
            )

    return errors


# ---------------------------------------------------------------------------
# Tensor-aware helpers
# ---------------------------------------------------------------------------


def _get_safetensors_tensor_entries(path: Path) -> tuple[list[TensorEntry], list[str], int]:
    """Try to read safetensors tensor metadata from a file."""
    if path.suffix.lower() != ".safetensors":
        return [], [], 0

    try:
        from .formats.safetensors import read_safetensors_info

        info = read_safetensors_info(path)
        entries = [
            TensorEntry(
                name=t.name,
                dtype=t.dtype,
                shape=t.shape,
                byte_offset=t.byte_offset,
                byte_size=t.byte_size,
            )
            for t in info.tensors
        ]
        return entries, info.dtypes, info.tensor_count
    except (ValueError, OSError, KeyError):
        return [], [], 0


def _compute_tensor_aware_block_boundaries(
    file_size: int,
    block_size: int,
    tensor_entries: list[TensorEntry],
    header_size: int = 0,
) -> list[int]:
    """Compute block split points that try to align with tensor boundaries."""
    if not tensor_entries:
        boundaries = []
        offset = 0
        while offset < file_size:
            boundaries.append(offset)
            offset += block_size
        return boundaries

    boundaries: list[int] = []
    current_block_start = 0

    for tensor in sorted(tensor_entries, key=lambda t: t.byte_offset):
        tensor_start = tensor.byte_offset

        if tensor_start < current_block_start:
            continue

        gap = tensor_start - current_block_start

        if gap > 0 and gap <= block_size:
            if gap <= block_size * 0.1:
                if boundaries:
                    boundaries.append(tensor_start)
                else:
                    if tensor_start < block_size:
                        boundaries.append(0)
                    else:
                        boundaries.append(0)
                        boundaries.append(tensor_start)
                current_block_start = tensor_start
            else:
                boundaries.append(current_block_start)
                current_block_start += block_size
                if tensor_start > current_block_start:
                    boundaries.append(tensor_start)
                    current_block_start = tensor_start
        elif gap > block_size:
            while current_block_start + block_size <= tensor_start:
                boundaries.append(current_block_start)
                current_block_start += block_size
            if current_block_start < tensor_start:
                boundaries.append(current_block_start)
                current_block_start = tensor_start
                if current_block_start < file_size:
                    boundaries.append(tensor_start)
        else:
            if not boundaries or current_block_start == 0:
                boundaries.append(current_block_start)
            current_block_start = max(current_block_start, tensor_start)

    while current_block_start < file_size:
        boundaries.append(current_block_start)
        current_block_start += block_size

    seen = set()
    result = []
    for b in boundaries:
        if b not in seen and b < file_size:
            seen.add(b)
            result.append(b)
    result.sort()

    return result


def _find_tensor_for_offset(
    offset: int, tensor_entries: list[TensorEntry], header_size: int = 0
) -> TensorEntry | None:
    """Find the tensor entry that contains the given file offset."""
    for t in tensor_entries:
        data_start = header_size + t.byte_offset
        data_end = data_start + t.byte_size
        if data_start <= offset < data_end:
            return t
    return None


# ---------------------------------------------------------------------------
# Compress/decompress blocks with new codec system
# ---------------------------------------------------------------------------


def _compress_block_with_codec(
    data: bytes,
    codec_name: str = "auto",
    context: CodecContext | None = None,
) -> tuple[bytes, str, dict]:
    """Compress a block using the specified codec.

    Returns (compressed_payload, codec_name_used, codec_metadata).
    """
    if codec_name == "auto":
        selection = select_codec(data, context=context, codec_override=None)
        result = selection.result
        return result.payload, result.codec, result.metadata

    # Specific codec requested
    selection = select_codec(data, context=context, codec_override=codec_name)
    result = selection.result
    return result.payload, result.codec, result.metadata


def _decompress_block_with_metadata(
    payload: bytes,
    codec_name: str,
    original_size: int,
    codec_metadata: dict | None = None,
) -> bytes:
    """Decompress a block using the specified codec and metadata.

    Handles both legacy codecs (zstd, zlib, raw) and new codecs
    (byteplane, floatplane) that require metadata for decompression.
    """
    if codec_name in ("zstd", "zlib", "raw"):
        # Use legacy decompression for v0.2/v0.3 compatible blocks
        from .codecs.legacy import CodecId, decompress_block

        try:
            codec_id = CodecId(codec_name)
        except ValueError:
            codec_id = codec_name
        result = decompress_block(payload, codec_id, original_size)
        return result.data

    # New codecs require metadata
    if codec_name == "byteplane":
        from .codecs.byteplane import BytePlaneCodec

        bp = BytePlaneCodec()
        ctx = CodecContext(original_size=original_size)
        if codec_metadata:
            ctx._codec_metadata = codec_metadata  # type: ignore[attr-defined]
        return bp.decompress(payload, context=ctx)

    if codec_name == "floatplane":
        from .codecs.floatplane import FloatPlaneCodec

        fp = FloatPlaneCodec()
        ctx = CodecContext(original_size=original_size)
        if codec_metadata:
            ctx._codec_metadata = codec_metadata  # type: ignore[attr-defined]
        return fp.decompress(payload, context=ctx)

    raise ValueError(f"Unsupported codec for decompression: {codec_name!r}")


# ---------------------------------------------------------------------------
# GGUF-aware helpers
# ---------------------------------------------------------------------------


def _get_gguf_format_metadata(path: Path) -> dict:
    """Read GGUF format metadata for the manifest."""
    try:
        from .formats.gguf import is_gguf_file, read_gguf_info

        if not is_gguf_file(path):
            return {}

        info = read_gguf_info(path, parse_tensors=True)
        metadata: dict = {
            "version": info.version,
            "endianness": info.endianness,
            "tensor_count": info.tensor_count,
            "metadata_kv_count": info.metadata_kv_count,
        }
        if info.quantization_summary:
            metadata["quantization_summary"] = info.quantization_summary
        if info.tensors:
            metadata["tensor_names"] = [t.name for t in info.tensors[:100]]
        if info.warnings:
            metadata["parse_warnings"] = info.warnings
        return metadata
    except (ValueError, OSError):
        return {}


def _is_gguf_quantized(path: Path) -> bool:
    """Check if a GGUF file contains quantized tensors."""
    try:
        from .formats.gguf import is_gguf_file, is_quantized_ggml_type, read_gguf_info

        if not is_gguf_file(path):
            return False

        info = read_gguf_info(path, parse_tensors=True)
        for tensor in info.tensors:
            if is_quantized_ggml_type(tensor.ggml_type):
                return True
        return False
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------


def pack(
    source: Path,
    output: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    level: int = 3,
    tensor_aware: bool = False,
    codec: str = "auto",
    gguf_aware: bool = False,
    artifact_type: str = "unknown",
    artifact_metadata: dict | None = None,
) -> None:
    """Pack a directory or single file into a .kmc archive.

    Args:
        source: Source directory or file to compress.
        output: Output .kmc file path.
        block_size: Block size in bytes (default 256 KiB).
        level: Compression level (codec-dependent).
        tensor_aware: If True, attempt to align blocks to tensor boundaries
            for safetensors files and record tensor metadata in the manifest.
        codec: Codec to use: 'auto', 'byteplane', 'floatplane', 'zstd', 'zlib', 'raw'.
        gguf_aware: If True, enable experimental GGUF-aware compression mode.
        artifact_type: Artifact type for the manifest (e.g., 'lora_adapter').
        artifact_metadata: Artifact-specific metadata for the manifest.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # Validate codec choice
    _validate_codec_choice(codec, tensor_aware)

    manifest = KMCManifest(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        artifact_type=artifact_type,
        artifact_metadata=artifact_metadata or {},
    )

    all_blocks: list[tuple[int, int, BlockEntry, bytes]] = []

    files_to_pack: list[Path] = []
    if source.is_dir():
        for root, _, filenames in os.walk(source):
            for fn in filenames:
                fpath = Path(root) / fn
                files_to_pack.append(fpath)
    else:
        files_to_pack.append(source)

    for fpath in sorted(files_to_pack):
        rel_path = fpath.relative_to(source) if source.is_dir() else fpath.name
        rel_str = PurePosixPath(rel_path).as_posix()

        file_hash = sha256_file(fpath, block_size)
        file_size = fpath.stat().st_size

        # Try to get tensor metadata if tensor_aware mode
        tensor_entries: list[TensorEntry] = []
        dtype_summary: list[str] = []
        tensor_count = 0
        header_size = 0

        if tensor_aware:
            tensor_entries, dtype_summary, tensor_count = _get_safetensors_tensor_entries(fpath)
            if tensor_entries:
                try:
                    from .formats.safetensors import read_safetensors_info

                    info = read_safetensors_info(fpath)
                    header_size = info.header_size
                except (ValueError, OSError):
                    pass

        # GGUF-aware mode: read GGUF metadata and adjust codec strategy
        file_codec = codec
        if gguf_aware and fpath.suffix.lower() == ".gguf":
            # Record GGUF format metadata
            gguf_meta = _get_gguf_format_metadata(fpath)
            if gguf_meta:
                if not manifest.format_metadata:
                    manifest.format_metadata = {}
                manifest.format_metadata["gguf"] = gguf_meta

                # Set artifact type if not already set
                if manifest.artifact_type == "unknown":
                    manifest.artifact_type = "gguf_model"

            # For GGUF files with quantized tensors, avoid floatplane
            # and prefer zstd/zlib/raw (quantized data doesn't benefit from
            # float-aware transforms)
            if _is_gguf_quantized(fpath):
                if codec in ("auto", "floatplane", "byteplane"):
                    file_codec = "zstd" if is_zstd_available() else "zlib"

        file_entry = FileEntry(
            path=rel_str,
            original_size=file_size,
            hash=file_hash,
            block_size=block_size,
            tensor_count=tensor_count,
            dtype_summary=dtype_summary,
            tensor_entries=tensor_entries,
        )

        # Compute block boundaries
        if tensor_aware and tensor_entries:
            block_starts = _compute_tensor_aware_block_boundaries(
                file_size, block_size, tensor_entries, header_size
            )
        else:
            block_starts = list(range(0, file_size, block_size))

        with open(fpath, "rb") as f:
            for block_index, block_start in enumerate(block_starts):
                if block_index + 1 < len(block_starts):
                    chunk_size = block_starts[block_index + 1] - block_start
                else:
                    chunk_size = file_size - block_start

                f.seek(block_start)
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                # Determine tensor context for this block
                tensor_name = ""
                tensor_dtype = ""
                tensor_shape: list[int] = []
                if tensor_aware and tensor_entries:
                    t = _find_tensor_for_offset(block_start, tensor_entries, header_size)
                    if t:
                        tensor_name = t.name
                        tensor_dtype = t.dtype
                        tensor_shape = t.shape

                # Build codec context
                ctx = CodecContext(
                    file_path=rel_str,
                    tensor_name=tensor_name or None,
                    dtype=tensor_dtype or None,
                    shape=tensor_shape or None,
                    original_size=len(chunk),
                    block_index=block_index,
                )

                # Compress with selected codec
                compressed_payload, codec_used, codec_meta = _compress_block_with_codec(
                    chunk, codec_name=file_codec, context=ctx
                )

                block_hash = sha256_block(compressed_payload)

                block_entry = BlockEntry(
                    index=block_index,
                    offset=0,  # placeholder, fixed below
                    compressed_size=len(compressed_payload),
                    original_size=len(chunk),
                    codec=codec_used,
                    hash=block_hash,
                    codec_metadata=codec_meta,
                    tensor_name=tensor_name,
                    tensor_dtype=tensor_dtype,
                    tensor_shape=tensor_shape,
                )
                file_entry.blocks.append(block_entry)
                all_blocks.append(
                    (len(manifest.files), block_index, block_entry, compressed_payload)
                )

                manifest.total_original_size += len(chunk)
                manifest.total_compressed_size += len(compressed_payload)

        manifest.files.append(file_entry)

    # Auto-detect artifact type from directory contents if still unknown
    if manifest.artifact_type == "unknown" and source.is_dir():
        manifest.artifact_type = _detect_artifact_type(source)

    # Auto-detect format metadata for safetensors if not already set
    if source.is_dir() and "safetensors" not in manifest.format_metadata:
        safetensors_meta = _detect_safetensors_format_metadata(source)
        if safetensors_meta:
            manifest.format_metadata["safetensors"] = safetensors_meta

    # Compute correct offsets iteratively
    for _ in range(20):
        manifest_bytes = manifest.to_bytes()
        data_offset = KMC_MAGIC_LEN + MANIFEST_LEN_SIZE + len(manifest_bytes)

        current_offset = data_offset
        for _file_idx, _block_idx, block_entry, _block_data in all_blocks:
            block_entry.offset = current_offset
            current_offset += block_entry.compressed_size

        new_manifest_bytes = manifest.to_bytes()
        if len(new_manifest_bytes) == len(manifest_bytes):
            manifest_bytes = new_manifest_bytes
            break
        manifest_bytes = new_manifest_bytes

    # Write archive
    with open(output, "wb") as out:
        out.write(KMC_MAGIC)
        out.write(struct.pack(MANIFEST_LEN_FMT, len(manifest_bytes)))
        out.write(manifest_bytes)
        for _file_idx, _block_idx, _block_entry, block_data in all_blocks:
            out.write(block_data)


def _detect_artifact_type(source: Path) -> str:
    """Auto-detect the artifact type from directory contents."""
    from .workflows.checkpoint import detect_checkpoint
    from .workflows.lora import detect_lora_adapter

    # Check LoRA first (more specific)
    adapter_info = detect_lora_adapter(source)
    if adapter_info.is_lora:
        return "lora_adapter"

    # Check checkpoint
    ckpt_info = detect_checkpoint(source)
    if ckpt_info.is_checkpoint:
        return "training_checkpoint"

    # Check for GGUF files
    for f in source.rglob("*.gguf"):
        if f.is_file():
            return "gguf_model"

    # Check for safetensors files
    for f in source.rglob("*.safetensors"):
        if f.is_file():
            return "huggingface_model"

    return "unknown"


def _detect_safetensors_format_metadata(source: Path) -> dict:
    """Detect safetensors format metadata from a directory."""
    from .formats.safetensors import detect_safetensors_shards

    shards = detect_safetensors_shards(source)
    if shards:
        return {"is_sharded": True, "shard_count": len(shards)}

    for f in source.rglob("*.safetensors"):
        if f.is_file():
            try:
                from .formats.safetensors import read_safetensors_info

                info = read_safetensors_info(f)
                return {
                    "is_sharded": info.is_shard,
                    "tensor_count": info.tensor_count,
                    "dtypes": info.dtypes,
                }
            except (ValueError, OSError):
                pass

    return {}


def _validate_codec_choice(codec: str, tensor_aware: bool) -> None:
    """Validate that the codec choice is compatible with other options."""
    valid_codecs = {"auto", "raw", "zlib", "zstd", "byteplane", "floatplane"}
    if codec not in valid_codecs:
        raise ValueError(f"Unknown codec: {codec!r}. Valid options: {sorted(valid_codecs)}")

    if codec in ("byteplane", "floatplane") and not tensor_aware:
        # byteplane/floatplane can work without tensor_aware,
        # but they need dtype information for best results
        pass  # Allow it but user should know

    if codec == "floatplane" and not is_zstd_available():
        # floatplane works with zlib inner too, so just warn silently
        pass


# ---------------------------------------------------------------------------
# Read manifest
# ---------------------------------------------------------------------------


def read_manifest_from_archive(archive: Path) -> tuple[KMCManifest, int]:
    """Read the manifest from a .kmc archive."""
    with open(archive, "rb") as f:
        magic = f.read(KMC_MAGIC_LEN)
        if magic != KMC_MAGIC:
            raise ValueError(f"Invalid KMC magic: {magic!r}")

        manifest_len_bytes = f.read(MANIFEST_LEN_SIZE)
        if len(manifest_len_bytes) < MANIFEST_LEN_SIZE:
            raise ValueError("Truncated manifest length")
        manifest_len = struct.unpack(MANIFEST_LEN_FMT, manifest_len_bytes)[0]

        if manifest_len > MAX_MANIFEST_SIZE:
            raise ValueError(
                f"Manifest too large: {manifest_len:,} bytes (max {MAX_MANIFEST_SIZE:,})"
            )

        manifest_raw = f.read(manifest_len)
        if len(manifest_raw) < manifest_len:
            raise ValueError("Truncated manifest data")

        try:
            json.loads(manifest_raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid manifest JSON: {e}") from e

        data_start = KMC_MAGIC_LEN + MANIFEST_LEN_SIZE + manifest_len
        return KMCManifest.from_bytes(manifest_raw), data_start


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def verify(archive: Path) -> list[str]:
    """Verify the integrity of a .kmc archive."""
    report = verify_full(archive)
    return report.errors


def verify_full(archive: Path) -> VerificationReport:
    """Perform a full verification of a .kmc archive and return a report."""
    archive = Path(archive).resolve()
    report = VerificationReport(archive_path=str(archive))

    try:
        manifest, data_start = read_manifest_from_archive(archive)
    except (ValueError, OSError) as e:
        report.integrity = "FAILED"
        report.errors.append(f"Failed to read archive: {e}")
        return report

    report.format_version = manifest.version
    report.tool = manifest.tool
    report.tool_version = manifest.tool_version
    report.created_at = manifest.created_at
    report.total_files = len(manifest.files)
    report.total_blocks = sum(len(f.blocks) for f in manifest.files)
    report.compressed_size = manifest.total_compressed_size
    report.restored_size = manifest.total_original_size

    if manifest.version not in (1, 2, 3, 4):
        report.warnings.append(f"Unknown manifest version: v{manifest.version}")

    structural_errors = validate_manifest(manifest)
    if structural_errors:
        report.integrity = "FAILED"
        report.errors.extend(structural_errors)
        return report

    archive_size = archive.stat().st_size
    with open(archive, "rb") as f:
        for file_entry in manifest.files:
            file_hasher = hashlib.sha256()
            file_reconstructed_size = 0

            for block in sorted(file_entry.blocks, key=lambda b: b.index):
                if block.offset + block.compressed_size > archive_size:
                    report.integrity = "FAILED"
                    report.errors.append(
                        f"Block {block.index} of '{file_entry.path}' exceeds archive size"
                    )
                    continue

                f.seek(block.offset)
                block_data = f.read(block.compressed_size)

                if len(block_data) < block.compressed_size:
                    report.integrity = "FAILED"
                    report.errors.append(
                        f"Block {block.index} of '{file_entry.path}': "
                        f"could not read full block data"
                    )
                    continue

                actual_hash = sha256_block(block_data)
                if actual_hash != block.hash:
                    report.integrity = "FAILED"
                    report.errors.append(
                        f"Block {block.index} of '{file_entry.path}': checksum mismatch"
                    )
                    continue

                try:
                    decompressed = _decompress_block_with_metadata(
                        block_data,
                        block.codec,
                        block.original_size,
                        block.codec_metadata,
                    )
                    file_hasher.update(decompressed)
                    file_reconstructed_size += len(decompressed)
                except Exception as e:
                    report.integrity = "FAILED"
                    report.errors.append(
                        f"Block {block.index} of '{file_entry.path}': decompression failed: {e}"
                    )
                    continue

            if file_reconstructed_size == file_entry.original_size:
                actual_file_hash = file_hasher.hexdigest()
                if actual_file_hash != file_entry.hash:
                    report.integrity = "FAILED"
                    report.errors.append(f"File '{file_entry.path}': hash mismatch")
            elif file_reconstructed_size > 0:
                report.integrity = "FAILED"
                report.errors.append(
                    f"File '{file_entry.path}': reconstructed size "
                    f"{file_reconstructed_size} != "
                    f"original size {file_entry.original_size}"
                )

    return report


# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------


def unpack(archive: Path, output_dir: Path) -> None:
    """Unpack a .kmc archive to a directory."""
    archive = Path(archive).resolve()
    output_dir = Path(output_dir).resolve()

    manifest, data_start = read_manifest_from_archive(archive)

    seen_paths: set[str] = set()
    for file_entry in manifest.files:
        if file_entry.path in seen_paths:
            raise ExtractionError(f"Duplicate path in manifest: {file_entry.path!r}")
        seen_paths.add(file_entry.path)
        safe_join_extract_path(output_dir, file_entry.path)

    structural_errors = validate_manifest(manifest)
    if structural_errors:
        raise ValueError(f"Manifest validation failed: {'; '.join(structural_errors)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(archive, "rb") as f:
        for file_entry in manifest.files:
            out_path = safe_join_extract_path(output_dir, file_entry.path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.is_symlink():
                raise ExtractionError(f"Refusing to overwrite symlink: {out_path}")

            with open(out_path, "wb") as out_f:
                file_hasher = hashlib.sha256()

                for block in sorted(file_entry.blocks, key=lambda b: b.index):
                    f.seek(block.offset)
                    block_data = f.read(block.compressed_size)

                    actual_hash = sha256_block(block_data)
                    if actual_hash != block.hash:
                        raise ValueError(
                            f"Block {block.index} of "
                            f"'{file_entry.path}': "
                            f"checksum mismatch during unpack"
                        )

                    decompressed = _decompress_block_with_metadata(
                        block_data,
                        block.codec,
                        block.original_size,
                        block.codec_metadata,
                    )
                    out_f.write(decompressed)
                    file_hasher.update(decompressed)

                actual_file_hash = file_hasher.hexdigest()
                if actual_file_hash != file_entry.hash:
                    raise ValueError(
                        f"File '{file_entry.path}': hash mismatch "
                        f"(expected={file_entry.hash}, "
                        f"got={actual_file_hash})"
                    )


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------


def inspect(archive: Path) -> KMCManifest:
    """Inspect a .kmc archive and return its manifest."""
    manifest, _ = read_manifest_from_archive(archive)
    return manifest
