"""Core archive operations: pack, unpack, verify.

KMC file format layout:
    [Magic: 8 bytes "KMC\\x00\\x01\\x00\\x00\\x00"]
    [Manifest length: 8 bytes, big-endian uint64]
    [Manifest: JSON, UTF-8 encoded]
    [Block data: concatenated compressed blocks]
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .codecs import CodecId, compress_block, decompress_block
from .hashing import sha256_block, sha256_file
from .manifest import BlockEntry, FileEntry, KMCManifest

KMC_MAGIC = b"KMC\x00\x01\x00\x00\x00"
KMC_MAGIC_LEN = 8
MANIFEST_LEN_FMT = ">Q"  # 8-byte big-endian unsigned
MANIFEST_LEN_SIZE = struct.calcsize(MANIFEST_LEN_FMT)
KMC_FORMAT_VERSION = 1

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

    Checks performed:
    - No empty or whitespace-only path components.
    - No absolute paths (leading ``/`` or drive letters).
    - No ``..`` components that would escape the output directory.
    - No null bytes or other control characters.
    - No duplicate file entries (caller must check separately).
    - Final resolved path must start with the output directory prefix.
    """
    # Reject null bytes and control characters
    if "\x00" in relative_path:
        raise ExtractionError(f"Null byte in path: {relative_path!r}")
    if any(ord(c) < 0x20 for c in relative_path):
        raise ExtractionError(f"Control character in path: {relative_path!r}")

    # Reject empty or whitespace-only paths
    stripped = relative_path.strip()
    if not stripped:
        raise ExtractionError("Empty or whitespace-only path in manifest")

    # Reject absolute paths
    if relative_path.startswith("/"):
        raise ExtractionError(f"Absolute path not allowed: {relative_path!r}")
    # Windows-style absolute paths
    if len(relative_path) >= 2 and relative_path[1] == ":":
        raise ExtractionError(f"Windows-style absolute path not allowed: {relative_path!r}")

    # Check for consecutive slashes (before PurePosixPath normalizes them)
    if "//" in relative_path:
        raise ExtractionError(f"Empty path component (consecutive slashes): {relative_path!r}")

    # Parse with PurePosixPath and reject '..' components
    parts = PurePosixPath(relative_path).parts
    if ".." in parts:
        raise ExtractionError(f"Path traversal ('..') not allowed: {relative_path!r}")

    # Resolve and verify it stays within output_dir
    output_dir = output_dir.resolve()
    candidate = (output_dir / relative_path).resolve()

    # The resolved path must be within output_dir
    # Use os.path.commonpath to handle edge cases
    try:
        common = os.path.commonpath([str(output_dir), str(candidate)])
        if common != str(output_dir):
            raise ExtractionError(
                f"Path escapes output directory: {relative_path!r} "
                f"resolves to {candidate!r} (outside {output_dir!r})"
            )
    except ValueError:
        # Different drives on Windows
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
    integrity: str = "OK"  # OK or FAILED
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
    """Validate a manifest for structural correctness and security.

    Returns a list of errors found. Empty list means the manifest is valid.
    """
    errors: list[str] = []
    supported_codecs = {c.value for c in CodecId}
    seen_paths: set[str] = set()

    for file_entry in manifest.files:
        # Check for duplicate paths
        if file_entry.path in seen_paths:
            errors.append(f"Duplicate path in manifest: {file_entry.path!r}")
        seen_paths.add(file_entry.path)

        # Check path safety
        try:
            # Use a dummy output dir just for validation
            safe_join_extract_path(Path("/tmp/kmc_check"), file_entry.path)
        except ExtractionError as e:
            errors.append(f"Unsafe path in manifest: {file_entry.path!r} — {e}")

        # Check original_size
        if file_entry.original_size < 0:
            errors.append(
                f"Negative original_size for '{file_entry.path}': {file_entry.original_size}"
            )

        # Check block count consistency
        if file_entry.original_size == 0 and file_entry.blocks:
            errors.append(
                f"File '{file_entry.path}' has zero original_size but "
                f"{len(file_entry.blocks)} block(s)"
            )

        # Check blocks
        block_indices: set[int] = set()
        total_block_original = 0
        for block in file_entry.blocks:
            # Check for duplicate block indices
            if block.index in block_indices:
                errors.append(f"Duplicate block index {block.index} in '{file_entry.path}'")
            block_indices.add(block.index)

            # Check codec is supported
            if block.codec not in supported_codecs:
                errors.append(
                    f"Unsupported codec '{block.codec}' in block "
                    f"{block.index} of '{file_entry.path}'"
                )

            # Check sizes
            if block.compressed_size < 0:
                errors.append(
                    f"Negative compressed_size in block {block.index} of '{file_entry.path}'"
                )
            if block.original_size < 0:
                errors.append(
                    f"Negative original_size in block {block.index} of '{file_entry.path}'"
                )
            total_block_original += block.original_size

        # Check total block original size matches file original size
        if file_entry.original_size > 0 and total_block_original != file_entry.original_size:
            errors.append(
                f"Block size mismatch for '{file_entry.path}': "
                f"file original_size={file_entry.original_size}, "
                f"sum of block original_sizes={total_block_original}"
            )

    return errors


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------


def pack(
    source: Path,
    output: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    level: int = 3,
) -> None:
    """Pack a directory or single file into a .kmc archive.

    Args:
        source: Source directory or file to compress.
        output: Output .kmc file path.
        block_size: Block size in bytes (default 256 KiB).
        level: Compression level (codec-dependent).
    """
    source = Path(source).resolve()
    output = Path(output).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    manifest = KMCManifest(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    # Collect all compressed blocks and their metadata
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
        file_entry = FileEntry(
            path=rel_str,
            original_size=file_size,
            hash=file_hash,
            block_size=block_size,
        )

        with open(fpath, "rb") as f:
            block_index = 0
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break

                compressed = compress_block(chunk, level=level)
                block_hash = sha256_block(compressed.data)

                block_entry = BlockEntry(
                    index=block_index,
                    offset=0,  # placeholder, fixed below
                    compressed_size=compressed.compressed_size,
                    original_size=compressed.original_size,
                    codec=compressed.codec.value,
                    hash=block_hash,
                )
                file_entry.blocks.append(block_entry)
                all_blocks.append((len(manifest.files), block_index, block_entry, compressed.data))

                manifest.total_original_size += compressed.original_size
                manifest.total_compressed_size += compressed.compressed_size

                block_index += 1

        manifest.files.append(file_entry)

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


# ---------------------------------------------------------------------------
# Read manifest
# ---------------------------------------------------------------------------


def read_manifest_from_archive(archive: Path) -> tuple[KMCManifest, int]:
    """Read the manifest from a .kmc archive.

    Returns:
        Tuple of (manifest, data_start_offset).
    """
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

        # Validate JSON
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
    """Verify the integrity of a .kmc archive.

    Returns a list of errors found. Empty list means the archive is valid.
    """
    report = verify_full(archive)
    return report.errors


def verify_full(archive: Path) -> VerificationReport:
    """Perform a full verification of a .kmc archive and return a report.

    Checks:
    - Magic header
    - Format version
    - Manifest size bounds
    - Manifest JSON validity
    - Manifest structural validation (paths, codecs, sizes)
    - Block checksums
    - File hash reconstruction
    - Block count consistency
    - Size coherence
    """
    archive = Path(archive).resolve()
    report = VerificationReport(archive_path=str(archive))

    # Read manifest
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

    # Check format version
    if manifest.version != KMC_FORMAT_VERSION:
        report.warnings.append(
            f"Format version mismatch: archive is v{manifest.version}, "
            f"expected v{KMC_FORMAT_VERSION}"
        )

    # Validate manifest structure
    structural_errors = validate_manifest(manifest)
    if structural_errors:
        report.integrity = "FAILED"
        report.errors.extend(structural_errors)
        return report

    # Verify blocks and checksums
    archive_size = archive.stat().st_size
    with open(archive, "rb") as f:
        for file_entry in manifest.files:
            file_hasher = hashlib.sha256()
            file_reconstructed_size = 0

            for block in sorted(file_entry.blocks, key=lambda b: b.index):
                # Check block offset is within archive
                if block.offset + block.compressed_size > archive_size:
                    report.integrity = "FAILED"
                    report.errors.append(
                        f"Block {block.index} of '{file_entry.path}' "
                        f"exceeds archive size (offset={block.offset}, "
                        f"size={block.compressed_size}, "
                        f"archive={archive_size})"
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

                # Verify block hash
                actual_hash = sha256_block(block_data)
                if actual_hash != block.hash:
                    report.integrity = "FAILED"
                    report.errors.append(
                        f"Block {block.index} of '{file_entry.path}': "
                        f"checksum mismatch (expected={block.hash}, "
                        f"got={actual_hash})"
                    )
                    continue

                # Decompress to verify file hash
                try:
                    codec_id = CodecId(block.codec)
                    decompressed = decompress_block(block_data, codec_id, block.original_size)
                    file_hasher.update(decompressed.data)
                    file_reconstructed_size += len(decompressed.data)
                except Exception as e:
                    report.integrity = "FAILED"
                    report.errors.append(
                        f"Block {block.index} of '{file_entry.path}': decompression failed: {e}"
                    )
                    continue

            # Verify file hash (only if no block errors so far)
            if file_reconstructed_size == file_entry.original_size:
                actual_file_hash = file_hasher.hexdigest()
                if actual_file_hash != file_entry.hash:
                    report.integrity = "FAILED"
                    report.errors.append(
                        f"File '{file_entry.path}': hash mismatch "
                        f"(expected={file_entry.hash}, "
                        f"got={actual_file_hash})"
                    )
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
    """Unpack a .kmc archive to a directory.

    Args:
        archive: Path to the .kmc archive.
        output_dir: Output directory (created if it doesn't exist).

    Raises:
        ExtractionError: If any path in the manifest is unsafe.
        ValueError: If integrity checks fail during unpack.
    """
    archive = Path(archive).resolve()
    output_dir = Path(output_dir).resolve()

    manifest, data_start = read_manifest_from_archive(archive)

    # Pre-validate all paths before writing anything
    seen_paths: set[str] = set()
    for file_entry in manifest.files:
        # Check for duplicate paths
        if file_entry.path in seen_paths:
            raise ExtractionError(f"Duplicate path in manifest: {file_entry.path!r}")
        seen_paths.add(file_entry.path)

        # Validate path safety
        safe_join_extract_path(output_dir, file_entry.path)

    # Validate manifest structure
    structural_errors = validate_manifest(manifest)
    if structural_errors:
        raise ValueError(f"Manifest validation failed: {'; '.join(structural_errors)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(archive, "rb") as f:
        for file_entry in manifest.files:
            out_path = safe_join_extract_path(output_dir, file_entry.path)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Check for symlinks — do not follow or overwrite them
            if out_path.is_symlink():
                raise ExtractionError(f"Refusing to overwrite symlink: {out_path}")

            with open(out_path, "wb") as out_f:
                file_hasher = hashlib.sha256()

                for block in sorted(file_entry.blocks, key=lambda b: b.index):
                    f.seek(block.offset)
                    block_data = f.read(block.compressed_size)

                    # Verify block hash before decompression
                    actual_hash = sha256_block(block_data)
                    if actual_hash != block.hash:
                        raise ValueError(
                            f"Block {block.index} of "
                            f"'{file_entry.path}': "
                            f"checksum mismatch during unpack"
                        )

                    codec_id = CodecId(block.codec)
                    decompressed = decompress_block(block_data, codec_id, block.original_size)
                    out_f.write(decompressed.data)
                    file_hasher.update(decompressed.data)

                # Verify file hash
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
