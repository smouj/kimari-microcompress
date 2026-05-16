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
import os
import struct
from pathlib import Path, PurePosixPath

from .codecs import CodecId, compress_block, decompress_block
from .hashing import sha256_block, sha256_file
from .manifest import BlockEntry, FileEntry, KMCManifest

KMC_MAGIC = b"KMC\x00\x01\x00\x00\x00"
KMC_MAGIC_LEN = 8
MANIFEST_LEN_FMT = ">Q"  # 8-byte big-endian unsigned
MANIFEST_LEN_SIZE = struct.calcsize(MANIFEST_LEN_FMT)

DEFAULT_BLOCK_SIZE = 256 * 1024  # 256 KiB


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
    # We'll compute offsets after we know the manifest size
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
        # When packing a directory, paths are relative to the source dir itself
        # When packing a single file, use just the filename
        rel_path = fpath.relative_to(source) if source.is_dir() else fpath.name
        # Use POSIX paths for portability
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

                # Offset is a placeholder; will be fixed below
                block_entry = BlockEntry(
                    index=block_index,
                    offset=0,  # placeholder
                    compressed_size=compressed.compressed_size,
                    original_size=compressed.original_size,
                    codec=compressed.codec.value,
                    hash=block_hash,
                )
                file_entry.blocks.append(block_entry)
                all_blocks.append(
                    (len(manifest.files), block_index, block_entry, compressed.data)
                )

                manifest.total_original_size += compressed.original_size
                manifest.total_compressed_size += compressed.compressed_size

                block_index += 1

        manifest.files.append(file_entry)

    # Compute correct offsets iteratively: the manifest size depends on the
    # offset values (larger offsets → more digits → larger JSON), which in
    # turn depends on the manifest size. Iterate until stable.
    for _ in range(20):  # convergence guaranteed in a few iterations
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

        manifest_raw = f.read(manifest_len)
        if len(manifest_raw) < manifest_len:
            raise ValueError("Truncated manifest data")

        data_start = KMC_MAGIC_LEN + MANIFEST_LEN_SIZE + manifest_len
        return KMCManifest.from_bytes(manifest_raw), data_start


def verify(archive: Path) -> list[str]:
    """Verify the integrity of a .kmc archive.

    Returns a list of errors found. Empty list means the archive is valid.
    """
    errors: list[str] = []

    try:
        manifest, data_start = read_manifest_from_archive(archive)
    except (ValueError, OSError) as e:
        return [f"Failed to read archive: {e}"]

    archive_size = archive.stat().st_size
    with open(archive, "rb") as f:
        for file_entry in manifest.files:
            for block in file_entry.blocks:
                # Verify block offset is within archive
                if block.offset + block.compressed_size > archive_size:
                    errors.append(
                        f"Block {block.index} of '{file_entry.path}' "
                        f"exceeds archive size (offset={block.offset}, "
                        f"size={block.compressed_size}, archive={archive_size})"
                    )
                    continue

                f.seek(block.offset)
                block_data = f.read(block.compressed_size)

                if len(block_data) < block.compressed_size:
                    errors.append(
                        f"Block {block.index} of '{file_entry.path}': "
                        f"could not read full block data"
                    )
                    continue

                # Verify block hash
                actual_hash = sha256_block(block_data)
                if actual_hash != block.hash:
                    errors.append(
                        f"Block {block.index} of '{file_entry.path}': "
                        f"hash mismatch (expected={block.hash}, got={actual_hash})"
                    )

    return errors


def unpack(archive: Path, output_dir: Path) -> None:
    """Unpack a .kmc archive to a directory.

    Args:
        archive: Path to the .kmc archive.
        output_dir: Output directory (created if it doesn't exist).
    """
    archive = Path(archive).resolve()
    output_dir = Path(output_dir).resolve()

    manifest, data_start = read_manifest_from_archive(archive)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(archive, "rb") as f:
        for file_entry in manifest.files:
            # SECURITY: sanitize path to prevent path traversal
            out_path = _safe_path(output_dir, file_entry.path)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "wb") as out_f:
                file_hasher = hashlib.sha256()

                for block in sorted(file_entry.blocks, key=lambda b: b.index):
                    f.seek(block.offset)
                    block_data = f.read(block.compressed_size)

                    # Verify block hash before decompression
                    actual_hash = sha256_block(block_data)
                    if actual_hash != block.hash:
                        raise ValueError(
                            f"Block {block.index} of '{file_entry.path}': "
                            f"hash mismatch during unpack"
                        )

                    codec_id = CodecId(block.codec)
                    decompressed = decompress_block(
                        block_data, codec_id, block.original_size
                    )
                    out_f.write(decompressed.data)
                    file_hasher.update(decompressed.data)

                # Verify file hash
                actual_file_hash = file_hasher.hexdigest()
                if actual_file_hash != file_entry.hash:
                    raise ValueError(
                        f"File '{file_entry.path}': hash mismatch "
                        f"(expected={file_entry.hash}, got={actual_file_hash})"
                    )


def _safe_path(base: Path, rel_path: str) -> Path:
    """Sanitize a relative path to prevent path traversal attacks.

    Ensures the resolved path stays within the base directory.
    """
    # Normalize and resolve
    candidate = (base / rel_path).resolve()

    # Ensure the resolved path is within the base directory
    if not str(candidate).startswith(str(base.resolve())):
        raise ValueError(
            f"Path traversal detected: '{rel_path}' resolves outside "
            f"the output directory '{base}'"
        )

    # Also reject absolute paths and paths with '..'
    parts = PurePosixPath(rel_path).parts
    if rel_path.startswith("/") or ".." in parts:
        raise ValueError(
            f"Unsafe path component in '{rel_path}': "
            f"absolute paths and '..' are not allowed"
        )

    return candidate


def inspect(archive: Path) -> KMCManifest:
    """Inspect a .kmc archive and return its manifest."""
    manifest, _ = read_manifest_from_archive(archive)
    return manifest
