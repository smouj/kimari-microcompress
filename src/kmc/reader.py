"""KMCReader: partial-access API for .kmc archives.

Provides read-only access to .kmc archive contents without requiring
full decompression. Supports listing files and tensors, reading specific
files or tensors, and selective extraction to disk.

Usage::

    from kmc.reader import KMCReader

    with KMCReader("model.kmc") as reader:
        print(reader.list_files())
        print(reader.list_tensors())
        data = reader.read_file("config.json")
        tensor_bytes = reader.read_tensor("model.layers.0.mlp.down_proj.weight")

KMCReader builds block, file, and tensor indexes on open, enabling
efficient partial reads. Only the blocks needed for a requested file
or tensor are read and decompressed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .archive import read_manifest_from_archive, safe_join_extract_path
from .hashing import sha256_block
from .index import (
    BlockIndex,
    BlockLocation,
    FileIndex,
    FileLocation,
    TensorIndex,
    TensorLocation,
)
from .manifest import KMCManifest


class KMCReader:
    """Read-only partial-access interface for .kmc archives.

    Opens the archive, reads the manifest, and builds indexes for
    efficient partial access. Supports reading individual files or
    tensors without decompressing the entire archive.

    Can be used as a context manager for automatic resource cleanup.

    Attributes:
        archive_path: Path to the .kmc archive file.
        manifest: The archive's manifest.
        block_index: Index of all blocks in the archive.
        file_index: Index of all files in the archive.
        tensor_index: Index of all tensors in the archive (may be empty).
    """

    def __init__(self, archive_path: str | Path) -> None:
        """Open a .kmc archive for reading.

        Args:
            archive_path: Path to the .kmc archive file.

        Raises:
            FileNotFoundError: If the archive does not exist.
            ValueError: If the archive is not a valid .kmc file.
        """
        self.archive_path = Path(archive_path).resolve()

        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {self.archive_path}")

        # Read manifest
        self.manifest, self._data_start = read_manifest_from_archive(self.archive_path)

        # Build indexes
        self.block_index = BlockIndex.from_manifest(self.manifest, self.archive_path)
        self.file_index = FileIndex.from_manifest(self.manifest)
        self.tensor_index = TensorIndex.from_manifest(self.manifest)

        # Keep the file handle closed between operations for safety
        self._fh = None

    def __enter__(self) -> KMCReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close any open file handles."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # -----------------------------------------------------------------------
    # Listing APIs
    # -----------------------------------------------------------------------

    def list_files(self) -> list[str]:
        """List all file paths in the archive.

        Returns:
            List of relative file paths (POSIX format).
        """
        return self.file_index.list_files()

    def list_tensors(self) -> list[str]:
        """List all tensor names in the archive.

        Returns an empty list if the archive was created without
        --tensor-aware mode.

        Returns:
            List of tensor name strings.
        """
        return self.tensor_index.list_tensors()

    def get_manifest(self) -> KMCManifest:
        """Return the archive manifest.

        Returns:
            The KMCManifest instance for this archive.
        """
        return self.manifest

    def get_file_info(self, path: str) -> FileLocation | None:
        """Get metadata for a specific file.

        Args:
            path: Relative path of the file.

        Returns:
            FileLocation if found, None otherwise.
        """
        return self.file_index.get(path)

    def get_tensor_info(self, name: str) -> TensorLocation | None:
        """Get metadata for a specific tensor.

        Args:
            name: Name of the tensor.

        Returns:
            TensorLocation if found, None otherwise.
        """
        return self.tensor_index.get(name)

    # -----------------------------------------------------------------------
    # Reading APIs
    # -----------------------------------------------------------------------

    def read_file(self, path: str) -> bytes:
        """Read and return the full contents of a file from the archive.

        Only the blocks needed for this file are read and decompressed.
        The file's SHA-256 hash is verified after reconstruction.

        Args:
            path: Relative path of the file to read.

        Returns:
            The file's uncompressed contents as bytes.

        Raises:
            FileNotFoundError: If the file is not in the archive.
            ValueError: If the reconstructed file hash doesn't match.
        """
        file_loc = self.file_index.get(path)
        if file_loc is None:
            raise FileNotFoundError(f"File not found in archive: {path!r}")

        return self._read_blocks_and_verify(file_loc.block_ids, file_loc.sha256, path)

    def read_file_range(self, path: str, offset: int, length: int) -> bytes:
        """Read a byte range from a file in the archive.

        Reads only the blocks that overlap the requested range, then
        slices the result. This is more efficient than reading the entire
        file when only a small portion is needed.

        The algorithm:
        1. Find the file's block list and their offsets within the file.
        2. Determine which blocks overlap the [offset, offset+length) range.
        3. Read and decompress only those blocks.
        4. Verify block checksums.
        5. Assemble the requested range from the decompressed blocks.

        Args:
            path: Relative path of the file.
            offset: Byte offset to start reading from.
            length: Number of bytes to read.

        Returns:
            The requested byte range as bytes.

        Raises:
            FileNotFoundError: If the file is not in the archive.
            ValueError: If offset or length are out of range.
        """
        if offset < 0:
            raise ValueError(f"Negative offset: {offset}")
        if length < 0:
            raise ValueError(f"Negative length: {length}")

        file_loc = self.file_index.get(path)
        if file_loc is None:
            raise FileNotFoundError(f"File not found in archive: {path!r}")

        if offset >= file_loc.size:
            return b""

        # Clamp length to file size
        effective_length = min(length, file_loc.size - offset)
        if effective_length <= 0:
            return b""

        # Get blocks for this file
        blocks = self.block_index.get_blocks_for_file(path)
        if not blocks:
            raise ValueError(f"No blocks found for file: {path!r}")

        # Compute block boundaries within the file
        # Each block has original_size and we know their order
        block_starts: list[int] = []
        current_start = 0
        for block in blocks:
            block_starts.append(current_start)
            current_start += block.original_size

        # Find which blocks overlap the requested range [offset, offset+effective_length)
        range_start = offset
        range_end = offset + effective_length

        first_block = -1
        last_block = -1
        for i, block_start in enumerate(block_starts):
            block_end = block_start + blocks[i].original_size
            if block_end > range_start and block_start < range_end:
                if first_block == -1:
                    first_block = i
                last_block = i

        if first_block == -1:
            return b""

        # Read and decompress only the needed blocks
        result = b""
        for i in range(first_block, last_block + 1):
            block_loc = blocks[i]
            block_data = self._read_and_decompress_block(block_loc)
            result += block_data

        # Extract the requested range from the assembled blocks
        # The first needed byte is at: range_start - block_starts[first_block]
        local_offset = range_start - block_starts[first_block]

        return result[local_offset : local_offset + effective_length]

    def read_tensor(self, name: str) -> bytes:
        """Read and return the raw bytes of a tensor from the archive.

        Only the blocks that belong to the requested tensor are read
        and decompressed. Block checksums are verified.

        Note: This returns the raw bytes of the tensor data, not a
        PyTorch or NumPy tensor. To convert to a tensor, use the
        experimental safetensors loader if available.

        Args:
            name: Name of the tensor.

        Returns:
            The tensor's raw bytes.

        Raises:
            FileNotFoundError: If the tensor is not in the archive.
            ValueError: If block checksums don't match.
        """
        tensor_loc = self.tensor_index.get(name)
        if tensor_loc is None:
            raise FileNotFoundError(f"Tensor not found in archive: {name!r}")

        if not tensor_loc.block_ids:
            # Tensor has no dedicated blocks; it's embedded within a file
            file_loc = self.file_index.get(tensor_loc.file_path)
            if file_loc is None:
                raise FileNotFoundError(f"File for tensor not found: {tensor_loc.file_path!r}")

            # Read the full file and extract the tensor range
            # Try to find byte offset/size from manifest tensor_entries
            for fentry in self.manifest.files:
                if fentry.path == tensor_loc.file_path:
                    for te in fentry.tensor_entries:
                        if te.name == name:
                            file_data = self.read_file(tensor_loc.file_path)
                            header_size = 0
                            # Find header size for safetensors
                            try:
                                from .formats.safetensors import read_safetensors_info

                                info = read_safetensors_info(
                                    self.archive_path.parent / tensor_loc.file_path
                                )
                                header_size = info.header_size
                            except (ValueError, OSError, ImportError):
                                pass
                            start = header_size + te.byte_offset
                            end = start + te.byte_size
                            if end <= len(file_data):
                                return file_data[start:end]
                            break

            raise ValueError(
                f"Cannot locate tensor data for {name!r}: "
                "no block-level or file-level tensor mapping available"
            )

        # Read the blocks that belong to this tensor
        data = b""
        for block_id in tensor_loc.block_ids:
            block_loc = self.block_index.get_by_id(block_id)
            if block_loc is None:
                raise ValueError(f"Block {block_id} not found in index")
            block_data = self._read_and_decompress_block(block_loc)
            data += block_data

        return data

    # -----------------------------------------------------------------------
    # Extraction APIs
    # -----------------------------------------------------------------------

    def extract_file(self, path: str, output_dir: str | Path) -> Path:
        """Extract a single file from the archive to disk.

        Args:
            path: Relative path of the file to extract.
            output_dir: Directory to extract the file into.

        Returns:
            Path to the extracted file.

        Raises:
            FileNotFoundError: If the file is not in the archive.
            ExtractionError: If the path is unsafe.
        """
        output_dir = Path(output_dir).resolve()
        data = self.read_file(path)
        out_path = safe_join_extract_path(output_dir, path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return out_path

    def extract_tensor(self, name: str, output_dir: str | Path) -> Path:
        """Extract a tensor's raw bytes to disk.

        The tensor data is written to a file named after the tensor
        (with .bin extension) in the output directory.

        Args:
            name: Name of the tensor.
            output_dir: Directory to extract the tensor into.

        Returns:
            Path to the extracted tensor file.

        Raises:
            FileNotFoundError: If the tensor is not in the archive.
        """
        output_dir = Path(output_dir).resolve()
        data = self.read_tensor(name)

        # Create a safe filename from the tensor name
        safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        if not safe_name:
            safe_name = "tensor"
        out_path = output_dir / f"{safe_name}.bin"

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return out_path

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _read_and_decompress_block(self, block_loc: BlockLocation) -> bytes:
        """Read a single block from the archive and decompress it.

        Verifies the block checksum before decompression.

        Args:
            block_loc: BlockLocation with archive offset and metadata.

        Returns:
            Decompressed block data.

        Raises:
            ValueError: If the block checksum doesn't match.
        """
        from .archive import _decompress_block_with_metadata

        with open(self.archive_path, "rb") as f:
            f.seek(block_loc.archive_offset)
            block_data = f.read(block_loc.compressed_size)

        if len(block_data) < block_loc.compressed_size:
            raise ValueError(
                f"Could not read full block at offset {block_loc.archive_offset}: "
                f"expected {block_loc.compressed_size} bytes, got {len(block_data)}"
            )

        # Verify block checksum
        if block_loc.block_hash:
            actual_hash = sha256_block(block_data)
            if actual_hash != block_loc.block_hash:
                raise ValueError(
                    f"Block at offset {block_loc.archive_offset}: "
                    f"checksum mismatch (expected={block_loc.block_hash[:16]}..., "
                    f"got={actual_hash[:16]}...)"
                )

        # Decompress
        return _decompress_block_with_metadata(
            block_data,
            block_loc.codec,
            block_loc.original_size,
            block_loc.codec_metadata,
        )

    def _read_blocks_and_verify(
        self,
        block_ids: list[int],
        expected_hash: str,
        file_path: str,
    ) -> bytes:
        """Read multiple blocks, concatenate, and verify the file hash.

        Args:
            block_ids: Ordered list of block IDs.
            expected_hash: Expected SHA-256 hash of the concatenated data.
            file_path: File path for error messages.

        Returns:
            Concatenated decompressed data.

        Raises:
            ValueError: If the reconstructed file hash doesn't match.
        """
        hasher = hashlib.sha256()
        data = b""

        for block_id in block_ids:
            block_loc = self.block_index.get_by_id(block_id)
            if block_loc is None:
                raise ValueError(f"Block {block_id} not found in index")
            block_data = self._read_and_decompress_block(block_loc)
            data += block_data
            hasher.update(block_data)

        actual_hash = hasher.hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"File '{file_path}': hash mismatch "
                f"(expected={expected_hash[:16]}..., "
                f"got={actual_hash[:16]}...)"
            )

        return data
