"""File index: maps file paths to their metadata and block lists.

FileIndex provides fast lookup of file metadata by path, enabling
selective extraction of specific files without scanning the entire
manifest. Each entry records the file's original size, SHA-256 hash,
and the list of block IDs that compose it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileLocation:
    """Location and metadata for a single file within the archive.

    Attributes:
        path: Relative path of the file inside the archive (POSIX).
        size: Original (uncompressed) size of the file in bytes.
        sha256: SHA-256 hash of the original file.
        block_ids: Ordered list of block IDs that compose this file.
    """

    path: str
    size: int
    sha256: str
    block_ids: list[int] = field(default_factory=list)


class FileIndex:
    """Index of all files in a .kmc archive, optimized for partial access.

    Supports lookup by file path (exact or pattern match) and provides
    the block IDs needed to extract a specific file.
    """

    def __init__(self) -> None:
        self._files: dict[str, FileLocation] = {}
        self._ordered: list[FileLocation] = []

    def add(self, file_loc: FileLocation) -> None:
        """Add a file location to the index."""
        self._files[file_loc.path] = file_loc
        self._ordered.append(file_loc)

    def get(self, path: str) -> FileLocation | None:
        """Look up a file by its relative path."""
        return self._files.get(path)

    def list_files(self) -> list[str]:
        """List all file paths in the archive."""
        return [f.path for f in self._ordered]

    @property
    def total_files(self) -> int:
        """Total number of indexed files."""
        return len(self._files)

    def match_pattern(self, pattern: str) -> list[FileLocation]:
        """Match files against a fnmatch-style pattern.

        Args:
            pattern: Glob pattern (e.g., '*.json', 'tokenizer*').

        Returns:
            List of matching FileLocation instances.
        """
        import fnmatch

        results = []
        for file_loc in self._ordered:
            if fnmatch.fnmatch(file_loc.path, pattern) or fnmatch.fnmatch(
                file_loc.path.split("/")[-1], pattern
            ):
                results.append(file_loc)
        return results

    @classmethod
    def from_manifest(cls, manifest: object) -> FileIndex:
        """Build a FileIndex from a KMCManifest.

        Args:
            manifest: KMCManifest instance.

        Returns:
            Populated FileIndex.
        """
        index = cls()
        global_block_id = 0

        for file_entry in manifest.files:  # type: ignore[attr-defined]
            block_ids: list[int] = []
            for block in file_entry.blocks:  # type: ignore[attr-defined]
                block_ids.append(global_block_id)
                global_block_id += 1

            file_loc = FileLocation(
                path=file_entry.path,  # type: ignore[attr-defined]
                size=file_entry.original_size,  # type: ignore[attr-defined]
                sha256=file_entry.hash,  # type: ignore[attr-defined]
                block_ids=block_ids,
            )
            index.add(file_loc)

        return index
