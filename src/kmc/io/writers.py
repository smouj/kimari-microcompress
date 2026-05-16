"""Block-based file writers for streaming decompression."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def write_blocks(path: Path, blocks: list[bytes]) -> None:
    """Write a list of byte blocks to a file sequentially.

    Creates parent directories if they don't exist.

    Args:
        path: Path to the output file.
        blocks: List of byte strings to write in order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for block in blocks:
            f.write(block)


def write_blocks_from_iter(path: Path, blocks: Iterable[bytes]) -> None:
    """Write an iterable of byte blocks to a file sequentially.

    Unlike ``write_blocks``, this accepts any iterable (including generators),
    allowing truly streaming writes without holding all blocks in memory.

    Creates parent directories if they don't exist.

    Args:
        path: Path to the output file.
        blocks: Iterable of byte strings to write in order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for block in blocks:
            f.write(block)
