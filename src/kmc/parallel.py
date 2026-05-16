"""Parallel block compression and decompression using ThreadPoolExecutor.

Provides optional parallelism for pack and unpack operations. When --jobs 1
(the default), behavior is identical to sequential operation. Results are
always deterministic: block order in the output file matches the manifest order,
regardless of worker completion order.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable


@dataclass
class BlockWorkItem:
    """A unit of work for parallel block compression."""

    file_index: int
    block_index: int
    data: bytes
    codec_name: str
    context: object  # CodecContext, but typed as object to avoid circular import


@dataclass
class BlockResult:
    """Result of compressing or decompressing a single block."""

    file_index: int
    block_index: int
    compressed_payload: bytes
    codec_used: str
    codec_metadata: dict
    block_hash: str
    original_size: int
    compressed_size: int


def resolve_jobs(jobs: int | str) -> int:
    """Resolve the --jobs parameter to an actual worker count.

    Args:
        jobs: Number of workers. 1 means sequential, 'auto' uses cpu_count.

    Returns:
        Positive integer worker count (minimum 1).
    """
    if isinstance(jobs, str):
        if jobs == "auto":
            return max(1, os.cpu_count() or 1)
        try:
            jobs = int(jobs)
        except ValueError:
            return 1
    return max(1, int(jobs))


def compress_blocks_parallel(
    work_items: list[BlockWorkItem],
    compress_fn: Callable,
    jobs: int = 1,
) -> list[BlockResult]:
    """Compress blocks in parallel using ThreadPoolExecutor.

    The compress_fn signature must be:
        compress_fn(data: bytes, codec_name: str, context) ->
            tuple[bytes, str, dict]  # (payload, codec_used, metadata)

    Results are returned in the same order as work_items, ensuring
    deterministic output regardless of parallel execution order.

    Args:
        work_items: List of BlockWorkItem instances to compress.
        compress_fn: Function that compresses a single block.
        jobs: Number of parallel workers. 1 = sequential.

    Returns:
        List of BlockResult instances in the same order as work_items.
    """
    if jobs <= 1 or len(work_items) <= 1:
        return _compress_sequential(work_items, compress_fn)

    results: dict[tuple[int, int], BlockResult] = {}

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_key = {}
        for item in work_items:
            future = executor.submit(compress_fn, item.data, item.codec_name, item.context)
            future_to_key[future] = (item.file_index, item.block_index)

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                payload, codec_used, metadata = future.result()
            except Exception as e:
                raise RuntimeError(f"Worker failed for block {key[1]} of file {key[0]}: {e}") from e

            from kmc.hashing import sha256_block

            results[key] = BlockResult(
                file_index=key[0],
                block_index=key[1],
                compressed_payload=payload,
                codec_used=codec_used,
                codec_metadata=metadata,
                block_hash=sha256_block(payload),
                original_size=len(
                    next(
                        item.data
                        for item in work_items
                        if item.file_index == key[0] and item.block_index == key[1]
                    )
                ),
                compressed_size=len(payload),
            )

    # Return in original order
    ordered = []
    for item in work_items:
        key = (item.file_index, item.block_index)
        result = results[key]
        result.original_size = len(item.data)
        ordered.append(result)

    return ordered


def _compress_sequential(
    work_items: list[BlockWorkItem],
    compress_fn: Callable,
) -> list[BlockResult]:
    """Compress blocks sequentially (jobs=1)."""
    from kmc.hashing import sha256_block

    results = []
    for item in work_items:
        payload, codec_used, metadata = compress_fn(item.data, item.codec_name, item.context)
        results.append(
            BlockResult(
                file_index=item.file_index,
                block_index=item.block_index,
                compressed_payload=payload,
                codec_used=codec_used,
                codec_metadata=metadata,
                block_hash=sha256_block(payload),
                original_size=len(item.data),
                compressed_size=len(payload),
            )
        )
    return results


def decompress_blocks_parallel(
    work_items: list[tuple[bytes, str, int, dict]],
    decompress_fn: Callable,
    jobs: int = 1,
) -> list[bytes]:
    """Decompress blocks in parallel using ThreadPoolExecutor.

    The decompress_fn signature must be:
        decompress_fn(payload, codec_name, original_size, codec_metadata) -> bytes

    Args:
        work_items: List of (payload, codec_name, original_size, codec_metadata).
        decompress_fn: Function that decompresses a single block.
        jobs: Number of parallel workers. 1 = sequential.

    Returns:
        List of decompressed bytes in the same order as work_items.
    """
    if jobs <= 1 or len(work_items) <= 1:
        return [decompress_fn(*item) for item in work_items]

    results: dict[int, bytes] = {}

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_idx = {}
        for i, item in enumerate(work_items):
            future = executor.submit(decompress_fn, *item)
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                raise RuntimeError(f"Decompression worker failed for block {idx}: {e}") from e

    return [results[i] for i in range(len(work_items))]
