"""Legacy codec interface: backward-compatible compression API.

This module preserves the original CodecId enum and compress_block/
decompress_block functions used by v0.2/v0.3 archives. New code
should use the codec subpackage directly.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

try:
    import zstandard as zstd

    _HAS_ZSTD = True
except ImportError:  # pragma: no cover
    _HAS_ZSTD = False


class CodecId(str, Enum):
    """Supported compression codec identifiers."""

    ZSTD = "zstd"
    ZLIB = "zlib"
    RAW = "raw"
    BYTEPLANE = "byteplane"
    FLOATPLANE = "floatplane"


@dataclass(frozen=True)
class CodecResult:
    """Result of a compression/decompression operation (legacy interface)."""

    data: bytes
    codec: CodecId
    original_size: int
    compressed_size: int

    @property
    def ratio(self) -> float:
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size


class Codec(Protocol):
    """Protocol for compression codecs (legacy interface)."""

    def compress(self, data: bytes, level: int = 3) -> CodecResult: ...
    def decompress(self, data: bytes, original_size: int) -> CodecResult: ...


class ZstdCodec:
    """Zstandard compression codec."""

    def compress(self, data: bytes, level: int = 3) -> CodecResult:
        if not _HAS_ZSTD:
            raise RuntimeError("zstandard package not installed")
        cctx = zstd.ZstdCompressor(level=level)
        compressed = cctx.compress(data)
        return CodecResult(
            data=compressed,
            codec=CodecId.ZSTD,
            original_size=len(data),
            compressed_size=len(compressed),
        )

    def decompress(self, data: bytes, original_size: int) -> CodecResult:
        if not _HAS_ZSTD:
            raise RuntimeError("zstandard package not installed")
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(data, max_output_size=original_size)
        return CodecResult(
            data=decompressed,
            codec=CodecId.ZSTD,
            original_size=original_size,
            compressed_size=len(data),
        )


class ZlibCodec:
    """zlib compression codec."""

    def compress(self, data: bytes, level: int = 6) -> CodecResult:
        compressed = zlib.compress(data, level=level)
        return CodecResult(
            data=compressed,
            codec=CodecId.ZLIB,
            original_size=len(data),
            compressed_size=len(compressed),
        )

    def decompress(self, data: bytes, original_size: int) -> CodecResult:
        decompressed = zlib.decompress(data)
        return CodecResult(
            data=decompressed,
            codec=CodecId.ZLIB,
            original_size=original_size,
            compressed_size=len(data),
        )


class RawCodec:
    """Passthrough codec."""

    def compress(self, data: bytes, level: int = 0) -> CodecResult:
        return CodecResult(
            data=data,
            codec=CodecId.RAW,
            original_size=len(data),
            compressed_size=len(data),
        )

    def decompress(self, data: bytes, original_size: int) -> CodecResult:
        return CodecResult(
            data=data,
            codec=CodecId.RAW,
            original_size=original_size,
            compressed_size=len(data),
        )


def select_codec(data: bytes, block_size: int = 256 * 1024) -> Codec:
    """Select the best codec for the given data."""
    if _HAS_ZSTD:
        return ZstdCodec()
    return ZlibCodec()


def compress_block(data: bytes, level: int = 3) -> CodecResult:
    """Compress a block using the best available codec."""
    codec = select_codec(data)
    result = codec.compress(data, level=level)
    if result.compressed_size >= result.original_size:
        raw = RawCodec()
        return raw.compress(data)
    return result


def decompress_block(data: bytes, codec_id: CodecId | str, original_size: int) -> CodecResult:
    """Decompress a block given its codec identifier and original size."""
    codec_str = codec_id.value if isinstance(codec_id, CodecId) else str(codec_id)

    if codec_str in ("byteplane", "floatplane"):
        raise ValueError(
            f"Codec '{codec_str}' requires codec_metadata from the manifest. "
            f"Use the new archive decompression API for v0.4+ archives."
        )

    codecs_map: dict[str, Codec] = {
        "zstd": ZstdCodec(),
        "zlib": ZlibCodec(),
        "raw": RawCodec(),
    }
    codec = codecs_map.get(codec_str)
    if codec is None:
        raise ValueError(f"Unsupported codec: {codec_str!r}")
    return codec.decompress(data, original_size)
