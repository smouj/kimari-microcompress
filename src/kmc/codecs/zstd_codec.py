"""zstd codec: Zstandard compression (optional, best ratio/speed).

Uses the ``zstandard`` Python package for compression. Falls back
gracefully when the package is not installed. When available, zstd
typically offers better compression ratios and faster decompression
than zlib.
"""

from __future__ import annotations

from .base import CodecContext, CodecResult

try:
    import zstandard as _zstd

    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


class ZstdCodec:
    """Zstandard compression codec — optional, best ratio/speed.

    Requires the ``zstandard`` package. Raises RuntimeError if the
    package is not installed.
    """

    name: str = "zstd"

    def __init__(self, level: int = 3) -> None:
        self.level = level

    def compress(self, data: bytes, *, context: CodecContext | None = None) -> CodecResult:
        """Compress data with zstd."""
        if not _HAS_ZSTD:
            raise RuntimeError("zstandard package not installed — pip install zstandard")
        cctx = _zstd.ZstdCompressor(level=self.level)
        compressed = cctx.compress(data)
        return CodecResult(
            codec="zstd",
            payload=compressed,
            original_size=len(data),
            compressed_size=len(compressed),
            metadata={"level": self.level},
        )

    def decompress(self, payload: bytes, *, context: CodecContext | None = None) -> bytes:
        """Decompress zstd-compressed data."""
        if not _HAS_ZSTD:
            raise RuntimeError("zstandard package not installed — pip install zstandard")
        # Use max_output_size from context or default to a generous limit
        max_size = 256 * 1024 * 1024  # 256 MB default
        if context and context.original_size:
            max_size = context.original_size
        dctx = _zstd.ZstdDecompressor()
        return dctx.decompress(payload, max_output_size=max_size)


def is_zstd_available() -> bool:
    """Check if the zstandard package is installed."""
    return _HAS_ZSTD
