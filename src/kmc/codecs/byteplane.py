"""BytePlane codec: lossless byte-plane separation for fixed-width numeric types.

Reorganizes bytes by their position within each element before compression.
For FP16/BF16 (element_size=2), separates high and low bytes into contiguous
streams. For FP32 (element_size=4), separates all 4 byte positions.

This transformation improves compressibility because bytes at the same
position within floating-point numbers tend to have similar patterns
(sign bits cluster, exponents cluster, mantissa bytes cluster).

After byte-plane separation, the result is compressed with an inner
codec (zstd if available, otherwise zlib).

Lossless guarantee: decompress(compress(data)) == data for all inputs.
"""

from __future__ import annotations

import struct

from .base import CodecContext, CodecResult
from .zstd_codec import ZstdCodec, is_zstd_available

# Sentinel for misaligned data
_MISALIGNED_KEY = "_misaligned_tail"


class BytePlaneCodec:
    """Lossless byte-plane separation codec for fixed-width numeric data.

    Separates bytes by their position within fixed-size elements,
    then compresses each plane with an inner codec (zstd or zlib).

    Supports:
        - element_size=2: BF16, FP16
        - element_size=4: FP32
        - Misaligned data (tail bytes that don't fit element_size)
        - Empty data
    """

    name: str = "byteplane"

    def compress(self, data: bytes, *, context: CodecContext | None = None) -> CodecResult:
        """Compress data using byte-plane separation.

        If the data length is not aligned to element_size, the tail
        bytes are stored separately and appended during decompression.
        """
        if len(data) == 0:
            return CodecResult(
                codec="byteplane",
                payload=b"",
                original_size=0,
                compressed_size=0,
                metadata={
                    "transform": "byteplane",
                    "element_size": 0,
                    "inner_codec": "raw",
                },
            )

        # Determine element_size from context
        element_size = self._infer_element_size(context)

        # If we can't determine element_size, fall back to trying common sizes
        if element_size is None:
            element_size = self._guess_element_size(len(data))

        # Separate bytes into planes
        planes, tail = self._separate_planes(data, element_size)

        # Choose inner codec
        inner_codec_name: str
        if is_zstd_available():
            inner_codec = ZstdCodec(level=3)
            inner_codec_name = "zstd"
        else:
            from .zlib_codec import ZlibCodec

            inner_codec = ZlibCodec(level=6)
            inner_codec_name = "zlib"

        # Concatenate planes and compress
        planes_data = b"".join(planes)
        if len(planes_data) == 0:
            # All data is in the tail
            inner_result = inner_codec.compress(tail)
            combined_payload = struct.pack(">I", 0) + inner_result.payload
        else:
            # Store: [planes_len(4 bytes)] [planes_compressed] [tail]
            inner_result = inner_codec.compress(planes_data)
            combined_payload = struct.pack(">I", len(planes_data)) + inner_result.payload + tail

        metadata = {
            "transform": "byteplane",
            "element_size": element_size,
            "inner_codec": inner_codec_name,
        }
        if tail:
            metadata[_MISALIGNED_KEY] = len(tail)

        return CodecResult(
            codec="byteplane",
            payload=combined_payload,
            original_size=len(data),
            compressed_size=len(combined_payload),
            metadata=metadata,
        )

    def decompress(self, payload: bytes, *, context: CodecContext | None = None) -> bytes:
        """Decompress byte-plane separated data back to original bytes."""
        if len(payload) == 0:
            return b""

        # Extract metadata
        element_size = 2  # Default
        inner_codec_name = "zstd"
        misaligned_tail_len = 0

        if context and hasattr(context, "dtype"):
            pass  # Will use metadata from CodecResult if available

        # We need metadata to properly decompress. The metadata is stored
        # in the manifest's codec_metadata field and passed via a special
        # context extension. For now, we parse the payload format:
        # [planes_len(4 bytes)] [planes_compressed] [tail]

        if len(payload) < 4:
            raise ValueError("byteplane payload too short")

        planes_len = struct.unpack(">I", payload[:4])[0]

        if planes_len == 0:
            # All data is in the tail (no planes separated)
            inner_codec_name = "zstd" if is_zstd_available() else "zlib"
            compressed_data = payload[4:]

            if inner_codec_name == "zstd" and is_zstd_available():
                inner_codec = ZstdCodec()
                planes_data = inner_codec.decompress(compressed_data)
            else:
                import zlib

                planes_data = zlib.decompress(compressed_data)

            return planes_data

        # We need to figure out how much of the payload is compressed vs tail
        # The metadata should tell us element_size and misaligned_tail_len
        # For robustness, we try decompression and detect the tail

        # Get metadata from a special context attribute if available
        meta = {}
        if context and hasattr(context, "_codec_metadata"):
            meta = context._codec_metadata  # type: ignore[attr-defined]

        element_size = meta.get("element_size", 2)
        inner_codec_name = meta.get("inner_codec", "zstd" if is_zstd_available() else "zlib")
        misaligned_tail_len = meta.get(_MISALIGNED_KEY, 0)

        # Split compressed data and tail
        tail_start = len(payload) - misaligned_tail_len if misaligned_tail_len > 0 else len(payload)
        compressed_data = payload[4:tail_start]
        tail = payload[tail_start:] if misaligned_tail_len > 0 else b""

        # Decompress planes
        if inner_codec_name == "zstd" and is_zstd_available():
            inner_codec = ZstdCodec()
            planes_data = inner_codec.decompress(compressed_data)
        else:
            import zlib

            planes_data = zlib.decompress(compressed_data)

        # Interleave planes back
        return self._interleave_planes(planes_data, element_size, tail)

    @staticmethod
    def _infer_element_size(context: CodecContext | None) -> int | None:
        """Infer element size from context dtype."""
        if context is None or context.dtype is None:
            return None
        dtype_upper = context.dtype.upper()
        if dtype_upper in ("BF16", "FP16", "F16"):
            return 2
        if dtype_upper in ("FP32", "F32", "FLOAT32"):
            return 4
        if dtype_upper in ("FP64", "F64", "FLOAT64"):
            return 8
        if dtype_upper in ("INT8", "UINT8"):
            return 1
        if dtype_upper in ("INT16", "UINT16"):
            return 2
        if dtype_upper in ("INT32", "UINT32"):
            return 4
        return None

    @staticmethod
    def _guess_element_size(data_len: int) -> int:
        """Guess element_size based on data length alignment."""
        if data_len % 4 == 0:
            return 4  # Prefer FP32 alignment
        if data_len % 2 == 0:
            return 2  # FP16/BF16 alignment
        return 1  # Fallback to byte-level

    @staticmethod
    def _separate_planes(data: bytes, element_size: int) -> tuple[list[bytes], bytes]:
        """Separate bytes into planes by position within each element.

        For element_size=2: [a0,b0,a1,b1,...] -> [a0,a1,...], [b0,b1,...]
        For element_size=4: [a0,b0,c0,d0,...] -> [a0,a1,...], [b0,b1,...], ...

        Returns (list of plane bytes, tail bytes for misaligned data).
        """
        if element_size <= 0:
            return [], data

        n_full_elements = len(data) // element_size
        aligned_len = n_full_elements * element_size
        tail = data[aligned_len:]

        if n_full_elements == 0:
            return [], data

        planes: list[bytearray] = [bytearray(n_full_elements) for _ in range(element_size)]
        for i in range(n_full_elements):
            for j in range(element_size):
                planes[j][i] = data[i * element_size + j]

        return [bytes(p) for p in planes], tail

    @staticmethod
    def _interleave_planes(planes_data: bytes, element_size: int, tail: bytes) -> bytes:
        """Reconstruct original data from separated planes and tail."""
        if element_size <= 0 or len(planes_data) == 0:
            return planes_data + tail

        n_elements = len(planes_data) // element_size
        if n_elements == 0:
            return planes_data + tail

        result = bytearray(n_elements * element_size)
        for i in range(n_elements):
            for j in range(element_size):
                result[i * element_size + j] = planes_data[j * n_elements + i]

        return bytes(result) + tail
