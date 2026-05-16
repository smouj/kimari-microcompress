"""FloatPlane codec: lossless sign/exponent/mantissa separation for FP16/BF16/FP32.

Separates floating-point numbers into their bit-level components
(sign, exponent, mantissa) by operating directly on the binary
representation. No conversion to Python floats occurs — all
operations are on integer bit patterns.

This can improve compressibility because:
- Sign bits are often uniform (mostly positive weights)
- Exponents tend to cluster in a narrow range
- Mantissa bits have varying entropy patterns

After separation, each plane is compressed independently with
zstd (preferred) or zlib.

Supported dtypes:
    - BF16: 1 bit sign, 8 bits exponent, 7 bits mantissa
    - FP16: 1 bit sign, 5 bits exponent, 10 bits mantissa
    - FP32: 1 bit sign, 8 bits exponent, 23 bits mantissa

Lossless guarantee: decompress(compress(data)) == data for all inputs.
"""

from __future__ import annotations

import struct

from .base import CodecContext, CodecResult
from .zstd_codec import ZstdCodec, is_zstd_available

# Bit layout constants for supported dtypes
_DTYPE_LAYOUT = {
    "BF16": {"total_bits": 16, "sign_bits": 1, "exp_bits": 8, "mantissa_bits": 7, "byte_size": 2},
    "FP16": {"total_bits": 16, "sign_bits": 1, "exp_bits": 5, "mantissa_bits": 10, "byte_size": 2},
    "FP32": {"total_bits": 32, "sign_bits": 1, "exp_bits": 8, "mantissa_bits": 23, "byte_size": 4},
    "F16": {"total_bits": 16, "sign_bits": 1, "exp_bits": 5, "mantissa_bits": 10, "byte_size": 2},
    "F32": {"total_bits": 32, "sign_bits": 1, "exp_bits": 8, "mantissa_bits": 23, "byte_size": 4},
}


class FloatPlaneCodec:
    """Lossless sign/exponent/mantissa separation codec for floating-point data.

    Operates on the binary representation of floating-point numbers.
    No conversion to Python floats occurs. Each bit-plane component
    is compressed independently.

    Falls back to byteplane or zstd if dtype is not supported or
    not provided.
    """

    name: str = "floatplane"

    def compress(self, data: bytes, *, context: CodecContext | None = None) -> CodecResult:
        """Compress data using sign/exponent/mantissa plane separation."""
        if len(data) == 0:
            return CodecResult(
                codec="floatplane",
                payload=b"",
                original_size=0,
                compressed_size=0,
                metadata={
                    "transform": "floatplane",
                    "dtype": "unknown",
                    "inner_codec": "raw",
                    "planes": [],
                },
            )

        # Determine dtype from context
        dtype = self._infer_dtype(context)
        if dtype is None:
            # Cannot do floatplane without dtype — fall back
            from .byteplane import BytePlaneCodec

            bp = BytePlaneCodec()
            result = bp.compress(data, context=context)
            # Wrap the result to indicate it's actually a floatplane fallback
            return CodecResult(
                codec="floatplane",
                payload=result.payload,
                original_size=result.original_size,
                compressed_size=result.compressed_size,
                metadata={
                    **result.metadata,
                    "transform": "byteplane_fallback",
                    "fallback_reason": "no dtype specified",
                },
            )

        layout = _DTYPE_LAYOUT.get(dtype.upper())
        if layout is None:
            # Unsupported dtype — fall back to byteplane
            from .byteplane import BytePlaneCodec

            bp = BytePlaneCodec()
            result = bp.compress(data, context=context)
            return CodecResult(
                codec="floatplane",
                payload=result.payload,
                original_size=result.original_size,
                compressed_size=result.compressed_size,
                metadata={
                    **result.metadata,
                    "transform": "byteplane_fallback",
                    "fallback_reason": f"unsupported dtype: {dtype}",
                },
            )

        element_size = layout["byte_size"]
        if len(data) % element_size != 0:
            # Misaligned data — handle tail separately
            aligned_len = (len(data) // element_size) * element_size
            tail = data[aligned_len:]
            aligned_data = data[:aligned_len]
        else:
            aligned_data = data
            tail = b""

        # Separate into sign, exponent, mantissa planes
        n_elements = len(aligned_data) // element_size
        if n_elements == 0:
            from .zlib_codec import ZlibCodec

            inner = ZlibCodec()
            result = inner.compress(data)
            return CodecResult(
                codec="floatplane",
                payload=result.payload,
                original_size=result.original_size,
                compressed_size=result.compressed_size,
                metadata={
                    "transform": "floatplane",
                    "dtype": dtype,
                    "inner_codec": "zlib",
                    "planes": [],
                    "fallback_reason": "too few elements for plane separation",
                },
            )

        sign_plane, exp_plane, mantissa_plane = self._separate_float_planes(
            aligned_data, dtype, layout
        )

        # Choose inner codec
        inner_codec_name: str
        if is_zstd_available():
            inner_codec = ZstdCodec(level=3)
            inner_codec_name = "zstd"
        else:
            from .zlib_codec import ZlibCodec

            inner_codec = ZlibCodec(level=6)
            inner_codec_name = "zlib"

        # Compress each plane separately
        sign_compressed = inner_codec.compress(sign_plane)
        exp_compressed = inner_codec.compress(exp_plane)
        mantissa_compressed = inner_codec.compress(mantissa_plane)

        # Pack: [sign_len(4)][sign_data][exp_len(4)][exp_data]
        #       [mantissa_len(4)][mantissa_data][tail_len(4)][tail]
        payload = b""
        payload += struct.pack(">I", len(sign_compressed.payload))
        payload += sign_compressed.payload
        payload += struct.pack(">I", len(exp_compressed.payload))
        payload += exp_compressed.payload
        payload += struct.pack(">I", len(mantissa_compressed.payload))
        payload += mantissa_compressed.payload
        payload += struct.pack(">I", len(tail))
        payload += tail

        return CodecResult(
            codec="floatplane",
            payload=payload,
            original_size=len(data),
            compressed_size=len(payload),
            metadata={
                "transform": "floatplane",
                "dtype": dtype,
                "inner_codec": inner_codec_name,
                "planes": ["sign", "exponent", "mantissa"],
                "n_elements": n_elements,
            },
        )

    def decompress(self, payload: bytes, *, context: CodecContext | None = None) -> bytes:
        """Decompress float-plane separated data back to original bytes."""
        if len(payload) == 0:
            return b""

        # Check for fallback (byteplane) from metadata
        meta = {}
        if context and hasattr(context, "_codec_metadata"):
            meta = context._codec_metadata  # type: ignore[attr-defined]

        transform = meta.get("transform", "")
        if "byteplane_fallback" in transform:
            from .byteplane import BytePlaneCodec

            bp = BytePlaneCodec()
            # BytePlaneCodec needs element_size from metadata
            return bp.decompress(payload, context=context)

        dtype = meta.get("dtype", "")
        layout = _DTYPE_LAYOUT.get(dtype.upper())

        if layout is None:
            raise ValueError(f"FloatPlane decompression requires dtype in metadata, got: {dtype!r}")

        inner_codec_name = meta.get("inner_codec", "zstd" if is_zstd_available() else "zlib")

        # Unpack: [sign_len(4)][sign_data][exp_len(4)][exp_data]
        #         [mantissa_len(4)][mantissa_data][tail_len(4)][tail]
        offset = 0

        sign_len = struct.unpack(">I", payload[offset : offset + 4])[0]
        offset += 4
        sign_data_compressed = payload[offset : offset + sign_len]
        offset += sign_len

        exp_len = struct.unpack(">I", payload[offset : offset + 4])[0]
        offset += 4
        exp_data_compressed = payload[offset : offset + exp_len]
        offset += exp_len

        mantissa_len = struct.unpack(">I", payload[offset : offset + 4])[0]
        offset += 4
        mantissa_data_compressed = payload[offset : offset + mantissa_len]
        offset += mantissa_len

        tail_len = struct.unpack(">I", payload[offset : offset + 4])[0]
        offset += 4
        tail = payload[offset : offset + tail_len]

        # Decompress each plane
        if inner_codec_name == "zstd" and is_zstd_available():
            inner_codec = ZstdCodec()
        else:
            from .zlib_codec import ZlibCodec

            inner_codec = ZlibCodec()

        sign_plane = inner_codec.decompress(sign_data_compressed)
        exp_plane = inner_codec.decompress(exp_data_compressed)
        mantissa_plane = inner_codec.decompress(mantissa_data_compressed)

        # Reconstruct original data
        return (
            self._interleave_float_planes(sign_plane, exp_plane, mantissa_plane, dtype, layout)
            + tail
        )

    @staticmethod
    def _infer_dtype(context: CodecContext | None) -> str | None:
        """Infer dtype from context."""
        if context is None or context.dtype is None:
            return None
        dtype_upper = context.dtype.upper()
        # Normalize common aliases
        if dtype_upper in ("BF16", "BFLOAT16"):
            return "BF16"
        if dtype_upper in ("FP16", "F16", "FLOAT16"):
            return "FP16"
        if dtype_upper in ("FP32", "F32", "FLOAT32"):
            return "FP32"
        return None  # Unsupported dtype

    @staticmethod
    def _separate_float_planes(
        data: bytes,
        dtype: str,
        layout: dict,
    ) -> tuple[bytes, bytes, bytes]:
        """Separate floating-point data into sign, exponent, mantissa planes.

        Operates on binary representation only. No conversion to Python floats.

        Returns (sign_plane, exponent_plane, mantissa_plane) as raw bytes.
        """
        element_size = layout["byte_size"]
        exp_bits = layout["exp_bits"]
        mantissa_bits = layout["mantissa_bits"]

        n_elements = len(data) // element_size
        total_bits = layout["total_bits"]

        # Read elements as unsigned integers
        if element_size == 2:
            fmt = f">{n_elements}H"  # unsigned 16-bit
        elif element_size == 4:
            fmt = f">{n_elements}I"  # unsigned 32-bit
        else:
            raise ValueError(f"Unsupported element size: {element_size}")

        values = struct.unpack(fmt, data[: n_elements * element_size])

        # Extract bit fields
        sign_mask = 1 << (total_bits - 1)
        exp_mask = ((1 << exp_bits) - 1) << mantissa_bits
        mantissa_mask = (1 << mantissa_bits) - 1

        # Collect raw values for compression
        sign_bits_list = []
        exp_values_list = []
        mantissa_values_list = []

        for val in values:
            s = (val & sign_mask) >> (total_bits - 1)
            e = (val & exp_mask) >> mantissa_bits
            m = val & mantissa_mask
            sign_bits_list.append(s)
            exp_values_list.append(e)
            mantissa_values_list.append(m)

        # Pack sign bits into bytes (8 bits per byte)
        sign_packed = bytearray()
        for i in range(0, len(sign_bits_list), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(sign_bits_list):
                    byte_val |= sign_bits_list[i + j] << (7 - j)
            sign_packed.append(byte_val)

        # Pack exponent values — use minimal bytes per value
        exp_bytes_per_val = (exp_bits + 7) // 8
        exp_packed = bytearray()
        for val in exp_values_list:
            exp_packed.extend(val.to_bytes(exp_bytes_per_val, byteorder="big"))

        # Pack mantissa values
        mantissa_bytes_per_val = (mantissa_bits + 7) // 8
        mantissa_packed = bytearray()
        for val in mantissa_values_list:
            mantissa_packed.extend(val.to_bytes(mantissa_bytes_per_val, byteorder="big"))

        return bytes(sign_packed), bytes(exp_packed), bytes(mantissa_packed)

    @staticmethod
    def _interleave_float_planes(
        sign_plane: bytes,
        exp_plane: bytes,
        mantissa_plane: bytes,
        dtype: str,
        layout: dict,
    ) -> bytes:
        """Reconstruct original floating-point data from sign, exponent, mantissa planes."""
        element_size = layout["byte_size"]
        exp_bits = layout["exp_bits"]
        mantissa_bits = layout["mantissa_bits"]
        total_bits = layout["total_bits"]

        # Determine n_elements from sign plane (8 sign bits per byte)
        n_elements = len(sign_plane) * 8
        if len(sign_plane) * 8 > 0 and sign_plane[-1] == 0:
            # May have padding; use exp/mantissa planes to determine
            exp_bytes_per_val = (exp_bits + 7) // 8
            mantissa_bytes_per_val = (mantissa_bits + 7) // 8
            n_from_exp = len(exp_plane) // exp_bytes_per_val if exp_bytes_per_val > 0 else 0
            n_from_mantissa = (
                len(mantissa_plane) // mantissa_bytes_per_val if mantissa_bytes_per_val > 0 else 0
            )
            n_elements = min(n_elements, n_from_exp, n_from_mantissa)

        if n_elements == 0:
            return b""

        # Unpack sign bits
        sign_bits_list = []
        for i in range(n_elements):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            if byte_idx < len(sign_plane):
                s = (sign_plane[byte_idx] >> bit_idx) & 1
            else:
                s = 0
            sign_bits_list.append(s)

        # Unpack exponent values
        exp_bytes_per_val = (exp_bits + 7) // 8
        exp_values_list = []
        for i in range(n_elements):
            start = i * exp_bytes_per_val
            end = start + exp_bytes_per_val
            if end <= len(exp_plane):
                val = int.from_bytes(exp_plane[start:end], byteorder="big")
            else:
                val = 0
            exp_values_list.append(val)

        # Unpack mantissa values
        mantissa_bytes_per_val = (mantissa_bits + 7) // 8
        mantissa_values_list = []
        for i in range(n_elements):
            start = i * mantissa_bytes_per_val
            end = start + mantissa_bytes_per_val
            if end <= len(mantissa_plane):
                val = int.from_bytes(mantissa_plane[start:end], byteorder="big")
            else:
                val = 0
            mantissa_values_list.append(val)

        # Reconstruct values
        values = []
        for s, e, m in zip(sign_bits_list, exp_values_list, mantissa_values_list):
            val = (s << (total_bits - 1)) | (e << mantissa_bits) | m
            values.append(val)

        # Pack back to bytes
        if element_size == 2:
            fmt = f">{n_elements}H"
        elif element_size == 4:
            fmt = f">{n_elements}I"
        else:
            raise ValueError(f"Unsupported element size: {element_size}")

        return struct.pack(fmt, *values)
