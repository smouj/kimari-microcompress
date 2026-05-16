"""GGUF quantized block codec: conservative compression for GGUF data.

This codec provides a format-aware compression strategy for blocks that
contain GGUF quantized tensor data. It applies conservative transforms:

- BytePlane for scale/metadata blocks (if applicable)
- zstd or zlib for quantized payload blocks
- raw fallback if compression doesn't improve size

This codec does NOT:
- Apply FloatPlane to quantized data (quantized values have no FP structure)
- Reinterpret or modify the data bytes
- Make assumptions about GGML types without metadata

WARNING: This is an experimental codec in KMC v0.8.0-alpha.
It only activates when gguf_aware=True in the CodecContext.
"""

from __future__ import annotations

from .base import Codec, CodecContext, CodecResult


class GGUFQuantCodec(Codec):
    """Codec for GGUF quantized blocks with conservative strategy.

    Applies lossless compression strategies appropriate for quantized
    tensor data found in GGUF files. The strategy is:
    1. Try zstd compression on the payload (quantized data compresses well)
    2. Try zlib as fallback
    3. Try byteplane separation (helps with scale factors)
    4. Fall back to raw if nothing helps

    This codec must NOT be applied to floating-point data (BF16, FP16, FP32).
    FloatPlane should be used for those dtypes instead.
    """

    name = "gguf_quant_block"

    def compress(self, data: bytes, *, context: CodecContext | None = None) -> CodecResult:
        """Compress GGUF quantized block data.

        Args:
            data: Original block data to compress.
            context: Codec context with dtype and format hints.

        Returns:
            CodecResult with the best compressed payload.
        """
        # Safety check: don't apply to float dtypes
        if context and context.dtype:
            dtype_upper = context.dtype.upper().strip()
            float_dtypes = {
                "BF16",
                "FP16",
                "F16",
                "FP32",
                "F32",
                "FP64",
                "F64",
                "BFLOAT16",
                "FLOAT16",
                "FLOAT32",
                "FLOAT64",
            }
            if dtype_upper in float_dtypes:
                # Should not apply gguf_quant to float data
                # Fall back to raw — selector should have picked floatplane instead
                return CodecResult(
                    payload=data,
                    original_size=len(data),
                    compressed_size=len(data),
                    codec=self.name,
                    metadata={"fallback": "raw", "reason": "float_dtype"},
                )

        best_payload = data
        best_codec = "raw"
        best_metadata: dict = {"gguf_aware": True}
        candidates_tried: list[str] = []

        # Try zstd first (best for quantized data)
        try:
            from .zstd_codec import ZstdCodec, is_zstd_available

            if is_zstd_available():
                zstd = ZstdCodec()
                result = zstd.compress(data, context=context)
                if result.compressed_size < len(best_payload):
                    best_payload = result.payload
                    best_codec = "zstd"
                    best_metadata["inner_codec"] = "zstd"
                candidates_tried.append("zstd")
        except Exception:
            candidates_tried.append("zstd:failed")

        # Try zlib
        try:
            from .zlib_codec import ZlibCodec

            zlib_codec = ZlibCodec()
            result = zlib_codec.compress(data, context=context)
            if result.compressed_size < len(best_payload):
                best_payload = result.payload
                best_codec = "zlib"
                best_metadata["inner_codec"] = "zlib"
            candidates_tried.append("zlib")
        except Exception:
            candidates_tried.append("zlib:failed")

        # Try byteplane (can help with scale factors in quantized blocks)
        if context and context.dtype:
            dtype_upper = context.dtype.upper().strip()
            # BytePlane only helps for fixed-width element types
            quant_dtypes = {
                "INT8",
                "UINT8",
                "INT16",
                "UINT16",
                "Q4_0",
                "Q4_1",
                "Q5_0",
                "Q5_1",
                "Q8_0",
                "Q2_K",
                "Q3_K",
                "Q4_K",
                "Q5_K",
                "Q6_K",
                "IQ2_XXS",
                "IQ2_XS",
                "IQ3_XXS",
                "Q4_0_4_4",
                "Q4_0_4_8",
                "Q4_0_8_8",
            }
            if dtype_upper in quant_dtypes:
                try:
                    from .byteplane import BytePlaneCodec

                    bp = BytePlaneCodec()
                    result = bp.compress(data, context=context)
                    if result.compressed_size < len(best_payload):
                        best_payload = result.payload
                        best_codec = "byteplane"
                        best_metadata["inner_codec"] = "byteplane"
                        if result.metadata:
                            best_metadata["byteplane_meta"] = result.metadata
                    candidates_tried.append("byteplane")
                except Exception:
                    candidates_tried.append("byteplane:failed")

        best_metadata["candidates_tried"] = candidates_tried

        compressed_size = len(best_payload) if best_codec != "raw" else len(data)

        return CodecResult(
            payload=best_payload if best_codec != "raw" else data,
            original_size=len(data),
            compressed_size=compressed_size,
            codec=self.name,
            metadata=best_metadata,
        )

    def decompress(self, payload: bytes, *, context: CodecContext | None = None) -> bytes:
        """Decompress a GGUF quantized block.

        Uses the metadata to determine which inner codec was used.

        Args:
            payload: Compressed block data.
            context: Codec context with metadata from compression.

        Returns:
            Original (uncompressed) block data.

        Raises:
            ValueError: If the inner codec is unknown or metadata is missing.
        """
        if context is None:
            raise ValueError(
                "GGUFQuantCodec requires context with codec_metadata for decompression"
            )

        metadata = getattr(context, "_codec_metadata", {}) or {}
        inner_codec = metadata.get("inner_codec", "raw")

        if inner_codec == "raw" or not metadata:
            # No compression was applied
            return payload

        if inner_codec == "zstd":
            from .zstd_codec import ZstdCodec

            zstd = ZstdCodec()
            return zstd.decompress(payload, context=context)

        if inner_codec == "zlib":
            from .zlib_codec import ZlibCodec

            zlib_codec = ZlibCodec()
            return zlib_codec.decompress(payload, context=context)

        if inner_codec == "byteplane":
            from .byteplane import BytePlaneCodec

            bp = BytePlaneCodec()
            return bp.decompress(payload, context=context)

        raise ValueError(f"Unknown inner codec for gguf_quant_block: {inner_codec!r}")
