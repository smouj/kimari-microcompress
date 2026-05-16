"""Synthetic tensor fixtures for testing tensor-aware codecs.

Generates byte data that simulates various floating-point tensor
patterns without requiring real model files or heavy dependencies.

All generated data is deterministic (seeded) for reproducibility.
"""

from __future__ import annotations

import struct

# Use a fixed seed for reproducibility
_RNG_SEED = 42


def _simple_rng(seed: int = _RNG_SEED):
    """Simple deterministic PRNG for test data generation."""
    state = seed
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state


def generate_bf16_repeated_exponents(n_elements: int = 1024) -> bytes:
    """Generate BF16 data with repeated exponent patterns.

    BF16: 1 bit sign, 8 bits exponent, 7 bits mantissa.
    Creates values where exponents cluster around common values.

    Args:
        n_elements: Number of BF16 values to generate.

    Returns:
        Bytes representing n_elements BF16 values (2 bytes each).
    """
    rng = _simple_rng()
    data = bytearray()
    base_exponent = 127
    for _ in range(n_elements):
        sign = 0
        exp_offset = (next(rng) % 5) - 2
        exponent = base_exponent + exp_offset
        mantissa = next(rng) % 128
        value = (sign << 15) | (exponent << 7) | mantissa
        data.extend(struct.pack(">H", value))
    return bytes(data)


def generate_fp16_smooth_patterns(n_elements: int = 1024) -> bytes:
    """Generate FP16 data with smooth value patterns.

    FP16: 1 bit sign, 5 bits exponent, 10 bits mantissa.

    Args:
        n_elements: Number of FP16 values to generate.

    Returns:
        Bytes representing n_elements FP16 values (2 bytes each).
    """
    rng = _simple_rng()
    data = bytearray()
    base_exp = 15
    for i in range(n_elements):
        sign = 0
        exp = base_exp + (i % 4) - 1
        mantissa = (i * 17 + next(rng) % 32) % 1024
        value = (sign << 15) | (exp << 10) | mantissa
        data.extend(struct.pack(">H", value))
    return bytes(data)


def generate_fp32_simulated(n_elements: int = 512) -> bytes:
    """Generate FP32 data with simulated weight patterns.

    FP32: 1 bit sign, 8 bits exponent, 23 bits mantissa.

    Args:
        n_elements: Number of FP32 values to generate.

    Returns:
        Bytes representing n_elements FP32 values (4 bytes each).
    """
    rng = _simple_rng()
    data = bytearray()
    base_exponent = 127
    for _ in range(n_elements):
        sign = 0
        exp_offset = (next(rng) % 7) - 3
        exponent = base_exponent + exp_offset
        mantissa = next(rng) % (2**23)
        value = (sign << 31) | (exponent << 23) | mantissa
        data.extend(struct.pack(">I", value))
    return bytes(data)


def generate_random_bytes(size: int = 4096, seed: int = _RNG_SEED) -> bytes:
    """Generate random bytes with high entropy (hard to compress).

    Args:
        size: Number of bytes to generate.
        seed: Random seed for reproducibility.

    Returns:
        Random bytes.
    """
    rng = _simple_rng(seed)
    return bytes(next(rng) % 256 for _ in range(size))


def generate_compressible_bytes(size: int = 4096, seed: int = _RNG_SEED) -> bytes:
    """Generate highly compressible data with repeated patterns.

    Args:
        size: Number of bytes to generate.
        seed: Random seed for reproducibility.

    Returns:
        Compressible bytes.
    """
    rng = _simple_rng(seed)
    pattern_len = min(64, size)
    pattern = bytes(next(rng) % 256 for _ in range(pattern_len))
    result = bytearray()
    while len(result) < size:
        result.extend(pattern)
    return bytes(result[:size])


def generate_misaligned_bf16(n_elements: int = 100, extra_bytes: int = 3) -> bytes:
    """Generate BF16 data with extra bytes that don't align to element_size.

    Args:
        n_elements: Number of BF16 values.
        extra_bytes: Extra bytes to append (not aligned to 2-byte boundary).

    Returns:
        Bytes with n_elements BF16 values + extra_bytes trailing bytes.
    """
    data = generate_bf16_repeated_exponents(n_elements)
    rng = _simple_rng(123)
    tail = bytes(next(rng) % 256 for _ in range(extra_bytes))
    return data + tail
