"""Format-specific parsers for AI model files.

Provides dedicated modules for reading metadata from AI model formats
without loading weights into memory. Each module degrades gracefully
when optional dependencies are unavailable.

Supported formats:
    - safetensors: Tensor metadata, shards, LoRA detection
    - gguf: Header parsing, version and tensor count
"""

from __future__ import annotations

from .gguf import GGUFInfo, read_gguf_info
from .safetensors import SafetensorsInfo, read_safetensors_info

__all__ = [
    "GGUFInfo",
    "SafetensorsInfo",
    "read_gguf_info",
    "read_safetensors_info",
]
