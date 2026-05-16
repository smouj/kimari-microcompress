"""Experimental delta compression for .kmc archives.

Provides block-level delta encoding between similar checkpoints or models.
When packing with --delta-base, blocks that are identical to the base
archive are referenced instead of stored. Only changed blocks are stored.

WARNING: This is an experimental feature in KMC v0.8.0-alpha.
Delta archives require the base archive for reconstruction.
Do NOT apply delta by default.
"""

from __future__ import annotations

from .delta_codec import DeltaBlock, DeltaCodec
from .delta_planner import DeltaPlan, DeltaPlanner
from .similarity import block_similarity, file_similarity

__all__ = [
    "DeltaBlock",
    "DeltaCodec",
    "DeltaPlan",
    "DeltaPlanner",
    "block_similarity",
    "file_similarity",
]
