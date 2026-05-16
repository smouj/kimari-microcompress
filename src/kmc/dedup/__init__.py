"""Cross-file deduplication for .kmc archives.

Detects and eliminates duplicate blocks within a .kmc archive, storing
only one copy of each unique block and referencing it from other locations.
Deduplication is exact (byte-for-byte SHA-256 match), not approximate.

WARNING: This is an experimental feature in KMC v0.8.0-alpha.
"""

from __future__ import annotations

from .block_fingerprint import compute_block_fingerprint, fingerprint_block_data
from .dedup_index import DedupEntry, DedupIndex
from .planner import DedupPlan, DedupPlanner

__all__ = [
    "compute_block_fingerprint",
    "fingerprint_block_data",
    "DedupIndex",
    "DedupEntry",
    "DedupPlan",
    "DedupPlanner",
]
