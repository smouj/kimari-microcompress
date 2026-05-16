# Cross-File Deduplication

> **Status**: Experimental (KMC v0.8.0-alpha)
> **Manifest version**: 7 (format_version field)
> **CLI flag**: `--dedup`

## Overview

Cross-file deduplication is an experimental feature in KMC v0.8.0-alpha that identifies and eliminates redundant block data across multiple files within a single `.kmc` archive. When two or more blocks contain identical original (uncompressed) data, only the first occurrence — called the **canonical block** — is written to the archive. Subsequent duplicates store only a lightweight reference (`dedup_ref`) in the manifest, pointing back to the canonical block's global block ID.

This is particularly effective for ML model archives where files frequently share identical data regions: sharded safetensors models contain duplicated metadata headers across shards, LoRA adapter archives may include duplicate configuration files, and training checkpoint directories often carry repeated optimizer state templates across step snapshots.

## How It Works

### SHA-256 Fingerprinting

Deduplication in KMC operates by computing a SHA-256 hash of each block's **original, uncompressed data** during the packing phase. The fingerprinting pipeline is implemented in `kmc.dedup.block_fingerprint`:

1. **Block data is split** from the source file(s) at the configured `block_size` boundary (default: 262,144 bytes / 256 KB).
2. **Each block's raw bytes are hashed** using `hashlib.sha256(data).hexdigest()`, producing a 64-character hex string.
3. **The hash is compared** against the `DedupIndex`, an in-memory dictionary that maps SHA-256 fingerprints to `DedupEntry` records.
4. **If a match is found**, the block is marked as a duplicate: its `dedup_ref` field is set to the canonical block's global block ID, and the block's compressed data is **not written** to the archive output stream.
5. **If no match is found**, the block becomes a new canonical entry in the index, and its compressed data is written normally.

The `DedupPlanner` (in `kmc.dedup.planner`) orchestrates this process. It accumulates blocks via `add_block(global_block_id, original_data)`, delegates fingerprinting and index lookup to `DedupIndex`, and finally produces a `DedupPlan` containing the complete set of dedup references, unique block IDs, and savings statistics.

### DedupPlan Structure

The `DedupPlan` dataclass contains the following fields:

| Field | Type | Description |
|---|---|---|
| `enabled` | `bool` | Whether deduplication is active (True when `--dedup` is used) |
| `dedup_refs` | `dict[int, int]` | Mapping from duplicate block ID → canonical block ID |
| `unique_block_ids` | `set[int]` | Set of block IDs whose data is actually written to the archive |
| `total_blocks` | `int` | Total number of blocks analyzed across all files |
| `unique_blocks` | `int` | Number of blocks with unique data |
| `deduplicated_blocks` | `int` | Number of blocks identified as duplicates |
| `saved_bytes` | `int` | Estimated bytes saved (sum of original sizes of deduplicated blocks) |

### DedupIndex Internals

The `DedupIndex` (in `kmc.dedup.dedup_index`) maintains two internal data structures:

- `_entries: dict[str, DedupEntry]` — maps SHA-256 fingerprint → `DedupEntry` (canonical block metadata + list of duplicate block IDs).
- `_block_to_entry: dict[int, str]` — maps global block ID → SHA-256 fingerprint for reverse lookups.

Each `DedupEntry` records the canonical block's global ID, file index, block index, fingerprint, original size, and a list of duplicate block IDs that reference it.

## When Deduplication Helps

Deduplication is most effective in the following scenarios:

### 1. Duplicate Files in the Same Archive

When a directory contains multiple copies of the same file (e.g., `config.json` appearing in both the root and a subdirectory), every block of the duplicate file will be deduplicated. Savings approach 100% of the duplicate file's original size.

### 2. Shared Tensor Data Across Shards

Sharded Hugging Face models (e.g., `model-00001-of-00005.safetensors` through `model-00005-of-00005.safetensors`) include a shared safetensors header in each shard. While the tensor payload differs between shards, the header regions are often identical. Deduplication captures these shared regions at the block level.

### 3. Repeated Optimizer State in Checkpoints

Training checkpoint directories at different steps (e.g., `checkpoint-1000/` and `checkpoint-2000/`) may share identical optimizer state templates or scheduler state binaries when packed into a single archive. Deduplication eliminates the redundant copies.

### 4. LoRA Adapter + Base Model Archives

When packing a combined archive that includes both a LoRA adapter and the base model, any overlapping configuration files or shared tokenizer data will be deduplicated.

## CLI Usage

Enable deduplication with the `--dedup` flag on the `pack` command:

```bash
# Basic dedup-enabled packing
kmc pack ./my-model/ ./model.kmc --dedup

# Combined with tensor-aware mode for best results
kmc pack ./my-model/ ./model.kmc --dedup --tensor-aware

# With parallel workers and custom block size
kmc pack ./my-model/ ./model.kmc --dedup --tensor-aware --jobs 4 --block-size 524288

# Dedup + GGUF-aware mode for quantized models
kmc pack ./gguf-model/ ./model.kmc --dedup --gguf-aware
```

The `--dedup` flag can be combined with all other pack options. During packing, you will see output indicating dedup activity:

```
Packing ./my-model/ -> ./model.kmc --dedup (block_size=262144, level=3)
```

## Manifest Fields

When deduplication is active, the manifest gains two types of metadata:

### Top-Level `deduplication` Field

The `KMCManifest.deduplication` dictionary records aggregate dedup statistics:

```json
{
  "deduplication": {
    "enabled": true,
    "fingerprint": "sha256",
    "unique_blocks": 247,
    "deduplicated_blocks": 13,
    "saved_bytes": 3407872
  }
}
```

| Key | Description |
|---|---|
| `enabled` | Whether deduplication was active during packing |
| `fingerprint` | Hash algorithm used (always `"sha256"` in v0.8) |
| `unique_blocks` | Number of blocks with unique data written to the archive |
| `deduplicated_blocks` | Number of duplicate blocks that were skipped |
| `saved_bytes` | Total original-size bytes saved by deduplication |

### Per-Block `dedup_ref` Field

Each `BlockEntry` in the manifest has a `dedup_ref` field (default: `-1`). When a block is a duplicate, this field contains the global block ID of the canonical block:

```json
{
  "index": 5,
  "offset": 0,
  "compressed_size": 0,
  "original_size": 262144,
  "codec": "raw",
  "hash": "",
  "dedup_ref": 2
}
```

A `dedup_ref` of `-1` means the block is not a duplicate (it is either canonical or dedup was not enabled). A non-negative value means this block's data can be reconstructed by reading the canonical block with that global ID.

## Impact on Other Operations

### Verify

The `verify` command correctly handles deduplicated archives. When verifying a block with `dedup_ref >= 0`, the verifier resolves the reference to the canonical block and verifies the canonical block's checksum instead. This ensures that deduplicated archives pass integrity checks without requiring the duplicate block data to be present.

### Unpack

During unpacking, blocks with `dedup_ref >= 0` are reconstructed by reading and decompressing the canonical block, then placing the data at the appropriate position in the output file. The unpacker maintains a cache of recently decompressed canonical blocks to avoid redundant decompression when multiple duplicates reference the same canonical block within a single file.

### KMCReader (Partial Access)

The `KMCReader` API transparently handles dedup references. When reading a file or tensor that contains deduplicated blocks, the reader resolves `dedup_ref` fields to find the canonical block, reads and decompresses it, and uses the result for the requesting block. Users of the `read_file()`, `read_tensor()`, and `read_file_range()` APIs do not need to handle dedup logic explicitly.

```python
from kmc.reader import KMCReader

with KMCReader("model.kmc") as reader:
    # Works transparently even if some blocks are deduplicated
    config = reader.read_file("config.json")
    weights = reader.read_tensor("model.layers.0.self_attn.q_proj.weight")
```

## Limitations

### Exact Match Only

KMC deduplication uses exact SHA-256 fingerprinting. Two blocks are considered duplicates **only if** their original, uncompressed bytes are identical. This means:

- **No approximate deduplication**: Blocks that differ by even a single byte are stored separately. There is no similarity-based or fuzzy matching.
- **No cross-dtype dedup**: A block containing BF16 tensor data and a block containing FP16 data that happen to have the same byte pattern will be deduplicated (which is correct, since the bytes are identical), but semantically different data with similar patterns will not.
- **Block-boundary sensitivity**: Identical data at different offsets within a file may not align to the same block boundaries, preventing dedup. Using a consistent `block_size` across related archives maximizes dedup effectiveness.

### No Inter-Archive Dedup

Deduplication only works **within a single archive**. Two separate `.kmc` files cannot share deduplicated blocks. If you need cross-archive dedup, consider using delta compression (`--delta-base`) instead.

### Memory Overhead

The `DedupIndex` maintains SHA-256 fingerprints and metadata for every block in memory during packing. For very large archives with millions of blocks, this can consume significant RAM. Each `DedupEntry` stores approximately 200 bytes of metadata plus the fingerprint string (64 bytes), so an archive with 1 million blocks requires roughly 250 MB of index memory.

### No Dedup on Streaming Pack

When using streaming pack with progress reporting, the dedup planner must see all blocks before writing begins. This means the full dedup analysis pass occurs before any data is written to the output file, adding latency proportional to the number of unique blocks.

## Examples

### Example 1: Sharded Model with Duplicate Headers

```bash
# Pack a sharded model — shared safetensors headers will be deduplicated
kmc pack ./llama-7b/ ./llama-7b.kmc --dedup --tensor-aware

# Inspect the dedup statistics
kmc inspect ./llama-7b.kmc --dedup

# Output:
# Deduplication:
#   Enabled: yes
#   Unique blocks: 312
#   Deduplicated blocks: 8
#   Saved bytes: 2.00 MB
```

### Example 2: Multi-Checkpoint Archive

```bash
# Pack multiple checkpoints into one archive — shared optimizer state deduplicates
kmc pack ./checkpoints/ ./all-checkpoints.kmc --dedup --tensor-aware

# Verify integrity (dedup refs are resolved automatically)
kmc verify ./all-checkpoints.kmc

# Unpack (canonical blocks are used to reconstruct duplicates)
kmc unpack ./all-checkpoints.kmc ./restored-checkpoints/
```

### Example 3: Programmatic Dedup Analysis

```python
from kmc.dedup.planner import DedupPlanner

planner = DedupPlanner()

# Add blocks from multiple files
for block_id, data in enumerate(all_block_data):
    is_dup = planner.add_block(block_id, data)
    if is_dup:
        print(f"Block {block_id} is a duplicate")

plan = planner.create_plan()
print(f"Unique: {plan.unique_blocks}, Deduped: {plan.deduplicated_blocks}")
print(f"Saved: {plan.saved_bytes:,} bytes")
```
