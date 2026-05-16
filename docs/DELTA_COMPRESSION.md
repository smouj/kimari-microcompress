# Experimental Delta Compression

> **Status**: Experimental (KMC v0.8.0-alpha)
> **Manifest version**: 7 (format_version field)
> **CLI flag**: `--delta-base <path>`

## Overview

Delta compression is an experimental feature in KMC v0.8.0-alpha that enables incremental archive creation by comparing new data against a **base archive**. When a new archive is created with `--delta-base`, blocks that are identical to blocks in the base archive are not stored — instead, a lightweight reference to the base archive's block is recorded in the manifest. Only blocks that have actually changed are written to the delta archive's data stream.

This is primarily designed for **training checkpoint chains** and **incremental model updates**, where successive versions of a model share the vast majority of their data. For example, two consecutive training checkpoints may differ in only 5–10% of their tensor data, making delta compression extremely effective: the delta archive stores only the changed blocks, yielding a file that is a fraction of the full archive's size.

Delta compression in KMC v0.8 is intentionally simple and conservative. It performs **block-level whole-match comparison** — not binary diffing. A block is either stored in full or referenced entirely; there is no xdelta-style patch encoding. This design prioritizes correctness and simplicity over maximum compression ratio.

## How It Works

### Block-Level Comparison

The delta compression pipeline compares blocks between the new (target) data and the base archive at the block level:

1. **Base archive is loaded**: The `DeltaPlanner` reads the base `.kmc` archive's manifest and, where available, the original block data.
2. **Each new block is compared**: For every block in the new data, the planner checks whether it matches any block in the base archive by comparing the original (uncompressed) data byte-for-byte.
3. **Matched blocks are referenced**: If a new block's data exactly matches a base block, a `DeltaBlock` entry is created with `is_changed=False`, and the reference to the base block is recorded (base file path, base block index, base block ID).
4. **Changed blocks are stored**: If no match is found, the block is marked as `is_changed=True` and will be written to the delta archive normally (compressed with the selected codec).

### DeltaCodec and DeltaPlanner

The delta system consists of two main components:

**`DeltaCodec`** (`kmc.delta.delta_codec`): Performs the actual block-by-block comparison. It maintains a list of `DeltaBlock` entries, each recording whether a block is changed or referenced. The codec provides properties for counting changed vs. referenced blocks.

**`DeltaPlanner`** (`kmc.delta.delta_planner`): Orchestrates the planning process. It takes the base archive path, reads the base manifest, and provides two methods for block comparison:

- `add_block(global_block_id, original_data, file_path, block_index)` — Compares against the base manifest's file and block structure (size-based quick check).
- `add_block_with_base_data(global_block_id, original_data, base_original_data, ...)` — Performs exact byte-for-byte comparison when the base block's original data is available. This is the preferred method for accurate delta detection.

### DeltaPlan Structure

The `DeltaPlan` dataclass records the complete delta plan:

| Field | Type | Description |
|---|---|---|
| `enabled` | `bool` | Whether delta compression is active |
| `base_archive_path` | `str` | Path to the base `.kmc` archive |
| `base_archive_sha256` | `str` | SHA-256 hash of the base archive file |
| `delta_blocks` | `list[DeltaBlock]` | Per-block delta status entries |
| `changed_block_ids` | `set[int]` | Block IDs that must be stored |
| `referenced_block_ids` | `set[int]` | Block IDs that reference the base |
| `total_blocks` | `int` | Total blocks in the new archive |
| `changed_blocks` | `int` | Count of changed blocks |
| `referenced_blocks` | `int` | Count of blocks referenced from base |

The plan's `should_store_block(block_id)` method returns `True` for changed blocks (which must be written) and `False` for referenced blocks (which are reconstructed from the base during unpack).

## When to Use Delta Compression

### Training Checkpoint Chains

The primary use case for delta compression is storing chains of training checkpoints. When a model is trained for thousands of steps, each checkpoint snapshot contains:

- **Model weights** that change incrementally between steps
- **Optimizer state** (Adam moments, etc.) that also changes incrementally
- **Trainer/scheduler state** that is typically small and may be identical between steps
- **Configuration files** that are usually identical across all checkpoints

For checkpoints that are close together in training steps (e.g., `checkpoint-1000` and `checkpoint-1100`), 90–95% of the tensor data may be identical. Delta compression captures this redundancy efficiently.

### Incremental Model Updates

When fine-tuning a model and saving intermediate snapshots, the base model weights remain largely unchanged while the LoRA or adapter weights evolve. A delta archive against the base model stores only the changed adapter weights.

### Multi-Version Model Storage

For organizations that maintain multiple versions of the same model (e.g., a model updated with new data or a model at different quantization levels that shares some metadata), delta compression can significantly reduce storage requirements.

## CLI Usage

### Creating a Delta Archive

Use the `--delta-base` flag to specify the base archive:

```bash
# First, create the base archive from the initial checkpoint
kmc pack ./checkpoint-1000/ ./checkpoint-1000.kmc --tensor-aware

# Then, create a delta archive against the base
kmc pack ./checkpoint-1100/ ./checkpoint-1100-delta.kmc \
    --tensor-aware \
    --delta-base ./checkpoint-1000.kmc
```

The delta archive will only contain blocks that differ from the base. If 95% of blocks are unchanged, the delta archive will be roughly 5% the size of a full archive (plus manifest overhead).

### Combining with Other Features

```bash
# Delta + dedup (dedup removes internal duplicates, delta removes cross-archive duplicates)
kmc pack ./checkpoint-1100/ ./checkpoint-1100-delta.kmc \
    --tensor-aware \
    --delta-base ./checkpoint-1000.kmc \
    --dedup

# Delta with GGUF-aware mode
kmc pack ./gguf-model-v2/ ./model-v2-delta.kmc \
    --gguf-aware \
    --delta-base ./model-v1.kmc

# Delta with custom codec and block size
kmc pack ./checkpoint-1100/ ./checkpoint-1100-delta.kmc \
    --tensor-aware \
    --delta-base ./checkpoint-1000.kmc \
    --codec zstd \
    --block-size 524288
```

## Manifest Fields

### Top-Level `delta` Field

The `KMCManifest.delta` dictionary records delta compression metadata:

```json
{
  "delta": {
    "enabled": true,
    "base_archive_sha256": "a3f2b8c1d4e5f6...7890abcdef1234567890abcdef1234567890abcdef12",
    "base_archive_path_hint": "checkpoint-1000.kmc",
    "mode": "experimental",
    "total_blocks": 260,
    "changed_blocks": 14,
    "referenced_blocks": 246
  }
}
```

| Key | Description |
|---|---|
| `enabled` | Whether delta compression was active |
| `base_archive_sha256` | SHA-256 hash of the base archive file (for integrity verification) |
| `base_archive_path_hint` | Original path to the base archive (informational only; may not exist at unpack time) |
| `mode` | Always `"experimental"` in v0.8 |
| `total_blocks` | Total blocks in the delta archive |
| `changed_blocks` | Blocks that were stored in the delta archive |
| `referenced_blocks` | Blocks that reference the base archive |

## Unpack Requirements

### Base Archive Required

**Unpacking a delta archive requires the base archive to be present.** The unpacker reads referenced blocks from the base archive and combines them with the changed blocks stored in the delta archive to reconstruct the full data.

The unpacker uses the `base_archive_sha256` field to verify that the correct base archive is being used. If the SHA-256 hash of the provided base archive does not match the recorded hash, the unpacker will raise an error.

```bash
# Unpack requires both the delta archive and the base archive
kmc unpack ./checkpoint-1100-delta.kmc ./restored-checkpoint-1100/ \
    --delta-base ./checkpoint-1000.kmc
```

### Verification with Delta Archives

The `verify` command checks both the delta archive's stored blocks and cross-references to the base archive. If the base archive is not available during verification, referenced blocks cannot be verified and a warning is issued:

```bash
# Verify with base archive present (full verification)
kmc verify ./checkpoint-1100-delta.kmc --delta-base ./checkpoint-1000.kmc

# Verify without base archive (partial verification — referenced blocks skipped)
kmc verify ./checkpoint-1100-delta.kmc
```

### KMCReader with Delta Archives

The `KMCReader` API can read delta archives, but blocks referenced from the base archive require the base archive to be accessible. If the base archive is not found, reading a referenced block raises a `FileNotFoundError`:

```python
from kmc.reader import KMCReader

# Open the delta archive — base archive must be co-located or specified
with KMCReader("checkpoint-1100-delta.kmc") as reader:
    manifest = reader.get_manifest()
    if manifest.delta.get("enabled"):
        print(f"Delta archive: {manifest.delta['changed_blocks']} changed blocks")
        print(f"Base archive: {manifest.delta['base_archive_path_hint']}")
    # Reading a file may require the base archive
    config = reader.read_file("config.json")
```

## Limitations

### Block-Level Only (No Binary Diffing)

KMC v0.8 delta compression operates at the **whole-block level**. A block is either stored in its entirety or referenced entirely. There is no xdelta-style binary diffing that computes minimal patches for partially changed blocks. This means:

- A block that changes by even a single byte must be stored in full.
- The compression ratio is bounded by the block size: smaller blocks (e.g., 64 KB) produce finer-grained deltas but increase manifest overhead; larger blocks (e.g., 1 MB) reduce overhead but may store unchanged data within a block that contains some changes.

For future versions, xdelta-style binary diffing or rsync-style rolling hash matching are under consideration, but they are not implemented in v0.8.

### No xdelta / No Rolling Hash

The current implementation does not use xdelta, bsdiff, or any rolling-hash algorithm for block alignment. Block boundaries are determined solely by the fixed `block_size` parameter during packing. If the base and target archives were packed with different block sizes, delta detection will fail for most blocks even if the underlying data is largely identical.

**Recommendation**: Always use the same `block_size` when creating base and delta archives for the same data.

### Experimental — API May Change

The delta compression feature is marked as experimental. The manifest format, CLI flags, and Python API may change in future KMC versions. Archives created with `--delta-base` in v0.8 may not be compatible with future versions without migration.

### Base Archive Must Be Available

Delta archives are not self-contained. They cannot be unpacked or fully verified without access to the base archive. This creates a dependency chain: if the base archive is lost or corrupted, the delta archive becomes unusable for blocks that reference it.

### Single Base Archive Only

In v0.8, a delta archive can reference only one base archive. Chained deltas (delta-of-delta) are not supported. Each delta archive must reference a full (non-delta) base archive.

## Examples

### Example 1: Training Checkpoint Chain

```bash
# Create base archive from the initial checkpoint
kmc pack ./checkpoints/step-1000/ ./ckpt-1000.kmc --tensor-aware
# Output: Done in 12.3s — 13,824,000,000 -> 5,248,000,000 bytes (ratio: 37.99%)

# Create delta archive from the next checkpoint
kmc pack ./checkpoints/step-1100/ ./ckpt-1100-delta.kmc \
    --tensor-aware \
    --delta-base ./ckpt-1000.kmc
# Output: Done in 8.1s — 13,824,000,000 -> 812,000,000 bytes (ratio: 5.87%)

# Inspect the delta archive
kmc inspect ./ckpt-1100-delta.kmc --delta
# Output:
# Delta:
#   Enabled: yes
#   Mode: experimental
#   Base archive: ckpt-1000.kmc
#   Changed blocks: 14
#   Referenced blocks: 246

# Unpack requires the base archive
kmc unpack ./ckpt-1100-delta.kmc ./restored-step-1100/ \
    --delta-base ./ckpt-1000.kmc
```

### Example 2: Incremental LoRA Update

```bash
# Archive the base LoRA adapter
kmc pack-lora ./lora-adapter-v1/ ./lora-v1.kmc

# Create a delta for the updated adapter
kmc pack ./lora-adapter-v2/ ./lora-v2-delta.kmc \
    --tensor-aware \
    --delta-base ./lora-v1.kmc
```

### Example 3: Programmatic Delta Planning

```python
from kmc.delta.delta_planner import DeltaPlanner
from kmc.delta.delta_codec import DeltaBlock

# Initialize planner with base archive
planner = DeltaPlanner(base_archive_path="checkpoint-1000.kmc")

# Add blocks from the new data with base data comparison
for block_id, (new_data, base_data) in enumerate(zip(new_blocks, base_blocks)):
    delta_block = planner.add_block_with_base_data(
        global_block_id=block_id,
        original_data=new_data,
        base_original_data=base_data,
        base_file_path="model.safetensors",
        base_block_index=block_id,
    )
    if delta_block.is_changed:
        print(f"Block {block_id}: CHANGED")
    else:
        print(f"Block {block_id}: referenced from base")

# Create the delta plan
plan = planner.create_plan()
print(f"Total: {plan.total_blocks}, Changed: {plan.changed_blocks}, "
      f"Referenced: {plan.referenced_blocks}")
print(f"Space savings: {plan.referenced_blocks / plan.total_blocks:.1%}")
```

### Example 4: Delta Archive Manifest (JSON)

```json
{
  "version": 7,
  "tool": "kimari-microcompress",
  "tool_version": "0.8.0-alpha",
  "delta": {
    "enabled": true,
    "base_archive_sha256": "e3b0c44298fc1c149afbf4c8996fb924...cfb3c",
    "base_archive_path_hint": "/models/checkpoint-1000.kmc",
    "mode": "experimental",
    "total_blocks": 260,
    "changed_blocks": 14,
    "referenced_blocks": 246
  },
  "files": [
    {
      "path": "model.safetensors",
      "blocks": [
        {"index": 0, "dedup_ref": -1, "codec": "zstd", "compressed_size": 0, "original_size": 262144},
        {"index": 1, "dedup_ref": 1, "codec": "referenced", "compressed_size": 0, "original_size": 262144}
      ]
    }
  ]
}
```
