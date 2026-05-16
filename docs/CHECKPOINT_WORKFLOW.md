# Checkpoint Workflow

This document describes how to use KMC's dedicated training checkpoint workflow for compressing, inspecting, and managing Hugging Face training checkpoint directories.

## Overview

Training checkpoints are snapshots of model state during training, typically saved by the Hugging Face Trainer. They contain model weights, optimizer states, scheduler states, and various training metadata. KMC v0.5 provides dedicated commands for checkpoint handling:

- `kmc pack-checkpoint` -- Compress a training checkpoint directory with automatic metadata extraction
- `kmc inspect --checkpoint` -- Inspect a training checkpoint directory with checkpoint-specific details

**Important safety rules:**
- KMC **never** uses pickle to inspect or load checkpoint data.
- Pickle-based files (`optimizer.pt`, `training_args.bin`, `pytorch_model.bin`, etc.) are detected by name and compressed as raw bytes only.
- Their contents are **never** deserialized. Only their presence, size, and hash are recorded.
- Compression is strictly lossless and reversible.

## Checkpoint Directory Structure

A typical Hugging Face training checkpoint directory looks like:

```
checkpoint-1000/
  config.json
  generation_config.json
  model.safetensors              (or pytorch_model.bin)
  optimizer.pt                   (pickle-based, NOT loaded by KMC)
  scheduler.pt                   (pickle-based, NOT loaded by KMC)
  scaler.pt                      (pickle-based, NOT loaded by KMC)
  trainer_state.json
  training_args.bin              (pickle-based, NOT loaded by KMC)
  rng_state.pth                  (pickle-based, NOT loaded by KMC)
  global_step.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.json
  merges.txt
  tokenizer.model
  README.md
```

## Packing a Training Checkpoint

### Basic Usage

```bash
# Pack a training checkpoint directory
kmc pack-checkpoint ./checkpoint-1000 ./checkpoint-1000.kmc
```

This command:

1. Detects checkpoint files in the directory
2. Infers the training step from directory name or `global_step.json`
3. Identifies component types (optimizer, scheduler, RNG, etc.)
4. Emits warnings for pickle-based files
5. Enables tensor-aware mode if `model.safetensors` is present
6. Builds artifact metadata (step, component flags)
7. Records `artifact_type: "training_checkpoint"` in the manifest

### Example Output

```
Warning: pytorch_model.bin detected (pickle-based). Only size/hash will be recorded; contents will NOT be deserialized.
Warning: optimizer.pt detected (pickle-based). Only size/hash will be recorded; contents will NOT be deserialized.
Packing training checkpoint: ./checkpoint-1000 -> ./checkpoint-1000.kmc
  Step: 1000
  Has optimizer state: True
  Has trainer state: True
  Has safetensors model: False
Done in 2.30s -- 1,500,000,000 -> 980,000,000 bytes (ratio: 65.33%)
```

### With Custom Options

```bash
# Specify codec
kmc pack-checkpoint ./checkpoint-1000 ./checkpoint-1000.kmc --codec zstd

# Specify compression level
kmc pack-checkpoint ./checkpoint-1000 ./checkpoint-1000.kmc -l 9

# Specify block size
kmc pack-checkpoint ./checkpoint-1000 ./checkpoint-1000.kmc -b 524288
```

### What Gets Compressed

All files in the checkpoint directory are compressed:

| File | Type | Treatment |
|------|------|-----------|
| `model.safetensors` | Safe | Tensor-aware mode with codec selection |
| `pytorch_model.bin` | Pickle | Compressed as raw bytes; never deserialized |
| `optimizer.pt` | Pickle | Compressed as raw bytes; never deserialized |
| `optimizer_state.pt` | Pickle | Compressed as raw bytes; never deserialized |
| `scheduler.pt` | Pickle | Compressed as raw bytes; never deserialized |
| `scaler.pt` | Pickle | Compressed as raw bytes; never deserialized |
| `rng_state.pth` | Pickle | Compressed as raw bytes; never deserialized |
| `rng_state_0.pth` | Pickle | Compressed as raw bytes; never deserialized |
| `training_args.bin` | Pickle | Compressed as raw bytes; never deserialized |
| `trainer_state.json` | JSON | Compressed as regular data |
| `global_step.json` | JSON | Compressed as regular data |
| `config.json` | JSON | Compressed as regular data |
| Tokenizer files | Various | Compressed as regular data |
| Other files | Various | Compressed as regular data |

### Pickle Safety

KMC identifies the following files as pickle-based and **never deserializes** them:

- `training_args.bin`
- `optimizer.pt`
- `optimizer_state.pt`
- `scheduler.pt`
- `scaler.pt`
- `rng_state.pth`
- `rng_state_0.pth`
- `pytorch_model.bin`

When any of these files are detected, KMC emits a warning:

```
Warning: optimizer.pt detected (pickle-based). Only size/hash will be recorded;
contents will NOT be deserialized.
```

These files are still compressed and included in the archive. On unpack, they are restored byte-for-byte identical. The only difference is that KMC does not attempt to read their internal structure.

### Tensor-Aware Mode

If `model.safetensors` is present in the checkpoint, `kmc pack-checkpoint` automatically enables tensor-aware mode for that file. This provides:

- Block boundaries aligned to tensor boundaries
- Per-tensor metadata in the manifest
- dtype-aware codec selection (FloatPlane, BytePlane for floating-point tensors)

If only `pytorch_model.bin` is present (no safetensors), tensor-aware mode is not used because KMC does not deserialize pickle files.

## Inspecting a Training Checkpoint

### Basic Usage

```bash
# Inspect a checkpoint directory
kmc inspect ./checkpoint-1000 --checkpoint

# Inspect with JSON output
kmc inspect ./checkpoint-1000 --checkpoint --json

# Inspect a packed .kmc archive
kmc inspect ./checkpoint-1000.kmc
```

### Example Output

```
KMC Training Checkpoint Inspection

Path: ./checkpoint-1000
Is checkpoint: yes
Step: 1000
Has trainer state: yes
Has optimizer state: yes
Has scheduler state: yes
Has RNG state: yes
Has safetensors model: no
Has pytorch model: yes

Detected files:
  config.json (config)
  global_step.json (global_step)
  optimizer.pt (optimizer)
  pytorch_model.bin (pytorch_model)
  rng_state.pth (rng_state)
  scheduler.pt (scheduler)
  trainer_state.json (trainer_state)
  training_args.bin (training_args)

Warnings:
  pytorch_model.bin detected (pickle-based). Only size/hash will be recorded; contents will NOT be deserialized.
  optimizer.pt detected (pickle-based). Only size/hash will be recorded; contents will NOT be deserialized.
```

### JSON Output

```bash
kmc inspect ./checkpoint-1000 --checkpoint --json
```

```json
{
  "artifact_type": "training_checkpoint",
  "path": "./checkpoint-1000",
  "is_checkpoint": true,
  "step": 1000,
  "has_trainer_state": true,
  "has_optimizer_state": true,
  "has_scheduler_state": true,
  "has_rng_state": true,
  "has_safetensors_model": false,
  "has_pytorch_model": true,
  "detected_files": {
    "config.json": "config",
    "global_step.json": "global_step",
    "optimizer.pt": "optimizer",
    "pytorch_model.bin": "pytorch_model",
    "rng_state.pth": "rng_state",
    "scheduler.pt": "scheduler",
    "trainer_state.json": "trainer_state",
    "training_args.bin": "training_args"
  },
  "warnings": [
    "pytorch_model.bin detected (pickle-based). Only size/hash will be recorded; contents will NOT be deserialized.",
    "optimizer.pt detected (pickle-based). Only size/hash will be recorded; contents will NOT be deserialized."
  ]
}
```

## Step Detection

KMC infers the training step from:

1. **Directory name**: If the directory is named `checkpoint-1000`, the step is `1000`.
2. **`global_step.json`**: If present, reads `{"global_step": 1000}`.
3. **`trainer_state.json`**: If present, reads the `global_step` field from the trainer state.

If none of these sources are available, the step is reported as `null`.

## Checkpoint Detection

A directory is classified as a training checkpoint if it contains at least one of:

- `trainer_state.json`
- `model.safetensors`
- `pytorch_model.bin`
- `optimizer.pt`
- Any other recognized checkpoint file pattern

KMC also detects safetensors shard files (`model-00001-of-00003.safetensors`) and classifies them as checkpoint model files.

## Manifest Metadata

When a checkpoint is packed with `kmc pack-checkpoint`, the v4 manifest includes:

```json
{
  "version": 4,
  "artifact_type": "training_checkpoint",
  "artifact_metadata": {
    "artifact_type": "training_checkpoint",
    "step": 1000,
    "has_optimizer_state": true,
    "has_scheduler_state": true,
    "has_rng_state": true,
    "has_trainer_state": true
  }
}
```

## Verifying and Unpacking

```bash
# Verify the packed checkpoint archive
kmc verify ./checkpoint-1000.kmc

# Unpack to a directory
kmc unpack ./checkpoint-1000.kmc ./restored-checkpoint/

# The unpacked files are byte-for-byte identical to the originals
# Pickle-based files are restored as-is (no deserialization occurred during packing)
```

## Programmatic Usage

```python
from kmc.workflows.checkpoint import detect_checkpoint, build_checkpoint_manifest_metadata

# Detect checkpoint
ckpt_info = detect_checkpoint("./checkpoint-1000")

if ckpt_info.is_checkpoint:
    print(f"Step: {ckpt_info.step}")
    print(f"Has optimizer: {ckpt_info.has_optimizer_state}")
    print(f"Has safetensors model: {ckpt_info.has_safetensors_model}")
    print(f"Detected files: {ckpt_info.detected_files}")

    if ckpt_info.warnings:
        for w in ckpt_info.warnings:
            print(f"Warning: {w}")

    # Build manifest metadata
    metadata = build_checkpoint_manifest_metadata(ckpt_info)
    # metadata = {
    #     "artifact_type": "training_checkpoint",
    #     "step": 1000,
    #     "has_optimizer_state": True,
    #     "has_scheduler_state": True,
    #     "has_rng_state": True,
    #     "has_trainer_state": True,
    # }
```

### Kimari CLI Integration

```python
from kmc.integrations.kimari import kimari_pack_checkpoint

result = kimari_pack_checkpoint(
    "./checkpoint-1000",
    "./checkpoint-1000.kmc",
    level=3,
    codec="auto",
)
# result = {
#     "status": "ok",
#     "artifact_type": "training_checkpoint",
#     "step": 1000,
#     "has_optimizer_state": True,
#     "has_trainer_state": True,
#     "has_safetensors_model": False,
#     "original_size": 1500000000,
#     "compressed_size": 980000000,
#     "ratio": 0.6533,
# }
```

## Common Workflows

### Compress Old Checkpoints to Save Space

```bash
# Compress all checkpoints in a training run
for ckpt in ./results/checkpoint-*/; do
    name=$(basename "$ckpt")
    kmc pack-checkpoint "$ckpt" "./compressed/${name}.kmc"
    kmc verify "./compressed/${name}.kmc"
done

# Original checkpoints can now be removed if desired
# They can be restored byte-for-byte from the .kmc archives
```

### Archive a Complete Training Run

```bash
# Pack the entire results directory (includes all checkpoints)
kmc pack ./results ./results.kmc --tensor-aware

# Or pack individual checkpoints for selective restoration
kmc pack-checkpoint ./results/checkpoint-1000 ./ckpt-1000.kmc
kmc pack-checkpoint ./results/checkpoint-2000 ./ckpt-2000.kmc
kmc pack-checkpoint ./results/checkpoint-final ./ckpt-final.kmc
```

### Restore and Resume Training

```bash
# Unpack a checkpoint
kmc unpack ./ckpt-1000.kmc ./restored-checkpoint/

# The restored directory is byte-for-byte identical to the original
# Resume training with the restored checkpoint
# (The optimizer state is preserved exactly, including RNG state)
```

## Limitations

1. **No optimizer state inspection.** Optimizer states (optimizer.pt) are pickle-based and are never deserialized. KMC cannot report optimizer type, learning rate, or other training hyperparameters from these files.

2. **No delta compression for checkpoints.** Checkpoints are compressed using standard or tensor-aware mode. Delta compression between checkpoints (e.g., storing only the differences between checkpoint-1000 and checkpoint-2000) is future work.

3. **No incremental checkpoint updates.** Each checkpoint must be packed independently. There is no way to update an existing archive with new checkpoint data.

4. **Pickle-based model files are opaque.** If a checkpoint uses `pytorch_model.bin` instead of `model.safetensors`, KMC cannot extract tensor metadata or apply tensor-aware codecs to the model weights.

5. **Detection heuristics.** Checkpoint detection relies on known file patterns. Unconventional checkpoint formats may not be detected automatically. Use `kmc inspect --checkpoint` to force checkpoint inspection mode.
