# LoRA Adapter Workflow

This document describes how to use KMC's dedicated LoRA/PEFT adapter workflow for compressing, inspecting, and managing LoRA adapter directories.

## Overview

LoRA (Low-Rank Adaptation) adapters are small weight deltas that modify a base model's behavior without changing the original weights. They are typically distributed as directories containing:

- `adapter_model.safetensors` -- The adapter weights (low-rank matrices)
- `adapter_config.json` -- PEFT configuration (type, rank, target modules, base model)
- `README.md` -- Optional documentation
- Tokenizer files -- Optional, inherited from base model

KMC v0.5 provides dedicated commands for LoRA adapter handling:

- `kmc pack-lora` -- Compress a LoRA adapter directory with automatic metadata extraction
- `kmc inspect --lora` -- Inspect a LoRA adapter directory with adapter-specific details

**Important:** KMC does **not** use pickle, does **not** load weights into memory, and does **not** modify any data. Only metadata is read from safetensors headers and JSON config files. Compression is strictly lossless.

## Packing a LoRA Adapter

### Basic Usage

```bash
# Pack a LoRA adapter directory
kmc pack-lora ./my-lora-adapter ./my-lora-adapter.kmc
```

This command:

1. Detects LoRA adapter files in the directory
2. Reads `adapter_config.json` for PEFT configuration
3. Extracts tensor metadata from `adapter_model.safetensors`
4. Builds artifact metadata (rank, target modules, base model, PEFT type)
5. Packs with tensor-aware mode enabled
6. Records `artifact_type: "lora_adapter"` in the manifest

### Example Output

```
Packing LoRA adapter: ./my-lora-adapter -> ./my-lora-adapter.kmc
  PEFT type: LORA
  Rank: 16
  Target modules: q_proj, v_proj
  Base model: meta-llama/Llama-2-7b-hf
Done in 0.85s -- 51,200,000 -> 28,300,000 bytes (ratio: 55.27%)
```

### With Custom Options

```bash
# Specify codec
kmc pack-lora ./my-lora ./my-lora.kmc --codec zstd

# Specify compression level
kmc pack-lora ./my-lora ./my-lora.kmc -l 9

# Specify block size
kmc pack-lora ./my-lora ./my-lora.kmc -b 524288
```

### What Gets Compressed

All files in the LoRA adapter directory are compressed, including:

| File | Treatment |
|------|-----------|
| `adapter_model.safetensors` | Tensor-aware mode with codec selection |
| `adapter_config.json` | Compressed as regular JSON |
| `README.md` | Compressed as regular text |
| Tokenizer files | Compressed as regular data |
| Any other files | Compressed as regular data |

**No pickle deserialization.** If `pytorch_model.bin` is present (which is unusual for LoRA adapters), it is compressed as raw bytes only.

## Inspecting a LoRA Adapter

### Basic Usage

```bash
# Inspect a LoRA adapter directory
kmc inspect ./my-lora-adapter --lora

# Inspect with JSON output (for scripting)
kmc inspect ./my-lora-adapter --lora --json

# Inspect a packed .kmc archive
kmc inspect ./my-lora-adapter.kmc
```

### Example Output

```
KMC LoRA Adapter Inspection

Path: ./my-lora-adapter
Is LoRA: yes
Has adapter model: yes
Has adapter config: yes
PEFT type: LORA
Rank: 16
Target modules: q_proj, v_proj
Base model: meta-llama/Llama-2-7b-hf
```

### JSON Output

```bash
kmc inspect ./my-lora-adapter --lora --json
```

```json
{
  "artifact_type": "lora_adapter",
  "path": "./my-lora-adapter",
  "is_lora": true,
  "has_adapter_model": true,
  "has_adapter_config": true,
  "has_readme": true,
  "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
  "peft_type": "LORA",
  "lora_rank": 16,
  "target_modules": ["q_proj", "v_proj"],
  "warnings": []
}
```

## LoRA Detection

KMC detects LoRA adapters through multiple signals:

1. **File naming**: `adapter_model.safetensors` is the standard PEFT filename.
2. **Alternative names**: Any `.safetensors` file with "adapter" in the name.
3. **Tensor name patterns**: Tensors named with `lora_A.weight` or `lora_B.weight` suffixes.
4. **Configuration file**: `adapter_config.json` with PEFT metadata.

### Detection Without adapter_config.json

If `adapter_config.json` is missing, KMC can still detect LoRA adapters by examining tensor names in safetensors files. The presence of tensors with `lora_A` or `lora_B` in their names is sufficient for detection. In this case:

- `peft_type` defaults to `"unknown"`
- `lora_rank` is inferred from tensor shapes (the first dimension of `lora_A.weight` or the second dimension of `lora_B.weight`)
- `target_modules` is inferred from tensor name prefixes
- `base_model_name_or_path` defaults to `"unknown"`

### Detection Without adapter_model.safetensors

If `adapter_config.json` is found but no adapter model file, KMC reports:

```
Warning: adapter_config.json found but no adapter model file
```

The `is_lora` flag is still set to `true` because the config file indicates a LoRA adapter, but packing will proceed without tensor-specific metadata.

## Manifest Metadata

When a LoRA adapter is packed with `kmc pack-lora`, the v4 manifest includes:

```json
{
  "version": 4,
  "artifact_type": "lora_adapter",
  "artifact_metadata": {
    "artifact_type": "lora_adapter",
    "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
    "peft_type": "LORA",
    "r": 16,
    "target_modules": ["q_proj", "v_proj"]
  },
  "format_metadata": {
    "safetensors": {
      "is_sharded": false,
      "tensor_count": 48,
      "dtypes": ["BF16"]
    }
  }
}
```

## Verifying and Unpacking

```bash
# Verify the packed LoRA archive
kmc verify ./my-lora-adapter.kmc

# Unpack to a directory
kmc unpack ./my-lora-adapter.kmc ./restored-lora/

# The unpacked files are byte-for-byte identical to the originals
```

## Programmatic Usage

```python
from kmc.workflows.lora import detect_lora_adapter, build_lora_manifest_metadata

# Detect LoRA adapter
adapter_info = detect_lora_adapter("./my-lora-adapter")

if adapter_info.is_lora:
    print(f"PEFT type: {adapter_info.peft_type}")
    print(f"Rank: {adapter_info.lora_rank}")
    print(f"Target modules: {adapter_info.target_modules}")
    print(f"Base model: {adapter_info.base_model_name_or_path}")

    # Build manifest metadata
    metadata = build_lora_manifest_metadata(adapter_info)
    # metadata = {
    #     "artifact_type": "lora_adapter",
    #     "base_model_name_or_path": "...",
    #     "peft_type": "LORA",
    #     "r": 16,
    #     "target_modules": ["q_proj", "v_proj"]
    # }
```

### Kimari CLI Integration

```python
from kmc.integrations.kimari import kimari_pack_lora

result = kimari_pack_lora(
    "./my-lora-adapter",
    "./my-lora-adapter.kmc",
    level=3,
    codec="auto",
)
# result = {
#     "status": "ok",
#     "artifact_type": "lora_adapter",
#     "peft_type": "LORA",
#     "lora_rank": 16,
#     "base_model": "meta-llama/Llama-2-7b-hf",
#     "target_modules": ["q_proj", "v_proj"],
#     "original_size": 51200000,
#     "compressed_size": 28300000,
#     "ratio": 0.5527,
# }
```

## Common Workflows

### Compress and Transfer a LoRA Adapter

```bash
# 1. Pack the adapter
kmc pack-lora ./lora-adapter ./lora-adapter.kmc

# 2. Verify before transfer
kmc verify ./lora-adapter.kmc

# 3. Transfer the .kmc file (it is smaller than the original)

# 4. On the receiving end, unpack
kmc unpack ./lora-adapter.kmc ./lora-adapter/

# 5. Load the adapter with your framework
# (e.g., from peft import PeftModel; model = PeftModel.from_pretrained(base_model, "./lora-adapter/"))
```

### Inspect Multiple Adapters

```bash
# Inspect each adapter in a collection
for dir in ./adapters/*/; do
    echo "=== $dir ==="
    kmc inspect "$dir" --lora
done
```

### Compare Compression Across Adapters

```bash
# Benchmark a LoRA adapter
kmc bench ./my-lora ./my-lora-bench.kmc --compare-codecs
```

## Limitations

1. **No delta compression.** LoRA adapters are compressed using standard or tensor-aware mode. Delta compression relative to a base model is future work.

2. **No rank modification.** KMC does not modify LoRA rank or target modules. It only reads and records the metadata.

3. **No weight loading.** KMC never loads tensor data into memory. It reads only the safetensors header for metadata.

4. **Detection heuristics.** LoRA detection relies on file naming and tensor name patterns. Unconventional adapter formats may not be detected automatically. Use `kmc inspect --lora` to force LoRA inspection mode.

5. **Missing metadata.** If `adapter_config.json` is missing, some fields (PEFT type, base model) will be `"unknown"`. Rank and target modules can be inferred from tensor shapes and names.
