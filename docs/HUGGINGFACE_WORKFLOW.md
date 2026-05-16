# Hugging Face Workflow

This document explains how to use Kimari MicroCompress (KMC) with models downloaded from Hugging Face, including compression, verification, decompression, and benchmarking.

## Overview

KMC is designed to work seamlessly with Hugging Face model directories. When you download a model from Hugging Face, you get a directory containing safetensors files, config files, tokenizer files, and potentially sharded model files. KMC can compress, verify, and inspect all of these.

## Step 1: Download a Model from Hugging Face

```bash
# Using huggingface-cli
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-2-7b

# Or using Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('gpt2', local_dir='./models/gpt2')
"
```

## Step 2: Inspect the Model

Before compressing, inspect the model directory to understand its structure:

```bash
# Basic inspection
kmc inspect ./models/gpt2

# With tensor details
kmc inspect ./models/gpt2 --tensors

# JSON output for scripting
kmc inspect ./models/gpt2 --json
```

Expected output:

```
KMC Model Inspection

Path: ./models/gpt2
Detected type: Hugging Face model folder
Safetensors: yes
Sharded: no
Tensor count: 291
Total tensor bytes: 501.20 MB
Dtypes:
  F32: 291 tensors

Files:
  config.json
  tokenizer.json
  model.safetensors

LoRA/PEFT: no
GGUF: no
```

## Step 3: Compress with Tensor-Aware Mode

The `--tensor-aware` flag aligns block boundaries with tensor boundaries in safetensors files, which can improve compression efficiency and enable future block-level operations:

```bash
# Standard compression
kmc pack ./models/gpt2 ./models/gpt2.kmc

# Tensor-aware compression (recommended)
kmc pack ./models/gpt2 ./models/gpt2.kmc --tensor-aware

# With custom compression level
kmc pack ./models/gpt2 ./models/gpt2.kmc --tensor-aware -l 9

# With a specific codec (v0.4+)
kmc pack ./models/gpt2 ./models/gpt2.kmc --codec byteplane
kmc pack ./models/gpt2 ./models/gpt2.kmc --codec floatplane --tensor-aware

# Automatic codec selection (default, v0.4+)
kmc pack ./models/gpt2 ./models/gpt2.kmc --codec auto --tensor-aware
```

### Choosing a Codec (v0.4+)

The `--codec` flag controls which compression codec is used:

| Codec | Description | Best For |
|-------|-------------|----------|
| `auto` | Try all candidates, pick smallest per block (default) | General use, mixed-dtype models |
| `byteplane` | Byte-plane separation + zstd/zlib | BF16/FP16/FP32 tensor data |
| `floatplane` | Sign/exponent/mantissa separation + zstd/zlib | BF16/FP16/FP32 tensor data |
| `zstd` | Pure zstd compression without transformation | General data, non-float types |
| `zlib` | Pure zlib compression | Fallback, always available |
| `raw` | No compression | Already-compressed data, baselines |

**Recommendation:** Use `--codec auto` (the default) for best results. The automatic selector will try tensor-aware codecs for floating-point data and fall back to zstd/zlib for other data types.

## Step 4: Verify the Archive

Always verify after compression to ensure data integrity:

```bash
kmc verify ./models/gpt2.kmc
```

Expected output:

```
KMC Verification Report

Archive: ./models/gpt2.kmc
Format: kimari-microcompress v2
Files: 3
Blocks: 12
Compressed size: 350.50 MB
Restored size: 501.20 MB
Compression ratio: 69.93%
Integrity: OK
```

## Step 5: Decompress When Needed

When you need the original model files (e.g., for inference), decompress the archive:

```bash
kmc unpack ./models/gpt2.kmc ./models/gpt2-restored
```

The decompressed files will be byte-for-byte identical to the originals, verified by SHA-256 hashes at both the file and block level.

## Step 6: Compare with ZipNN

If you want to compare KMC's compression against ZipNN (IBM Research's AI model compression tool), use the `--compare-zipnn` flag:

```bash
# Install ZipNN first (optional)
pip install zipnn

# Run benchmark with ZipNN comparison
kmc bench ./models/gpt2 ./models/gpt2-bench.kmc --compare-zipnn
```

If ZipNN is not installed, the benchmark will show:

```
--- ZipNN Comparison ---
  ZipNN: not available
  Suggestion: pip install zipnn
```

If ZipNN is installed, you'll see a side-by-side comparison:

```
--- ZipNN Comparison ---
  ZipNN version: 0.3.0
  ZipNN compressed: 320,000,000 bytes
  ZipNN ratio: 63.85%
  ZipNN compress time: 8.500s
  ZipNN decompress time: 2.100s
  Note: ZipNN ratio (63.85%) is better than KMC (69.93%)
  Disclaimer: This is a measurement, not a claim of superiority.
```

**Important**: The comparison is a measurement, not a claim that KMC is superior or inferior to ZipNN. Results depend on model type, data format, compression level, and hardware.

## Step 7: Compare Codecs (v0.4+)

The `--compare-codecs` flag benchmarks all available codecs on the same data:

```bash
# Compare all codecs
kmc bench ./models/gpt2 ./models/gpt2-bench.kmc --compare-codecs

# Compare codecs with JSON output
kmc bench ./models/gpt2 ./models/gpt2-bench.kmc --compare-codecs --json --output codec-comparison.json
```

You can also use the dedicated benchmark script for more detailed codec comparison:

```bash
python scripts/bench_small_hf_model.py ./models/gpt2
```

See [REAL_MODEL_BENCHMARK.md](REAL_MODEL_BENCHMARK.md) for details.

## Working with Sharded Models

Large models are often split into multiple shard files:

```
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
```

KMC handles sharded models automatically:

```bash
# Inspect will detect shards
kmc inspect ./models/llama-2-7b

# Pack compresses all shards
kmc pack ./models/llama-2-7b ./models/llama-2-7b.kmc --tensor-aware

# Unpack restores all shards
kmc unpack ./models/llama-2-7b.kmc ./models/llama-2-7b-restored
```

## Working with LoRA Adapters

KMC detects and handles LoRA/PEFT adapters:

```bash
# Inspect a LoRA adapter directory
kmc inspect ./models/my-lora-adapter --tensors

# Compress the adapter
kmc pack ./models/my-lora-adapter ./models/my-lora-adapter.kmc --tensor-aware
```

The inspector will show adapter-specific information:

```
Detected type: PEFT/LoRA adapter
LoRA/PEFT: yes
  Rank: 16
  Target modules: q_proj, v_proj
  Base model: meta-llama/Llama-2-7b-hf
```

## Complete Example Workflow

```bash
# 1. Download model
huggingface-cli download gpt2 --local-dir ./models/gpt2

# 2. Inspect
kmc inspect ./models/gpt2 --tensors

# 3. Compress with tensor-aware mode
kmc pack ./models/gpt2 ./models/gpt2.kmc --tensor-aware

# 4. Verify
kmc verify ./models/gpt2.kmc

# 5. Inspect compression summary (v0.4+)
kmc inspect ./models/gpt2.kmc --compression

# 6. Benchmark with codec comparison
kmc bench ./models/gpt2 ./models/gpt2-bench.kmc --compare-codecs

# 7. Benchmark with ZipNN comparison
kmc bench ./models/gpt2 ./models/gpt2-bench.kmc --compare-zipnn --json --output reports/gpt2-bench.json

# 8. When needed, decompress
kmc unpack ./models/gpt2.kmc ./models/gpt2-restored
```

## What KMC Does NOT Do

It is important to understand the limitations of KMC when working with Hugging Face models:

### KMC does NOT reduce VRAM during inference

KMC compresses files for storage and transfer. To use a model, you must decompress it first. The model will consume the same amount of VRAM whether it was previously compressed or not. If you need smaller VRAM usage, use quantization techniques (GGUF Q4_K, GPTQ, AWQ, etc.). This is true even with tensor-aware codecs like BytePlane and FloatPlane — they improve storage compression ratio, not runtime memory usage.

### KMC does NOT guarantee improved inference speed

KMC is not designed to speed up model loading or inference. While faster transfer (due to smaller file size) may reduce download time, the decompression step adds overhead before the model can be loaded.

### KMC does NOT improve model quality

KMC is a lossless compression tool. It preserves every byte exactly. It cannot and does not modify model weights, alter model behavior, or improve model output quality in any way.

### KMC is NOT a replacement for quantization

Quantization reduces model precision (e.g., FP32 to INT4) to shrink model size at the cost of some accuracy. KMC is lossless — it compresses without any quality loss. They serve different purposes and can be used together: quantize your model first, then use KMC to compress the quantized files for storage.

### KMC does NOT promise fixed compression ratios

Compression ratios vary significantly based on:
- Model format (safetensors vs GGUF vs PyTorch)
- Data types (FP32, FP16, BF16, INT8, INT4)
- Model architecture and weight distribution
- Compression level used

Do not assume specific ratios without benchmarking your particular model.

## Publishing Benchmark Results

If you publish benchmark results comparing KMC with other tools, please follow these guidelines:

1. **Always include the environment**: Python version, OS, CPU, RAM, and tool versions.
2. **Mark synthetic vs real data**: Synthetic benchmarks produce misleadingly high ratios.
3. **Report all metrics**: Compression ratio, pack time, unpack time, and throughput.
4. **Do not cherry-pick**: Report results for multiple model types and sizes.
5. **Be honest about limitations**: If KMC performs worse than another tool on a specific workload, say so.
6. **Make results reproducible**: Provide the exact commands and data used.

The `kmc bench --json --compare-zipnn` command outputs all the information needed for a fair comparison.

The `kmc bench --compare-codecs` command outputs codec comparison results including which codec was actually selected per block in `auto` mode.
