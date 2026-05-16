# Benchmark Plan

## Goals

1. Measure compression ratio, speed, and memory usage for real AI model files.
2. Compare KMC against general-purpose compression tools (tar+zstd, zip).
3. Compare against ZipNN where applicable.
4. Provide reproducible benchmark results that inform optimization priorities.
5. Never invent or fabricate benchmark results.
6. Clearly distinguish between synthetic and real data benchmarks.

## KMC v0.5.0-alpha Benchmark Capabilities

### Core Benchmarking

KMC includes a built-in benchmark system accessible via `kmc bench`:

```bash
# Basic benchmark
kmc bench ./model ./model-bench.kmc

# With JSON output
kmc bench ./model ./model-bench.kmc --json --output report.json

# With tensor-aware mode
kmc bench ./model ./model-bench.kmc --tensor-aware

# Mark data as synthetic
kmc bench ./model ./model-bench.kmc --synthetic

# With a specific codec
kmc bench ./model ./model-bench.kmc --codec byteplane
kmc bench ./model ./model-bench.kmc --codec floatplane
```

### Codec Comparison Benchmarks (v0.4+)

The `--compare-codecs` flag runs the benchmark with all available codecs and produces a side-by-side comparison:

```bash
# Compare all codecs on the same data
kmc bench ./model ./model-bench.kmc --compare-codecs

# Compare codecs with JSON output
kmc bench ./model ./model-bench.kmc --compare-codecs --json --output codec-comparison.json
```

When `--compare-codecs` is used, the benchmark tests each available codec (`auto`, `byteplane`, `floatplane`, `zstd`, `zlib`, `raw`) on the same input and reports compressed size, ratio, and timing for each.

**No claims of superiority are made.** The results are measurements, not marketing. Different models and dtypes may favor different codecs.

### GGUF-Aware Benchmarks (v0.5+)

The `--gguf-aware` flag on `kmc pack` adjusts codec selection for quantized GGUF tensors:

```bash
# Pack a GGUF file with GGUF-aware mode
kmc pack ./model.gguf ./model.kmc --gguf-aware

# Compare with non-GGUF-aware mode
kmc pack ./model.gguf ./model-default.kmc

# Compare sizes and compression summary
kmc inspect ./model.kmc --compression
kmc inspect ./model-default.kmc --compression
```

For benchmarking, compare the GGUF-aware and non-GGUF-aware results on the same GGUF file:

```bash
# Benchmark with default mode
kmc bench ./model.gguf ./bench-default.kmc --json --output gguf-default.json

# Manually pack with GGUF-aware mode and measure
time kmc pack ./model.gguf ./bench-gguf-aware.kmc --gguf-aware
ls -la ./bench-default.kmc ./bench-gguf-aware.kmc
```

### LoRA Adapter Benchmarks (v0.5+)

Use `kmc pack-lora` for dedicated LoRA adapter benchmarking:

```bash
# Pack a LoRA adapter
kmc pack-lora ./lora-adapter ./lora-adapter.kmc

# Compare with standard pack
kmc pack ./lora-adapter ./lora-standard.kmc --tensor-aware

# Compare sizes
ls -la ./lora-adapter.kmc ./lora-standard.kmc
```

### Checkpoint Benchmarks (v0.5+)

Use `kmc pack-checkpoint` for dedicated checkpoint benchmarking:

```bash
# Pack a training checkpoint
kmc pack-checkpoint ./checkpoint-1000 ./checkpoint-1000.kmc

# Compare with standard pack
kmc pack ./checkpoint-1000 ./checkpoint-standard.kmc --tensor-aware

# Compare sizes
ls -la ./checkpoint-1000.kmc ./checkpoint-standard.kmc
```

### Synthetic Tensor Fixtures (v0.4+)

For reproducible codec comparison benchmarks, KMC uses synthetic tensor fixtures with known properties:

| Fixture | Dtype | Shape | Size | Description |
|---------|-------|-------|------|-------------|
| BF16 weights | BF16 | [256, 256] | 128 KB | Simulated weight matrix |
| FP16 weights | FP16 | [256, 256] | 128 KB | Simulated weight matrix |
| FP32 weights | FP32 | [256, 256] | 256 KB | Simulated weight matrix |
| Mixed BF16 model | BF16 | Various | ~1 MB | Multiple tensors with different shapes |
| Random bytes | N/A | N/A | 256 KB | Incompressible baseline |

These fixtures are generated programmatically and marked with `synthetic: true` in JSON output. They are useful for validating codec behavior but **do not represent real-world compression ratios**.

### Real Model Benchmark Script

The `scripts/bench_small_hf_model.py` script provides a more realistic benchmark using actual HuggingFace models. See [REAL_MODEL_BENCHMARK.md](REAL_MODEL_BENCHMARK.md) for details.

## ZipNN Comparison

```bash
# Compare with ZipNN (if installed)
kmc bench ./model ./model-bench.kmc --compare-zipnn
```

When ZipNN is not installed, the benchmark reports:
```
ZipNN: not available
Suggestion: pip install zipnn
```

When ZipNN is installed, the benchmark measures and reports both KMC and ZipNN results side by side. **No claims of superiority are made.** The results are measurements, not marketing.

### Environment Metadata

All benchmark results include environment information for reproducibility:

```json
{
  "environment": {
    "python_version": "3.12.1",
    "os_name": "Linux",
    "os_version": "6.1.0",
    "cpu": "AMD Ryzen 9 7950X",
    "ram_gb": 64.0,
    "kmc_version": "0.5.0-alpha",
    "zipnn_version": "0.3.0",
    "zstd_available": true
  }
}
```

## Benchmark Targets

### Small Models (< 1 GB)

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| GPT-2 (124M) | safetensors | ~500 MB | Widely available, good baseline |
| DistilGPT-2 | safetensors | ~350 MB | Smaller variant |
| BERT-base | safetensors | ~440 MB | Encoder model variety |

### GGUF Models (v0.5+)

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| LLaMA-2 7B (Q4_K_M) | GGUF | ~4 GB | Popular quantized model |
| Mistral-7B (Q5_0) | GGUF | ~5 GB | Mistral quantized variant |
| LLaMA-2 7B (Q8_0) | GGUF | ~7 GB | Less aggressive quantization |
| Small LLaMA (1.1B, Q4_0) | GGUF | ~600 MB | Smaller GGUF baseline |

### LoRA Adapters (v0.5+)

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| LoRA for LLaMA-7B | safetensors | ~50-200 MB | Small, potentially compressible |
| QLoRA adapters | safetensors | ~100-500 MB | Quantized adapters |

### Training Checkpoints (v0.5+)

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| GPT-2 checkpoint (step 1000) | Mixed | ~1.5 GB | Includes optimizer state |
| BERT-base checkpoint | Mixed | ~1.2 GB | Includes optimizer state |

### Medium Models (1-10 GB)

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| LLaMA-2 7B | safetensors | ~13 GB | Popular open model |
| Mistral-7B | safetensors | ~14 GB | Efficient architecture |
| LLaMA-2 7B (Q4_K_M) | GGUF | ~4 GB | Quantized comparison |

## Metrics

### Compression Ratio

```
ratio = compressed_size / original_size
```

Lower is better. We report both per-file and overall archive ratio.

### Throughput

```
pack_throughput = original_size / pack_time    (bytes/second)
unpack_throughput = original_size / unpack_time (bytes/second)
```

### Memory Usage

Peak RSS during pack and unpack operations, measured via `resource.getrusage()` or similar.

### Verification Time

Time to verify an archive including decompression for hash verification.

## Benchmark Procedure

### 1. Download Model

Download the model files using Hugging Face `from_pretrained` or direct download. Record the exact revision/commit hash.

### 2. Pack with KMC

```bash
kmc pack ./model ./model.kmc -l 3
kmc pack ./model ./model.kmc -l 3 --tensor-aware
kmc pack ./model ./model-l9.kmc -l 9

# Pack with tensor-aware codecs
kmc pack ./model ./model.kmc --codec auto --tensor-aware
kmc pack ./model ./model-bp.kmc --codec byteplane --tensor-aware
kmc pack ./model ./model-fp.kmc --codec floatplane --tensor-aware

# Pack with GGUF-aware mode (for GGUF files)
kmc pack ./model.gguf ./model.kmc --gguf-aware
kmc pack ./model.gguf ./model.kmc --gguf-aware --tensor-aware
```

### 3. Pack with Baseline Tools

```bash
tar cf - ./model | zstd -3 -o model.tar.zst
tar cf - ./model | zstd -9 -o model-l9.tar.zst
zip -r model.zip ./model
```

### 4. ZipNN Comparison

```bash
pip install zipnn
kmc bench ./model ./model-bench.kmc --compare-zipnn --json --output bench-results.json
```

### 5. Multi-Codec Comparison (v0.4+)

```bash
# Compare all available codecs
kmc bench ./model ./model-bench.kmc --compare-codecs --json --output codec-comparison.json

# Or use the real model benchmark script
python scripts/bench_small_hf_model.py ./model
```

### 6. Artifact-Specific Benchmarks (v0.5+)

```bash
# LoRA adapter
kmc pack-lora ./lora-adapter ./lora.kmc
kmc verify ./lora.kmc

# Training checkpoint
kmc pack-checkpoint ./checkpoint-1000 ./checkpoint.kmc
kmc verify ./checkpoint.kmc
```

### 7. Verify and Unpack

```bash
kmc verify ./model.kmc
kmc unpack ./model.kmc ./restored/
diff -r ./model ./restored/
```

### 8. Record Results

For each combination of model, tool, codec, and compression level, record:
- Original size
- Compressed size
- Compression ratio
- Pack time
- Unpack time
- Verify time
- Peak memory usage
- Environment metadata
- Artifact type (if applicable)

### 9. Compare

Generate comparison tables and charts showing KMC vs. baselines vs. ZipNN.

## Expected Results

Based on ZipNN's published results and the nature of AI model weights:

- **safetensors**: Expected 30-50% compression ratio, similar to ZipNN. The uniform float32/float16 data may benefit from zstd's dictionary mode across blocks. BytePlane and FloatPlane codecs may improve on this for FP16/BF16 tensors.
- **GGUF (quantized)**: Expected 5-15% compression, since quantized data is already compact. Some metadata and vocabulary sections may compress well. GGUF-aware mode may improve ratios by avoiding float-aware transforms on quantized data. Tensor-aware codecs are not expected to help here.
- **LoRA adapters**: Expected 40-60% compression, as low-rank matrices have significant structure.
- **Training checkpoints**: Expected 30-50% compression for model weights; optimizer states may compress differently depending on their structure.
- **PyTorch .bin**: Expected 30-50%, similar to safetensors, but with additional pickle overhead that may not compress as well.

**These are expectations, not guarantees.** Actual results will be published once real benchmarks are completed.

## Automation

Benchmarks should be automated via a script that:
1. Downloads models to a cache directory.
2. Runs all pack/unpack/verify operations.
3. Records timing and size metrics.
4. Generates a summary report (Markdown + charts).

Future CI integration could run benchmarks on a schedule and track performance regressions.

## Important Notes

### No Fabricated Results

KMC will never invent or fabricate benchmark results. If a measurement is not available (e.g., ZipNN is not installed), the output will clearly indicate that the measurement was not taken, not substitute a fake number.

### Fair Comparison

When comparing with ZipNN:
- Use the same input data and compression level where possible.
- Report both tools' versions.
- Report the full environment (CPU, RAM, OS).
- Acknowledge when ZipNN performs better -- this is a measurement, not a competition.
- Note that KMC and ZipNN may optimize for different things (KMC for block-level access, ZipNN for maximum ratio).

### Synthetic Data Warning

Synthetic data benchmarks (using random or repetitive test data) produce results that are not representative of real-world compression. All synthetic benchmarks are clearly marked with `synthetic: true` in JSON output.

### VRAM Disclaimer

KMC benchmarks measure **storage compression ratio**, not inference VRAM reduction. A smaller .kmc archive does not mean the model uses less VRAM during inference. The model must be fully decompressed before loading, and occupies the same VRAM regardless of how it was stored.

## Codec Comparison Methodology (v0.4+)

When comparing codecs (via `--compare-codecs` or `bench_small_hf_model.py`), the following methodology applies:

1. **Same input data**: All codecs compress the exact same source files.
2. **Same block size**: All codecs use the same block size (default 256 KiB).
3. **Roundtrip verified**: Each codec must pass roundtrip verification (decompress matches original) to be included in results.
4. **Metrics reported**: Compressed size, compression ratio, pack time, throughput.
5. **Actual codecs used**: For `auto` mode, the report shows which codec was actually selected for each block.
6. **No invented numbers**: If a codec fails or is unavailable, it is marked as such -- no placeholder numbers are used.
7. **Environment metadata**: All results include Python version, OS, CPU, RAM, and dependency versions for reproducibility.

### Interpreting Codec Comparison Results

- **`auto` mode** may choose different codecs for different blocks. A model with mixed dtypes may use FloatPlane for FP16 blocks and zstd for INT8 blocks.
- **`byteplane`** and **`floatplane`** apply a transformation before compression. For already-compressed or non-floating-point data, these may produce larger output than raw zstd.
- **`raw`** is a passthrough that stores data uncompressed. It serves as a baseline.
- Results are **specific to the model and hardware** and should not be generalized without additional testing.

### GGUF-Aware Benchmark Methodology (v0.5+)

When comparing GGUF-aware vs. non-GGUF-aware compression:

1. **Same GGUF file**: Both modes compress the exact same source file.
2. **Same block size and level**: All other parameters are identical.
3. **Report quantization summary**: Include the quantization breakdown (e.g., Q4_K: 199, F32: 1) so readers can understand the model composition.
4. **Per-block codec comparison**: Show which codec was selected for each type of tensor (quantized vs. floating-point).
5. **GGUF-aware mode should not worsen results**: On files that are not GGUF, `--gguf-aware` should have no effect.
