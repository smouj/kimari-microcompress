# Benchmark Plan

## Goals

1. Measure compression ratio, speed, and memory usage for real AI model files.
2. Compare KMC against general-purpose compression tools (tar+zstd, zip).
3. Compare against ZipNN where applicable.
4. Provide reproducible benchmark results that inform optimization priorities.
5. Never invent or fabricate benchmark results.
6. Clearly distinguish between synthetic and real data benchmarks.

## KMC v0.3.0-alpha Benchmark Capabilities

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
```

### ZipNN Comparison

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
    "kmc_version": "0.3.0-alpha",
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
| Small LLaMA (1.1B) | GGUF (Q4_0) | ~600 MB | Quantized format baseline |
| BERT-base | safetensors | ~440 MB | Encoder model variety |

### Medium Models (1-10 GB)

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| LLaMA-2 7B | safetensors | ~13 GB | Popular open model |
| Mistral-7B | safetensors | ~14 GB | Efficient architecture |
| LLaMA-2 7B (Q4_K_M) | GGUF | ~4 GB | Quantized comparison |

### LoRA Adapters

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| LoRA for LLaMA-7B | safetensors | ~50-200 MB | Small, potentially compressible |
| QLoRA adapters | safetensors | ~100-500 MB | Quantized adapters |

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

### 5. Verify and Unpack

```bash
kmc verify ./model.kmc
kmc unpack ./model.kmc ./restored/
diff -r ./model ./restored/
```

### 6. Record Results

For each combination of model, tool, and compression level, record:
- Original size
- Compressed size
- Compression ratio
- Pack time
- Unpack time
- Verify time
- Peak memory usage
- Environment metadata

### 7. Compare

Generate comparison tables and charts showing KMC vs. baselines vs. ZipNN.

## Expected Results

Based on ZipNN's published results and the nature of AI model weights:

- **safetensors**: Expected 30-50% compression ratio, similar to ZipNN. The uniform float32/float16 data may benefit from zstd's dictionary mode across blocks.
- **GGUF (quantized)**: Expected 5-15% compression, since quantized data is already compact. Some metadata and vocabulary sections may compress well.
- **LoRA adapters**: Expected 40-60% compression, as low-rank matrices have significant structure.
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
- Acknowledge when ZipNN performs better — this is a measurement, not a competition.
- Note that KMC and ZipNN may optimize for different things (KMC for block-level access, ZipNN for maximum ratio).

### Synthetic Data Warning

Synthetic data benchmarks (using random or repetitive test data) produce results that are not representative of real-world compression. All synthetic benchmarks are clearly marked with `synthetic: true` in JSON output.
