# Benchmark Plan

## Goals

1. Measure compression ratio, speed, and memory usage for real AI model files.
2. Compare KMC against general-purpose compression tools (tar+zstd, zip).
3. Compare against ZipNN where applicable.
4. Provide reproducible benchmark results that inform optimization priorities.

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

Time to verify an archive without decompressing (should be I/O-bound, not CPU-bound).

## Benchmark Procedure

### 1. Download Model

Download the model files using Hugging Face `from_pretrained` or direct download. Record the exact revision/commit hash.

### 2. Pack with KMC

```bash
kmc pack ./model ./model.kmc -l 3
kmc pack ./model ./model-l9.kmc -l 9
```

### 3. Pack with Baseline Tools

```bash
tar cf - ./model | zstd -3 -o model.tar.zst
tar cf - ./model | zstd -9 -o model-l9.tar.zst
zip -r model.zip ./model
```

### 4. Verify and Unpack

```bash
kmc verify ./model.kmc
kmc unpack ./model.kmc ./restored/
diff -r ./model ./restored/
```

### 5. Record Results

For each combination of model, tool, and compression level, record:
- Original size
- Compressed size
- Compression ratio
- Pack time
- Unpack time
- Verify time
- Peak memory usage

### 6. Compare

Generate comparison tables and charts showing KMC vs. baselines.

## Expected Results

Based on ZipNN's published results and the nature of AI model weights:

- **safetensors**: Expected 30-50% compression ratio, similar to ZipNN. The uniform float32/float16 data may benefit from zstd's dictionary mode across blocks.
- **GGUF (quantized)**: Expected 5-15% compression, since quantized data is already compact. Some metadata and vocabulary sections may compress well.
- **LoRA adapters**: Expected 40-60% compression, as low-rank matrices have significant structure.
- **PyTorch .bin**: Expected 30-50%, similar to safetensors, but with additional pickle overhead that may not compress as well.

## Automation

Benchmarks should be automated via a script that:
1. Downloads models to a cache directory.
2. Runs all pack/unpack/verify operations.
3. Records timing and size metrics.
4. Generates a summary report (Markdown + charts).

Future CI integration could run benchmarks on a schedule and track performance regressions.
