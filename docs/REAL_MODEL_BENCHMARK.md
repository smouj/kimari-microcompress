# Real Model Benchmark

This document explains how to run the `bench_small_hf_model.py` script to benchmark KMC compression on real HuggingFace models, and how to interpret the results.

## Overview

The `scripts/bench_small_hf_model.py` script compresses a small model from HuggingFace using multiple KMC codecs and generates a comparison report. It tests `auto`, `zstd`, `zlib`, `byteplane`, and `floatplane` codecs on the same model directory and reports compressed size, ratio, timing, and which codecs were actually selected per block.

**The script does NOT download models automatically.** You must download the model first using `huggingface-cli` or the `huggingface_hub` Python package.

## Prerequisites

```bash
# Install KMC with all optional dependencies
pip install -e ".[all]"

# Install HuggingFace Hub for model downloads
pip install huggingface_hub
```

## Suggested Models

These small models are ideal for quick benchmarking:

| Model | Size | Dtype | Notes |
|-------|------|-------|-------|
| `sshleifer/tiny-gpt2` | ~5 MB | FP32 | Very small, fast to test |
| `hf-internal-testing/tiny-random-gpt2` | ~2 MB | FP32 | Minimal test model |
| `gpt2` | ~500 MB | FP32 | Full GPT-2, realistic benchmark |
| `distilgpt2` | ~350 MB | FP32 | Smaller GPT-2 variant |
| `bert-base-uncased` | ~440 MB | FP32 | Encoder model variety |
| `google/flan-t5-small` | ~300 MB | FP32 | Encoder-decoder model |

For GGUF benchmarks (v0.5+):

| Model | Size | Quantization | Notes |
|-------|------|-------------|-------|
| `TheBloke/Llama-2-7B-Chat-GGUF` (Q4_K_M) | ~4 GB | Q4_K_M | Popular quantized model |
| `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` (Q5_0) | ~5 GB | Q5_0 | Mistral quantized variant |

For LoRA adapter benchmarks (v0.5+):

| Model | Size | Notes |
|-------|------|-------|
| `winddude/wizardLM-7B-uncensored-lora` | ~50-200 MB | LoRA adapter for LLaMA |
| Any PEFT adapter from HuggingFace | Varies | Use `kmc pack-lora` for best results |

> **Tip:** Start with `sshleifer/tiny-gpt2` for a quick smoke test (~5 MB), then move to `gpt2` for realistic results.

## Running the Benchmark

### Step 1: Download a Model

```bash
# Download a tiny model for quick testing
huggingface-cli download sshleifer/tiny-gpt2 --local-dir ./tiny-gpt2

# Or download a full model for realistic results
huggingface-cli download gpt2 --local-dir ./models/gpt2
```

### Step 2: Run the Benchmark

```bash
# Run with default settings
python scripts/bench_small_hf_model.py ./tiny-gpt2

# Specify a custom output directory
python scripts/bench_small_hf_model.py ./models/gpt2 --output-dir ./bench-results
```

### Step 3: Review Results

The script outputs:

1. **Console table** -- A markdown-formatted comparison table printed to stdout.
2. **JSON file** -- Detailed results saved to `benchmark_results.json` in the output directory.
3. **Per-codec verification** -- Each archive is verified after compression to confirm byte-exact roundtrip.

## Benchmarking LoRA Adapters (v0.5+)

```bash
# Download a LoRA adapter
huggingface-cli download winddude/wizardLM-7B-uncensored-lora --local-dir ./lora-adapter

# Benchmark with standard pack
python scripts/bench_small_hf_model.py ./lora-adapter

# Or use pack-lora for dedicated workflow
kmc pack-lora ./lora-adapter ./lora-adapter.kmc
kmc verify ./lora-adapter.kmc
```

## Benchmarking GGUF Files (v0.5+)

```bash
# Download a GGUF file (manual download from HuggingFace)
# Then benchmark with GGUF-aware mode
kmc pack ./model.gguf ./model.kmc --gguf-aware
kmc verify ./model.kmc
kmc inspect ./model.kmc --compression

# Compare with non-GGUF-aware mode
kmc pack ./model.gguf ./model-default.kmc
kmc inspect ./model-default.kmc --compression

# Compare sizes
ls -la ./model.kmc ./model-default.kmc
```

## Expected Output

### Console Output (Example)

```
Model: /path/to/tiny-gpt2
Original size: 5.12 MB
Files: 4

Testing codec: auto...
  5.12 MB -> 2.87 MB (56.05%) in 0.45s
Testing codec: zstd...
  5.12 MB -> 2.91 MB (56.84%) in 0.12s
Testing codec: zlib...
  5.12 MB -> 3.05 MB (59.57%) in 0.08s
Testing codec: byteplane...
  5.12 MB -> 2.78 MB (54.30%) in 0.52s
Testing codec: floatplane...
  5.12 MB -> 2.65 MB (51.76%) in 0.89s

## Codec Comparison

| Codec | Compressed | Ratio | Time (s) | Throughput (MB/s) | Actual Codecs |
|-------|-----------|-------|----------|-------------------|---------------|
| auto | 2.87 MB | 56.05% | 0.450 | 11.38 | floatplane, zstd |
| zstd | 2.91 MB | 56.84% | 0.120 | 42.67 | zstd |
| zlib | 3.05 MB | 59.57% | 0.080 | 64.00 | zlib |
| byteplane | 2.78 MB | 54.30% | 0.520 | 9.85 | byteplane |
| floatplane | 2.65 MB | 51.76% | 0.890 | 5.75 | floatplane |

> **Disclaimer**: Results are specific to this model and hardware.
> KMC does NOT reduce inference VRAM.
> This is a measurement, not a claim of superiority.
```

> **Note:** The numbers above are illustrative placeholders. Actual results will vary based on model, hardware, and software versions.

### JSON Output

The JSON file includes:

```json
{
  "model_path": "/path/to/tiny-gpt2",
  "original_size": 5369856,
  "original_size_human": "5.12 MB",
  "num_files": 4,
  "zstd_available": true,
  "disclaimer": "Results are specific to this model and hardware. KMC does NOT reduce inference VRAM. This is a measurement, not a claim of superiority.",
  "codec_results": [
    {
      "codec": "auto",
      "original_size": 5369856,
      "compressed_size": 3008512,
      "ratio": 0.5605,
      "pack_time_s": 0.45,
      "throughput_mb_s": 11.38,
      "blocks": 4,
      "actual_codecs_used": ["floatplane", "zstd"],
      "verify_ok": true
    }
  ]
}
```

## Understanding the Results

### Codec Explanations

- **`auto`**: Uses the automatic codec selector. For FP32/FP16/BF16 tensors, it tries `floatplane -> byteplane -> zstd -> zlib -> raw` in order and picks the smallest verified result per block. The "Actual Codecs" column shows which codecs were actually selected.
- **`zstd`**: Pure zstd compression without any tensor-aware transformation. Fast and effective for general data.
- **`zlib`**: Pure zlib compression. Slightly worse ratio than zstd but always available (built-in).
- **`byteplane`**: Byte-plane separation + zstd. Reorganizes bytes by their position within each element before compressing.
- **`floatplane`**: Sign/exponent/mantissa bit-level separation + zstd. Most granular floating-point decomposition.

### Tradeoffs

| Codec | Compression Ratio | Speed | Notes |
|-------|-------------------|-------|-------|
| `raw` | 100% (no compression) | Fastest | Baseline, data stored as-is |
| `zstd` | Good | Fast | Best general-purpose codec |
| `zlib` | Good | Moderate | Always available fallback |
| `byteplane` | Better (for floats) | Slower | Requires dtype metadata |
| `floatplane` | Best (for floats) | Slowest | Most granular, requires dtype |
| `auto` | Best overall | Moderate | Tries all, picks smallest per block |

### Actual Codecs Used

In `auto` mode, the "Actual Codecs" column shows which codecs were selected per block. A model directory typically contains non-tensor files (JSON configs, tokenizer files) that are not floating-point data -- these will use zstd or zlib, not BytePlane or FloatPlane.

### GGUF-Aware Results (v0.5+)

When benchmarking GGUF files with `--gguf-aware`, the results will show:

- Quantized tensor blocks (Q4_K, Q5_0, etc.) use `zstd` or `zlib` only.
- Floating-point tensor blocks (F32, F16, BF16) may use `floatplane` or `byteplane`.
- The `auto` codec selector automatically adjusts its strategy based on the GGML type of each tensor.

## Disclaimers

1. **KMC does NOT reduce inference VRAM.** These benchmarks measure storage compression ratio. The decompressed model occupies the same VRAM regardless of how it was compressed.

2. **Results are model-specific.** Compression ratios depend on the model's weight distribution, dtype, and architecture. Results from one model cannot be generalized to all models.

3. **Small models may not be representative.** Very small models (like `sshleifer/tiny-gpt2` at ~5 MB) may show different compression characteristics than large production models due to smaller block counts and different weight distributions.

4. **This is a measurement, not a claim of superiority.** If BytePlane or FloatPlane produces larger output than plain zstd for a particular model, that is a valid result. The `auto` selector handles this by choosing the smallest verified result.

5. **No invented benchmark numbers.** If a codec fails or is unavailable, the result will show an error message rather than a fabricated number.

6. **No fabricated benchmarks.** KMC does not invent, fabricate, or estimate benchmark results. All numbers come from actual measurements.

## Integration with CI

The script can be integrated into CI pipelines for regression testing:

```bash
# Quick smoke test with tiny model
python scripts/bench_small_hf_model.py ./tiny-gpt2 --output-dir ./ci-bench
# Check that all codecs produced verify_ok=true
python -c "
import json
results = json.load(open('./ci-bench/benchmark_results.json'))
failures = [r for r in results['codec_results'] if not r.get('verify_ok', False)]
if failures:
    print('FAIL: codec verification errors:', failures)
    exit(1)
print('All codecs verified OK')
"
```

## Related Documentation

- [Benchmark Plan](BENCHMARK_PLAN.md) -- Full benchmark strategy and methodology
- [Architecture](ARCHITECTURE.md) -- Codec architecture and design decisions
- [Research Notes](RESEARCH_NOTES.md) -- BytePlane and FloatPlane design rationale
- [Hugging Face Workflow](HUGGINGFACE_WORKFLOW.md) -- General HuggingFace model workflow
- [GGUF Support](GGUF_SUPPORT.md) -- GGUF parsing and `--gguf-aware` mode
- [LoRA Workflow](LORA_WORKFLOW.md) -- LoRA adapter compression and inspection
