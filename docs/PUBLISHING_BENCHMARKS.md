# Publishing Benchmarks

> **Guidelines for publishing KMC benchmark results**
> **KMC version**: v0.8.0-alpha
> **Last updated**: 2025

## Overview

Benchmark results are one of the most visible and influential aspects of any compression tool. They are also one of the most easily misused. This document provides guidelines for publishing benchmark results for Kimari MicroCompress (KMC) that are honest, reproducible, and fair.

These guidelines apply to benchmark results published in any medium: GitHub issues, blog posts, papers, presentations, social media, or marketing materials. They are binding for anyone publishing results under the KMC project name and strongly recommended for anyone publishing results about KMC.

## What to Include

Every published KMC benchmark result **must** include the following information to be considered valid and reproducible.

### 1. Environment Information

The complete hardware and software environment in which the benchmark was run:

| Required Field | Example |
|---|---|
| CPU model | AMD Ryzen 9 7950X (32 cores) |
| RAM | 64 GB DDR5-5600 |
| OS | Ubuntu 22.04 LTS |
| Python version | 3.11.5 |
| KMC version | 0.8.0-alpha |
| zstd availability | Yes (python-zstandard 0.21.0) |
| Storage type | NVMe SSD (Samsung 990 PRO) |
| Filesystem | ext4 |

The KMC benchmark tool (`kmc bench`) automatically collects most of this information in its JSON output via the `EnvironmentInfo` dataclass. Use it:

```bash
kmc bench ./my-model/ ./benchmark.kmc --json --output-file bench-results.json
```

The `environment` field in the JSON output contains Python version, OS, CPU, RAM, KMC version, and zstd availability. Storage type and filesystem must be recorded manually.

### 2. KMC Version and Configuration

The exact KMC version and all configuration parameters used:

| Parameter | Required | Example |
|---|---|---|
| KMC version | Yes | 0.8.0-alpha |
| Block size | Yes | 262144 (256 KB) |
| Compression level | Yes | 3 |
| Codec | Yes | auto |
| Tensor-aware | Yes | true |
| GGUF-aware | Yes | false |
| Dedup | Yes | false |
| Delta base | Yes | (none) |
| Number of workers (`--jobs`) | Yes | 1 |

Never omit configuration parameters, even if they are set to defaults. Different default values across KMC versions can produce different results, making historical comparisons misleading.

### 3. Input Data Description

A clear description of the data that was compressed:

| Required Field | Example |
|---|---|
| Model name | Llama-2-7B-Chat |
| Source | Hugging Face (meta-llama/Llama-2-7b-chat-hf) |
| Format | safetensors (sharded, 3 files) |
| Original size | 13,484,934,656 bytes (12.56 GB) |
| Number of files | 15 (3 safetensors + 12 config/metadata) |
| Number of tensors | 291 |
| Dtypes | BF16 (273 tensors), INT64 (6 tensors), etc. |
| Is synthetic? | No |

**Synthetic data must be explicitly labeled.** Never present results on synthetic data as representative of real-world performance without clear labeling. Synthetic benchmarks are useful for testing edge cases and codec behavior, but they do not reflect the compression ratios or throughput that users will see on actual model data.

### 4. Methodology

A description of how the benchmark was conducted:

- **Warm-up runs**: Were any warm-up runs performed before the measured run? (Recommended: 1 warm-up run, discarded.)
- **Number of measured runs**: How many runs were averaged? (Recommended: minimum 3 runs.)
- **What was measured**: Pack time, unpack time, verify time, compressed size, ratio.
- **I/O inclusion**: Does the reported time include disk I/O or only CPU time? (KMC benchmarks include I/O by default; state this explicitly.)
- **Compression direction**: Results for "compression" should include the full `kmc pack` pipeline (reading, codec selection, compression, writing). Results for "decompression" should include the full `kmc unpack` pipeline (reading, decompression, writing, hash verification).

### 5. Results

The actual numerical results, presented in a clear and complete format. At minimum:

| Metric | Required |
|---|---|
| Original size (bytes) | Yes |
| Compressed size (bytes) | Yes |
| Compression ratio (compressed / original) | Yes |
| Pack time (seconds) | Yes |
| Unpack time (seconds) | Yes |
| Verify time (seconds) | Recommended |
| Pack throughput (MB/s) | Recommended |
| Unpack throughput (MB/s) | Recommended |
| Per-codec breakdown | Recommended |

## What NOT to Do

### 1. Do Not Invent Results

Never fabricate, estimate, or extrapolate benchmark results. All published numbers must come from actual benchmark runs using the `kmc bench` tool or equivalent measurement. If a benchmark was not run, do not publish results for it.

### 2. Do Not Compare Unfairly

When comparing KMC with other tools, ensure the comparison is fair:

- **Same input data**: Both tools must compress the same data. Do not compare KMC on a 7B model against a competitor on a 13B model.
- **Same environment**: Both tools must run on the same hardware, OS, and Python version. Do not compare KMC results from a fast NVMe SSD against a competitor running from a network drive.
- **Same or equivalent settings**: Compare compression levels that produce similar trade-offs. A KMC level-3 result should be compared against a competitor's default or medium-compression setting, not against their fastest or highest-compression setting unless explicitly stated.
- **Full pipeline comparison**: When comparing pack/unpack performance, compare the complete pipeline (reading, compression, writing, verification) — not just the raw codec throughput. Real-world performance includes I/O overhead that codec-only benchmarks ignore.
- **State the comparison criteria**: Clearly state what metric is being compared (ratio, speed, memory usage) and why it is relevant.

### 3. Do Not Claim Superiority

Benchmark results are measurements, not claims. Present them as observations:

**Acceptable**: "On this dataset, KMC achieved a 38% compression ratio in 12.3 seconds."

**Unacceptable**: "KMC is 2× faster than Tool X." (Unless you have fair, reproducible benchmarks showing this on the same data and hardware.)

**Acceptable**: "In our tests, KMC's compression ratio was comparable to ZipNN on BF16 safetensors data."

**Unacceptable**: "KMC compresses better than ZipNN." (Without specifying the data, settings, and conditions under which this was measured.)

### 4. Do Not Cherry-Pick Results

Do not selectively present only the best results while omitting unfavorable ones. If KMC performs poorly on a particular data type or configuration, include those results. Honest reporting builds trust.

Examples of cherry-picking to avoid:

- Reporting only the model file compression ratio while omitting the config files (which may not compress well).
- Reporting only the best codec (e.g., floatplane for BF16) without showing the default `auto` codec behavior.
- Reporting results from only the most compressible model in a family while omitting results from less compressible variants.
- Reporting pack speed but not unpack speed, or vice versa.

### 5. Do Not Misrepresent Compression Ratio

Compression ratio is `compressed_size / original_size`. A ratio of 0.38 means the compressed file is 38% the size of the original. Do not express this as "62% compression" (which is ambiguous) or "2.63× compression" (which is the inverse ratio and can be confused with the actual ratio). Always use the `compressed / original` convention and label it clearly.

### 6. Do Not Imply VRAM Reduction

KMC is a storage compression tool. It reduces disk space and transfer size. It does NOT reduce VRAM usage during inference. Never state or imply that KMC can reduce the GPU memory required to run a model. The model must be fully decompressed in memory before inference can proceed.

## Using the Benchmark Scripts

KMC includes several benchmark scripts in the `scripts/` directory:

### `kmc bench` (Built-in CLI)

The primary benchmark tool is the built-in `kmc bench` command:

```bash
# Basic benchmark
kmc bench ./my-model/ ./benchmark.kmc

# With codec comparison
kmc bench ./my-model/ ./benchmark.kmc --compare-codecs

# With ZipNN comparison (if ZipNN is installed)
kmc bench ./my-model/ ./benchmark.kmc --compare-zipnn

# Tensor-aware mode
kmc bench ./my-model/ ./benchmark.kmc --tensor-aware

# JSON output for programmatic processing
kmc bench ./my-model/ ./benchmark.kmc --json --output-file results.json

# Custom codec
kmc bench ./my-model/ ./benchmark.kmc --codec floatplane
```

The benchmark tool measures:
- Per-codec compression/decompression (1 MB sample)
- Full KMC pack pipeline (with throughput)
- Full KMC unpack pipeline (with throughput)
- KMC verify time
- Optional ZipNN comparison (with disclaimer)

### `scripts/bench_small_hf_model.py`

Benchmarks KMC on a small Hugging Face model downloaded from the Hub:

```bash
python scripts/bench_small_hf_model.py --model gpt2 --output bench-results/
```

### `scripts/bench_real_small_model.py`

Benchmarks KMC on a real small model with realistic tensor distributions:

```bash
python scripts/bench_real_small_model.py --output bench-results/
```

### `scripts/bench_real_gguf.py`

Benchmarks KMC on real GGUF model files with the `--gguf-aware` codec path:

```bash
python scripts/bench_real_gguf.py --gguf-file model-Q4_K_M.gguf --output bench-results/
```

## Reproducibility Requirements

### Minimum Requirements for Reproducibility

For a benchmark result to be reproducible, another person must be able to run the same benchmark and obtain results within ±5% of the published numbers. This requires:

1. **Exact KMC version**: The git commit hash or PyPI version used.
2. **Exact input data**: Either a public URL for the model, or a precise description (name, version, hash) so the same data can be obtained.
3. **Exact configuration**: All CLI flags and configuration values.
4. **Environment description**: Sufficient detail about the hardware and software environment.

### Recommended Reproducibility Practices

- **Publish the raw JSON output**: The `kmc bench --json` output contains all measured values and metadata. Include it as a supplementary file.
- **Publish the benchmark script**: If you used a custom benchmark script, include it in the publication or repository.
- **Run multiple times**: Report the mean and standard deviation across at least 3 runs. This accounts for system variability (background processes, thermal throttling, etc.).
- **Use a dedicated benchmark machine**: Avoid running benchmarks on shared or overloaded systems.
- **Disable frequency scaling**: Set the CPU governor to `performance` mode to avoid variability from dynamic frequency scaling.
- **Clear filesystem caches**: Run `sync && echo 3 > /proc/sys/vm/drop_caches` (Linux) between benchmark runs to ensure cold-cache behavior.

## Example Markdown Output Format

Below is a template for presenting benchmark results in Markdown. Use this format (or a close variant) when publishing KMC benchmarks:

```markdown
# KMC Benchmark: Llama-2-7B-Chat

## Test Metadata

| Field | Value |
|---|---|
| KMC version | 0.8.0-alpha |
| Date | 2025-01-15 |
| Input | meta-llama/Llama-2-7b-chat-hf |
| Format | safetensors (sharded, 3 files) |
| Original size | 13,484,934,656 bytes (12.56 GB) |
| Block size | 262,144 (256 KB) |
| Compression level | 3 |
| Codec | auto |
| Tensor-aware | yes |
| Number of runs | 3 (mean reported) |

## Environment

| Field | Value |
|---|---|
| CPU | AMD Ryzen 9 7950X (32 cores) |
| RAM | 64 GB DDR5-5600 |
| OS | Ubuntu 22.04 LTS |
| Python | 3.11.5 |
| Storage | NVMe SSD (Samsung 990 PRO) |
| zstd | Available (python-zstandard 0.21.0) |

## Results

| Metric | Value |
|---|---|
| Compressed size | 5,248,912,384 bytes (4.89 GB) |
| Compression ratio | 0.3893 |
| Pack time | 12.34 s |
| Unpack time | 8.91 s |
| Verify time | 1.23 s |
| Pack throughput | 1,093 MB/s |
| Unpack throughput | 1,513 MB/s |

## Codec Breakdown

| Codec | Blocks | Avg Ratio |
|---|---|---|
| floatplane | 245 | 0.37 |
| zstd | 12 | 0.42 |
| raw | 3 | 1.00 |

## Notes

- The 3 raw blocks correspond to small config files (< 1 KB each) that do not benefit from compression.
- This benchmark was run on a cold filesystem cache (dropped caches between runs).
- Results are the mean of 3 runs; standard deviation was < 2% across all metrics.

## Reproducibility

To reproduce these results:

```bash
# Install KMC v0.8.0-alpha
pip install kimari-microcompress==0.8.0a0

# Download the model
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./llama2-7b/

# Run the benchmark
kmc bench ./llama2-7b/ ./llama2-7b.kmc --tensor-aware --json --output-file results.json
```
```

---

By following these guidelines, you help ensure that KMC benchmark results are trustworthy, fair, and useful to the community. If you have questions about benchmark methodology or encounter results that seem inconsistent, please open an issue on the KMC GitHub repository.
