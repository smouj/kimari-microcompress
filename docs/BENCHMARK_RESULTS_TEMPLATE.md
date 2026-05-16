# Benchmark Results Template

> **Template for recording KMC benchmark results**
> **KMC version**: v0.8.0-alpha
> **Last updated**: 2025

---

This template provides a standardized format for recording and publishing KMC benchmark results. Copy this template and fill in all applicable fields. Fields marked **(required)** must be completed for the result to be considered valid per the [Publishing Benchmarks](./PUBLISHING_BENCHMARKS.md) guidelines.

---

## Test Metadata

| Field | Value |
|---|---|
| **Benchmark ID** | _(unique identifier, e.g., `bench-2025-001`)_ **(required)** |
| **Date** | _(YYYY-MM-DD)_ **(required)** |
| **KMC version** | _(e.g., `0.8.0-alpha`)_ **(required)** |
| **KMC git commit** | _(short hash, e.g., `a3f2b8c`)_ |
| **Input name** | _(model name or description)_ **(required)** |
| **Input source** | _(URL or description of where the data came from)_ **(required)** |
| **Input format** | _(safetensors / GGUF / LoRA / checkpoint / mixed)_ **(required)** |
| **Input SHA-256** | _(hash of the input directory or file)_ |
| **Is synthetic data?** | _(yes / no)_ **(required)** |
| **Number of runs** | _(how many times the benchmark was repeated)_ **(required)** |
| **Warm-up runs** | _(how many warm-up runs were performed before measurement)_ |

## Configuration

| Parameter | Value |
|---|---|
| **Block size** | _(bytes, e.g., `262144`)_ **(required)** |
| **Compression level** | _(e.g., `3`)_ **(required)** |
| **Codec** | _(auto / zstd / zlib / byteplane / floatplane / gguf_quant_block / raw)_ **(required)** |
| **Tensor-aware** | _(yes / no)_ **(required)** |
| **GGUF-aware** | _(yes / no)_ **(required)** |
| **Dedup enabled** | _(yes / no)_ **(required)** |
| **Delta base** | _(path to base archive, or "none")_ **(required)** |
| **Jobs (--jobs)** | _(number of parallel workers)_ **(required)** |
| **Custom YAML config** | _(path to config file, or "default")_ |

## Environment

| Field | Value |
|---|---|
| **CPU** | _(model and core count, e.g., "AMD Ryzen 9 7950X, 32 cores")_ **(required)** |
| **RAM** | _(total RAM, e.g., "64 GB DDR5-5600")_ **(required)** |
| **GPU** | _(model, if relevant for I/O benchmarks)_ |
| **OS** | _(e.g., "Ubuntu 22.04 LTS")_ **(required)** |
| **Python version** | _(e.g., "3.11.5")_ **(required)** |
| **Storage type** | _(NVMe SSD / SATA SSD / HDD / network)_ **(required)** |
| **Filesystem** | _(e.g., ext4 / APFS / NTFS)_ **(required)** |
| **zstd available** | _(yes / no, with version if yes)_ **(required)** |
| **ZipNN available** | _(yes / no, with version if yes)_ |
| **CPU governor** | _(performance / powersave / ondemand)_ |
| **Caches dropped** | _(yes / no — were filesystem caches cleared between runs?)_ |

## Input Data Summary

| Field | Value |
|---|---|
| **Original size** | _(bytes)_ **(required)** |
| **Number of files** | _(total files in the input)_ **(required)** |
| **Number of model files** | _(safetensors / GGUF / bin files)_ |
| **Number of config/metadata files** | _(JSON, TXT, etc.)_ |
| **Sharded?** | _(yes / no)_ |
| **Number of shards** | _(if sharded)_ |
| **Number of tensors** | _(total tensor count)_ |
| **Dtype breakdown** | _(e.g., "BF16: 273, INT64: 6, FP32: 12")_ **(required)** |
| **Total tensor bytes** | _(bytes of tensor data only)_ |
| **GGUF quantization level** | _(e.g., "Q4_K_M", or "N/A")_ |

## Results

### Overall Results

| Metric | Run 1 | Run 2 | Run 3 | Mean | Std Dev |
|---|---|---|---|---|---|
| **Compressed size (bytes)** | | | | | |
| **Compression ratio** | | | | | |
| **Pack time (s)** | | | | | |
| **Unpack time (s)** | | | | | |
| **Verify time (s)** | | | | | |
| **Pack throughput (MB/s)** | | | | | |
| **Unpack throughput (MB/s)** | | | | | |

> **Note**: Add or remove run columns as needed to match your actual number of runs. If only a single run was performed, fill in the "Run 1" column and leave others blank. Report mean and standard deviation if 3+ runs were performed.

### Per-File Results (Optional)

If the input contains multiple files with significantly different compression characteristics, break down results per file:

| File | Original Size | Compressed Size | Ratio | Codec(s) |
|---|---|---|---|---|
| _(e.g., model-00001-of-00003.safetensors)_ | | | | |
| _(e.g., model-00002-of-00003.safetensors)_ | | | | |
| _(e.g., model-00003-of-00003.safetensors)_ | | | | |
| _(e.g., config.json)_ | | | | |
| _(e.g., tokenizer.json)_ | | | | |
| **Total** | | | | |

## Codec Comparison

If `--compare-codecs` was used, record the per-codec benchmark results here. These are measured on a 1 MB sample from the input data.

| Codec | Original (bytes) | Compressed (bytes) | Ratio | Compress Time (s) | Decompress Time (s) | Comp Throughput (MB/s) | Decomp Throughput (MB/s) | Roundtrip OK? |
|---|---|---|---|---|---|---|---|---|
| raw | | | 1.000 | — | — | — | — | yes |
| zlib | | | | | | | | |
| zstd | | | | | | | | |
| byteplane | | | | | | | | |
| floatplane | | | | | | | | |

### Codec Selection Summary

When using `--codec auto`, record which codec was selected for each block category:

| Block Category | Selected Codec | Block Count | Avg Ratio |
|---|---|---|---|
| BF16 tensors | _(e.g., floatplane)_ | | |
| FP16 tensors | | | |
| FP32 tensors | | | |
| INT8 / INT16 tensors | | | |
| Config / metadata | | | |
| GGUF quantized | | | |
| Other / unknown | | | |

## External Comparison (Optional)

If comparing against other compression tools, record their results here. Ensure fair comparison per the [Publishing Benchmarks](./PUBLISHING_BENCHMARKS.md) guidelines.

| Tool | Version | Compressed Size (bytes) | Ratio | Pack Time (s) | Unpack Time (s) | Settings |
|---|---|---|---|---|---|---|
| KMC | _(version)_ | | | | | _(settings)_ |
| ZipNN | _(version)_ | | | | | _(settings)_ |
| gzip | _(version)_ | | | | | _(level)_ |
| zstd | _(version)_ | | | | | _(level)_ |
| tar+zstd | _(version)_ | | | | | _(level)_ |

> **Disclaimer**: This is a measurement, not a claim of superiority. Results are specific to the input data, environment, and settings listed above. Different inputs or environments may produce different relative performance.

## Dedup Statistics (If --dedup Was Used)

| Metric | Value |
|---|---|
| **Total blocks** | |
| **Unique blocks** | |
| **Deduplicated blocks** | |
| **Saved bytes** | |
| **Dedup savings ratio** | _(saved_bytes / total_original_bytes)_ |

## Delta Statistics (If --delta-base Was Used)

| Metric | Value |
|---|---|
| **Base archive** | _(path)_ |
| **Base archive SHA-256** | |
| **Total blocks** | |
| **Changed blocks** | |
| **Referenced blocks** | |
| **Delta size vs. full archive size** | _(ratio of delta archive to full archive)_ |

## Runtime Hints (From Manifest)

Record the `runtime_hints` from the archive manifest:

| Hint | Value |
|---|---|
| **partial_file_access** | _(supported / unsupported)_ |
| **tensor_access** | _(index_based / none)_ |
| **compressed_inference** | _(always false)_ |

## Notes

_(Use this section for any additional observations, caveats, or context that would help someone interpret these results.)_

Examples of useful notes:

- "The model's embedding layer (token_embedding.weight) is FP32 and accounts for 40% of the model size. It does not compress well with floatplane (ratio: 0.95) but compresses well with zstd (ratio: 0.42)."
- "Three config files (< 1 KB each) were stored raw because compression increased their size."
- "The benchmark machine was under light load from a background process during run 2, which may explain the slightly higher pack time."
- "This model uses BF16 for all linear layers and FP32 for layer norms. The FP32 tensors are only 2% of the total model size."
- "GGUF Q4_K_M quantization reduces the model to 4.1 GB; KMC's additional compression on top of quantization yields 3.6 GB (12% further reduction)."

## Reproducibility Statement

To reproduce these results:

```bash
# 1. Install the exact KMC version
pip install kimari-microcompress==<VERSION>

# 2. Obtain the input data
# _(describe how to obtain the exact input data used in this benchmark)_

# 3. Run the benchmark
kmc bench <source> <output.kmc> \
    --tensor-aware \
    --codec <codec> \
    --json \
    --output-file <benchmark-id>.json
```

**Expected variance**: Results should be within ±5% of the reported mean on comparable hardware. Variance is primarily caused by filesystem cache state, background system load, and CPU thermal throttling.

---

## Appendix: JSON Output Reference

When using `kmc bench --json --output-file results.json`, the output is a JSON document with the following top-level structure:

```json
{
  "tool": "kmc-bench",
  "kmc_version": "0.8.0-alpha",
  "source": "/path/to/input",
  "synthetic": false,
  "original_size": 13484934656,
  "num_files": 15,
  "num_blocks": 260,
  "block_size": 262144,
  "detected_formats": ["safetensors"],
  "kmc_pack_time": 12.34,
  "kmc_unpack_time": 8.91,
  "kmc_verify_time": 1.23,
  "kmc_compressed_size": 5248912384,
  "kmc_ratio": 0.3893,
  "kmc_pack_throughput": 1093000000,
  "kmc_unpack_throughput": 1513000000,
  "tensor_aware": true,
  "codec_used": "auto",
  "codec_benchmarks": [
    {
      "codec": "zstd",
      "original_size": 1048576,
      "compressed_size": 412672,
      "ratio": 0.3935,
      "compress_time": 0.0123,
      "decompress_time": 0.0045,
      "compress_throughput": 85240000,
      "decompress_throughput": 233013000,
      "applicable": true,
      "roundtrip_ok": true
    }
  ],
  "zipnn_benchmark": null,
  "environment": {
    "python_version": "3.11.5",
    "os_name": "Linux",
    "os_version": "5.15.0-91-generic",
    "cpu": "AMD Ryzen 9 7950X 32-Core Processor",
    "ram_gb": 62.8,
    "kmc_version": "0.8.0-alpha",
    "zipnn_version": "",
    "zstd_available": true
  }
}
```

This JSON output contains all the information needed to fill in the template above. When publishing benchmark results, consider including the raw JSON output as a supplementary file for maximum transparency.
