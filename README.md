<div align="center">

# Kimari MicroCompress

**Reversible lossless compression for AI model files**

*Safetensors · GGUF · LoRA/PEFT · Training Checkpoints · Hugging Face Models*

[![CI](https://github.com/smouj/kimari-microcompress/actions/workflows/ci.yml/badge.svg)](https://github.com/smouj/kimari-microcompress/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.7.0--alpha-orange.svg)](https://github.com/smouj/kimari-microcompress/releases)
[![Tests](https://img.shields.io/badge/tests-330%20passing-brightgreen.svg)](https://github.com/smouj/kimari-microcompress/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-DB7093.svg)](https://docs.astral.sh/ruff/)

</div>

---

## ⚠️ Important Limitations

> **Please read before using KMC:**

- **KMC does NOT perform compressed inference.** It is designed for storage, transfer, and verification — not runtime memory optimization. Partial access decompresses data before returning it.
- **KMC does NOT modify model weights.** Compression is lossless and reversible; every byte is preserved exactly.
- **Partial tensor loading returns bytes.** The `read_tensor` method and `--tensor` flag return raw bytes. To convert to native tensor objects (PyTorch, NumPy), the experimental safetensors loader is required and depends on optional tensor libraries being installed.
- **Tensor extraction depends on tensor metadata.** Archives must be created with `--tensor-aware` mode for tensor-level partial access. Older archives support file-level partial access but not tensor-level access.
- **GGUF-aware compression is experimental.** The `--gguf-aware` flag adjusts codec selection for quantized GGUF tensors but does not yet implement block-level GGUF-specific compression strategies.
- **No fixed compression ratios should be assumed.** Results vary significantly by model format, data type, and content. Synthetic benchmarks do not represent real-world ratios.
- **KMC is not a replacement for quantization.** If you need smaller models for inference, use quantization (GGUF Q4_K, GPTQ, AWQ, etc.). KMC is complementary: it compresses already-quantized files for storage/transfer.
- **No pickle is used.** KMC never deserializes pickle-based files. Only presence, size, and hash are recorded.
- **KMC is lossless only.** There is no lossy mode and no weight modification of any kind.
- **Safetensors loader is experimental.** The `load_tensor()` function may change without notice between versions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Codecs](#codecs)
- [Archive Format](#archive-format)
- [Architecture](#architecture)
- [Security](#security)
- [Documentation](#documentation)
- [Development](#development)
- [Technical Foundation](#technical-foundation)
- [License](#license)

---

## Overview

Kimari MicroCompress (KMC) is an experimental tool for **lossless, reversible compression** of AI model files. It focuses on **storage, transfer, verification, and packaging** without modifying the original weights. The approach is grounded in the observation that AI model files — particularly `safetensors` and quantized formats — contain significant redundancy that general-purpose compression tools don't exploit optimally.

**Key principle:** Every byte that goes in must come out identically. KMC provides byte-exact roundtrip integrity verified via SHA-256 hashes at both the file and block level.

### Why KMC?

| Problem | KMC Solution |
|---------|-------------|
| AI model files are large and expensive to store | Lossless compression with tensor-aware codecs (BytePlane, FloatPlane) |
| General-purpose tools ignore tensor structure | dtype-aware codec selection per block (FP32, BF16, FP16, quantized) |
| No integrity guarantees after compression | SHA-256 verification at file and block level |
| Mixed artifacts (model + LoRA + checkpoints) | Artifact-type detection and specialized workflows |
| GGUF quantized data doesn't compress well | Experimental `--gguf-aware` mode that adapts codec selection |
| No visibility into what's inside an archive | `kmc inspect` with format-specific metadata and tensor details |

---

## Features

| Feature | Status |
|---------|--------|
| `kmc pack` — Compress files/directories | ✅ Working |
| `kmc pack --tensor-aware` — Tensor-aware block alignment | ✅ Working |
| `kmc pack --gguf-aware` — Experimental GGUF-aware compression | 🧪 Experimental |
| `kmc pack-lora` — LoRA adapter workflow | ✅ Working |
| `kmc pack-checkpoint` — Training checkpoint workflow | ✅ Working |
| `kmc unpack` — Decompress archives (path-safe) | ✅ Working |
| `kmc unpack --only PATTERN` — Selective file extraction | ✅ Working |
| `kmc unpack --tensor NAME` — Selective tensor extraction | ✅ Working |
| `kmc unpack --list` — List archive contents before extracting | ✅ Working |
| `kmc verify` — Full verification report | ✅ Working |
| `kmc inspect` — AI model inspection with tensor metadata | ✅ Working |
| `kmc inspect` — Partial access info display | ✅ Working |
| `kmc list` — List archive files and tensors | ✅ Working |
| `kmc bench --partial-access` — Partial access benchmarks | ✅ Working |
| KMCReader Python API — Partial access without full decompression | ✅ Working |
| Experimental safetensors tensor loader | 🧪 Experimental |
| Artifact auto-detection (HuggingFace, GGUF, LoRA, checkpoint) | ✅ Working |
| GGUF tensor metadata parser (names, shapes, types, offsets, sizes) | ✅ Working |
| GGUF quantization summary (Q4_K, Q5_1, F32, etc.) | ✅ Working |
| Manifest v6 with index metadata and archive_offset | ✅ Working |
| SHA-256 per-file and per-block hashing | ✅ Working |
| 256 KiB micro-blocks (configurable) | ✅ Working |
| zstd / zlib / raw / byteplane / floatplane codec selection | ✅ Working |
| Automatic codec selector (dtype-based) | ✅ Working |
| safetensors real tensor metadata (names, shapes, dtypes, offsets) | ✅ Working |
| LoRA/PEFT adapter detection with rank and target modules | ✅ Working |
| Path traversal protection in unpack | ✅ Working |
| Backward compatible with .kmc v0.2/v0.3/v0.4/v0.5/v0.6 | ✅ Working |
| GGUF block-level compression | 🔬 Research |
| Runtime compressed loading (keeping blocks compressed in memory) | 🔬 Research |

---

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/smouj/kimari-microcompress.git
cd kimari-microcompress
pip install -e ".[dev]"

# With safetensors optional dependency (enhanced header parsing)
pip install -e ".[safetensors]"

# With ZipNN optional dependency (for benchmark comparison)
pip install -e ".[zipnn]"

# All optional dependencies
pip install -e ".[all]"
```

### Requirements

| Dependency | Required | Purpose |
|-----------|----------|---------|
| Python 3.10+ | Yes | Runtime |
| `zstandard` | Yes | Best compression codec |
| `zlib` | Yes (built-in) | Fallback compression |
| `safetensors` | No (optional) | Enhanced safetensors header parsing |
| `zipnn` | No (optional) | Benchmark comparison |

---

## Quick Start

```bash
# Pack a model directory
kmc pack ./my-model ./my-model.kmc

# Pack with tensor-aware mode (recommended for safetensors)
kmc pack ./my-model ./my-model.kmc --tensor-aware

# Pack with GGUF-aware mode (experimental, for GGUF files)
kmc pack ./my-model ./my-model.kmc --gguf-aware

# Pack a LoRA adapter
kmc pack-lora ./my-lora-adapter ./my-lora.kmc

# Pack a training checkpoint
kmc pack-checkpoint ./checkpoint-1000 ./checkpoint-1000.kmc

# Verify integrity (full report)
kmc verify ./my-model.kmc

# Inspect archive manifest
kmc inspect ./my-model.kmc

# Inspect AI model directory (detects formats, reads tensor metadata)
kmc inspect ./my-model/ --tensors

# Inspect as LoRA adapter
kmc inspect ./my-lora/ --lora

# Inspect as training checkpoint
kmc inspect ./checkpoint-1000/ --checkpoint

# Inspect GGUF file with tensor details
kmc inspect ./model.gguf --gguf

# Inspect with JSON output
kmc inspect ./my-model/ --json

# Unpack to a directory
kmc unpack ./my-model.kmc ./restored-model/

# Run benchmark with codec comparison
kmc bench ./my-model ./my-model-bench.kmc --compare-codecs

# Benchmark with ZipNN comparison
kmc bench ./my-model ./my-model-bench.kmc --compare-zipnn --json --output report.json
```

---

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `kmc pack SOURCE OUTPUT` | Compress a directory/file into a `.kmc` archive |
| `kmc pack-lora SOURCE OUTPUT` | Compress a LoRA adapter directory |
| `kmc pack-checkpoint SOURCE OUTPUT` | Compress a training checkpoint directory |
| `kmc unpack ARCHIVE OUTPUT` | Decompress a `.kmc` archive |
| `kmc verify ARCHIVE` | Full integrity verification report |
| `kmc inspect TARGET` | Inspect archive or AI model directory |
| `kmc list ARCHIVE` | List files and tensors in an archive |
| `kmc bench SOURCE OUTPUT` | Benchmark compression performance |

### Key Flags

| Flag | Command | Description |
|------|---------|-------------|
| `--tensor-aware` | pack | Align blocks to tensor boundaries for safetensors files |
| `--gguf-aware` | pack | Adjust codec selection for quantized GGUF tensors |
| `--codec` | pack, bench | Codec: `auto`, `byteplane`, `floatplane`, `zstd`, `zlib`, `raw` |
| `--only PATTERN` | unpack | Extract only files matching a glob pattern |
| `--tensor NAME` | unpack | Extract a specific tensor by name |
| `--list` | unpack | List available files/tensors without extracting |
| `--lora` | inspect | Inspect as LoRA adapter |
| `--checkpoint` | inspect | Inspect as training checkpoint |
| `--gguf` | inspect | Inspect as GGUF model with tensor details |
| `--tensors` | inspect, list | Show detailed tensor information |
| `--compression` | inspect | Show compression summary with codec usage |
| `--partial-access` | bench | Benchmark partial access performance |
| `--json` | inspect, bench, list, unpack | Output as JSON |
| `--compare-codecs` | bench | Compare all available codecs |
| `--compare-zipnn` | bench | Compare with ZipNN (if installed) |

---

## Codecs

KMC v0.7 supports six codecs, selected per-block for optimal results:

| Codec | Type | Best For | Description |
|-------|------|----------|-------------|
| `auto` | Selector | General use | Tries candidates per dtype, picks smallest result |
| `floatplane` | Tensor-aware | FP32/BF16/FP16 | Sign/exponent/mantissa bit-level separation |
| `byteplane` | Tensor-aware | FP32/BF16/FP16 | Byte-plane separation by position within element |
| `zstd` | General | Mixed data | High-ratio general-purpose compression |
| `zlib` | General | Fallback | Always available, decent compression |
| `raw` | Passthrough | Incompressible | No compression, used when compression expands data |

### Automatic Codec Selection

When `--codec auto` (default), the selector chooses per-block based on tensor dtype:

| Tensor dtype | Candidate chain |
|-------------|-----------------|
| FP32, BF16, FP16 | `floatplane → byteplane → zstd → zlib → raw` |
| Quantized (Q4_K, Q8_0, etc.) | `zstd → zlib → raw` |
| Unknown / non-float | `zstd → zlib → raw` |

With `--gguf-aware`, quantized GGUF tensors skip float-aware transforms automatically.

---

## Archive Format

The `.kmc` format is designed for verifiable, block-oriented storage:

```
┌─────────────────────────────────────────┐
│  Magic: "KMC\x00\x01\x00\x00\x00"  8B │
├─────────────────────────────────────────┤
│  Manifest length: uint64 BE         8B │
├─────────────────────────────────────────┤
│  Manifest: JSON (UTF-8)        Variable│
│   - version, tool info                 │
│   - file entries with paths & hashes   │
│   - block entries with codecs          │
│   - per-block codec_metadata (v3+)     │
│   - tensor entries (v2+, optional)     │
│   - artifact_type (v4+)                │
│   - artifact_metadata (v4+)            │
│   - format_metadata (v4+)              │
├─────────────────────────────────────────┤
│  Block data: concatenated       Variable│
│   - Each block independently compressed │
│   - SHA-256 verified per block         │
└─────────────────────────────────────────┘
```

See [FORMAT_SPEC.md](docs/FORMAT_SPEC.md) for the complete specification.

---

## Architecture

```
src/kmc/
├── archive.py              # Core pack/unpack/verify with security checks
├── benchmark.py            # Performance benchmarking with codec comparison
├── cli.py                  # Command-line interface
├── hashing.py              # SHA-256 integrity hashing
├── inspector.py            # AI model format detection with metadata
├── manifest.py             # KMC manifest (v6: index, archive_offset)
├── reader.py               # KMCReader partial-access API (v0.7+)
├── gguf.py                 # Legacy GGUF module (see formats/gguf.py)
├── tensor_inspector.py     # Legacy safetensors metadata (see formats/)
├── codecs/
│   ├── __init__.py         # Public codec API
│   ├── base.py             # Codec protocol, CodecContext, CodecResult
│   ├── byteplane.py        # BytePlane codec (byte-plane separation)
│   ├── floatplane.py       # FloatPlane codec (sign/exp/mantissa separation)
│   ├── registry.py         # Codec registry (discover/instantiate by name)
│   ├── selector.py         # Automatic codec selector (dtype-based candidates)
│   ├── legacy.py           # Legacy CodecId/compress_block API (v0.2/v0.3 compat)
│   ├── raw.py              # Raw passthrough codec
│   ├── zlib_codec.py       # zlib codec
│   └── zstd_codec.py       # zstd codec
├── formats/
│   ├── __init__.py         # Format module registry
│   ├── safetensors.py      # Safetensors metadata, shards, LoRA detection
│   └── gguf.py             # GGUF header + tensor metadata parsing (v0.5+)
├── index/
│   ├── __init__.py         # Index module exports
│   ├── block_index.py      # BlockIndex: block ID -> BlockLocation
│   ├── file_index.py       # FileIndex: file path -> FileLocation
│   └── tensor_index.py     # TensorIndex: tensor name -> TensorLocation
├── loaders/
│   ├── __init__.py         # Loader module exports
│   └── safetensors_loader.py  # Experimental tensor-byte loader (v0.7+)
├── workflows/
│   ├── __init__.py         # Workflow module registry
│   ├── lora.py             # LoRA/PEFT adapter detection and packing
│   └── checkpoint.py       # Training checkpoint detection and packing
└── integrations/
    └── kimari.py           # Kimari CLI integration adapters
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design decisions.

---

## Partial Access

KMC v0.7 introduces partial access features that allow reading specific files and tensors from `.kmc` archives without full decompression. This is powered by the `KMCReader` Python API and the `--only`/`--tensor`/`--list` CLI flags.

### Python API

```python
from kmc.reader import KMCReader

with KMCReader("model.kmc") as reader:
    # List contents
    files = reader.list_files()
    tensors = reader.list_tensors()

    # Read specific files without full decompression
    config = reader.read_file("config.json")
    weight_bytes = reader.read_tensor("model.layers.0.mlp.down_proj.weight")

    # Extract to disk
    reader.extract_file("config.json", "./output/")
```

### CLI Selective Extraction

```bash
# List archive contents
kmc list model.kmc

# Extract only JSON files
kmc unpack model.kmc ./output --only "*.json"

# Extract a specific tensor
kmc unpack model.kmc ./output --tensor "transformer.h.0.attn.weight"

# List before extracting
kmc unpack model.kmc ./output --list
```

**Important:** Partial access decompresses the requested data before returning it. It does NOT reduce VRAM during inference. Tensor extraction requires archives created with `--tensor-aware` mode.

See [PARTIAL_ACCESS.md](docs/PARTIAL_ACCESS.md) and [KMC_READER_API.md](docs/KMC_READER_API.md) for details.

---

## Security

KMC takes extraction security seriously:

- **Path traversal protection** — All file paths validated before extraction; `..`, absolute paths, null bytes, and control characters are rejected
- **Symlink protection** — Refuses to overwrite existing symlinks during unpack
- **Duplicate path detection** — Manifests with duplicate file paths are rejected
- **Manifest size limits** — Oversized manifests rejected to prevent DoS
- **Block hash verification** — Every block verified against SHA-256 hash
- **File hash verification** — Reconstructed files verified against SHA-256 hash
- **No pickle deserialization** — Pickle-based files detected and compressed as raw bytes only

See [SECURITY_MODEL.md](docs/SECURITY_MODEL.md) for the complete security model.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Design decisions and module structure |
| [Format Specification](docs/FORMAT_SPEC.md) | Complete `.kmc` format spec (v6) |
| [Security Model](docs/SECURITY_MODEL.md) | Threat model and mitigations |
| [Partial Access](docs/PARTIAL_ACCESS.md) | Partial access features and architecture |
| [KMCReader API](docs/KMC_READER_API.md) | Python API reference for partial access |
| [Selective Extraction](docs/SELECTIVE_EXTRACTION.md) | CLI selective extraction guide |
| [Experimental Loaders](docs/EXPERIMENTAL_LOADERS.md) | Safetensors tensor loader documentation |
| [GGUF Support](docs/GGUF_SUPPORT.md) | GGUF parsing and `--gguf-aware` mode |
| [LoRA Workflow](docs/LORA_WORKFLOW.md) | LoRA adapter compression and inspection |
| [Checkpoint Workflow](docs/CHECKPOINT_WORKFLOW.md) | Training checkpoint compression and inspection |
| [Hugging Face Workflow](docs/HUGGINGFACE_WORKFLOW.md) | Working with Hugging Face models |
| [Real Model Benchmark](docs/REAL_MODEL_BENCHMARK.md) | Running benchmarks with HuggingFace models |
| [Kimari Integration](docs/KIMARI_INTEGRATION.md) | Integration with Kimari CLI |
| [Benchmark Plan](docs/BENCHMARK_PLAN.md) | Performance testing strategy |
| [Research Notes](docs/RESEARCH_NOTES.md) | Technical references and codec design rationale |
| [Roadmap](docs/ROADMAP.md) | Development priorities |
| [Changelog](CHANGELOG.md) | Version history |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -q

# Lint
ruff check .

# Format check
ruff format --check .

# CLI help
kmc --help
kmc pack-lora --help
kmc pack-checkpoint --help

# Create demo model and test
python scripts/create_demo_model.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## Technical Foundation

KMC's approach is informed by research and industry practice:

- **ZipNN** (IBM Research) — Demonstrates that lossless compression specific to AI models can save ~1/3 of size on popular models, and >50% in some cases, without changing weights.
- **safetensors** (Hugging Face) — Treated as the priority format because it's secure, fast, and avoids `pickle` vulnerabilities.
- **GGUF** (llama.cpp) — The standard binary format for quantized models. KMC v0.5 adds full tensor metadata parsing and experimental GGUF-aware compression.
- **NetZIP** (IBM Research) — Explores lossless compression for gradients and activations in distributed training — a research direction documented for KMC's roadmap.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
