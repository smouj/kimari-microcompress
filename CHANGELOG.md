# Changelog

All notable changes to Kimari MicroCompress are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0-alpha] — 2026-05-16

### Added

- **GGUF tensor metadata parser** (`src/kmc/formats/gguf.py`) — Full tensor info extraction including name, shape, GGML type, offset, and estimated byte size; quantization summary (e.g., Q4_K: 201, F32: 1); `GGUFTensorInfo` and `GGUFInfo` dataclasses with safe degradation and warnings
- **Experimental `--gguf-aware` compression mode** — Adjusts codec selection for quantized GGUF tensors (skips floatplane/byteplane on quantized blocks, uses zstd/zlib/raw); records GGUF format metadata in the manifest; quantization-aware candidate chains
- **LoRA/PEFT adapter workflow** (`src/kmc/workflows/lora.py`) — `kmc pack-lora` command with automatic LoRA detection; `kmc inspect --lora` flag for adapter-specific inspection; detection via adapter_model.safetensors, adapter_config.json, and tensor name patterns (lora_A/lora_B); rank inference from tensor shapes; manifest records artifact_type=lora_adapter with base_model, peft_type, rank, and target_modules
- **Training checkpoint workflow** (`src/kmc/workflows/checkpoint.py`) — `kmc pack-checkpoint` command with automatic checkpoint detection; `kmc inspect --checkpoint` flag; detects optimizer.pt, scheduler.pt, rng_state.pth, trainer_state.json, pytorch_model.bin; step detection from directory name, global_step.json, or trainer_state.json; pickle-based files are never deserialized (presence/size/hash only)
- **Manifest v4** — New top-level fields: `artifact_type` (huggingface_model, gguf_model, lora_adapter, training_checkpoint, unknown), `artifact_metadata` (artifact-specific schema), `format_metadata` (safetensors and GGUF sub-objects); backward compatible with v0.2/v0.3/v0.4 manifests
- **Artifact auto-detection** — Automatic classification during pack: LoRA adapter → gguf_model → training_checkpoint → huggingface_model → unknown
- **Improved `kmc inspect`** — New flags: `--lora`, `--checkpoint`, `--gguf`, `--tensors`; auto-detect artifact type when no flag is specified; JSON output with structured metadata
- **Real small-model benchmark script** (`scripts/bench_small_hf_model.py`) — Reproducible benchmarking on locally-downloaded HuggingFace models; no auto-download; generates JSON + Markdown table; includes environment metadata (Python, OS, CPU, RAM, versions); honest results only
- **Kimari CLI integration preparation** — New adapter wrappers: `kimari_pack_lora`, `kimari_pack_checkpoint`, `kimari_inspect_model`, `kimari_compress_model`; updated `KIMARI_COMMAND_MAP` with new commands
- **Documentation** — New docs: GGUF_SUPPORT.md, LORA_WORKFLOW.md, CHECKPOINT_WORKFLOW.md; updated: ARCHITECTURE.md, FORMAT_SPEC.md, ROADMAP.md, KIMARI_INTEGRATION.md, HUGGINGFACE_WORKFLOW.md, RESEARCH_NOTES.md

### Changed

- **Version bumped** to 0.5.0-alpha in pyproject.toml and __init__.py
- **`src/kmc/manifest.py`** — Extended with artifact_type, artifact_metadata, format_metadata fields; backward compatibility with v0.2/v0.3/v0.4
- **`src/kmc/archive.py`** — Added `--gguf-aware` mode support; artifact auto-detection logic; GGUF format metadata extraction
- **`src/kmc/inspector.py`** — Added --lora, --checkpoint, --gguf, --tensors flags; auto-detect artifact type
- **`src/kmc/cli.py`** — Added pack-lora, pack-checkpoint subcommands; added --gguf-aware to pack; added --lora/--checkpoint/--gguf/--tensors to inspect
- **`src/kmc/integrations/kimari.py`** — Added kimari_pack_lora, kimari_pack_checkpoint, kimari_inspect_model, kimari_compress_model wrappers

### Tests

- 228 tests passing (up from 161 in v0.4.0-alpha)
- New test modules: test_gguf.py, test_lora_workflow.py, test_checkpoint_workflow.py, test_v05_manifest.py, test_v05_cli.py
- GGUF tests: synthetic valid/invalid files, tensor count, metadata KV count, quantization summary, inspect
- LoRA tests: valid adapter, config with base model, incomplete config, pack roundtrip, inspect JSON
- Checkpoint tests: detect checkpoint-1000, detect optimizer/scheduler/rng, pack roundtrip, inspect
- Manifest tests: artifact_type default, backward compat v0.2/v0.3/v0.4, format_metadata GGUF/safetensors
- CLI tests: help for new subcommands, inspect flags

## [0.4.0-alpha] — 2026-05-14

### Added

- **Tensor-aware block codecs** — BytePlane codec (byte-plane separation for BF16/FP16/FP32); FloatPlane codec (sign/exponent/mantissa bit-level separation); automatic codec selector with dtype-based candidate chains
- **Per-block codec metadata** (Manifest v3) — codec_metadata with transform type, element_size, inner_codec, planes, n_elements
- **`--codec` CLI flag** — Select codec: auto, byteplane, floatplane, zstd, zlib, raw
- **`--compare-codecs` benchmark flag** — Compare all available codecs on the same data
- **`--compression` inspect flag** — Show compression summary with codec usage per file
- **Codec subpackage** (`src/kmc/codecs/`) — Structured architecture with Codec protocol, CodecContext, CodecResult, registry, and selector
- **Real model benchmark script** (`scripts/bench_small_hf_model.py`)
- **161 tests passing**

### Changed

- Version bumped to 0.4.0-alpha
- Extended manifest to v3 with per-block codec_metadata
- Updated CLI with new flags and codec options

## [0.3.0-alpha] — 2026-05-12

### Added

- **Safetensors format support** — Dedicated `src/kmc/formats/safetensors.py` module; read header without loading weights; extract tensor metadata (name, dtype, shape, offsets, byte size); detect sharded models; detect LoRA/PEFT adapters with rank and target modules; graceful degradation when safetensors package is not installed
- **Tensor-aware packing mode** (`--tensor-aware`) — Align block boundaries with tensor boundaries; tensor entries in manifest (v2 format)
- **GGUF header parsing** — Read magic, version, endianness, tensor count, metadata KV count; synthetic GGUF test files
- **Enhanced inspector** — `--json` and `--tensors` CLI flags; directory-level model inspection (detected type, sharding, LoRA, etc.)
- **ZipNN benchmark comparison** (`--compare-zipnn`) — Side-by-side comparison with honest results
- **Environment metadata in benchmarks** — Python version, OS, CPU, RAM, tool versions
- **Kimari CLI integration adapters** — compress, decompress, verify, bench, inspect
- **Hugging Face workflow documentation**
- **Backward compatible with .kmc v0.2**

### Changed

- Version bumped to 0.3.0-alpha
- Manifest v2 with optional tensor entries

## [0.2.0] — 2026-05-10

### Added

- **Security hardening** — `safe_join_extract_path()` function; comprehensive path traversal tests; manifest size limits (100 MB max); manifest validation (duplicate paths, unsupported codecs, size mismatches); symlink overwrite protection
- **Full verification report** (`verify_full()`) — Structured output with block and file hash checks
- **Realistic benchmark system** — Codec comparison; JSON output and file export; synthetic data flag
- **Enhanced inspector** — LoRA adapters, tokenizer files, config files, model shards; safetensors real tensor metadata (names, shapes, dtypes, param counts); GGUF header: version, tensor count, metadata KV count
- **Kimari CLI integration adapters**
- **Documented limitations** in README and ROADMAP

### Changed

- Version bumped to 0.2.0
- Improved security model documentation

## [0.1.0] — 2026-05-08

### Added

- Initial release of Kimari MicroCompress
- `.kmc` archive format with JSON manifest
- `kmc pack`, `kmc unpack`, `kmc verify`, `kmc inspect`, `kmc bench` CLI commands
- zstd / zlib / raw codec selection per-block
- SHA-256 integrity verification at file and block level
- 256 KiB micro-blocks (configurable)
- AI model format detection (safetensors, GGUF, .bin, .pt, .ckpt)
- Safetensors tensor metadata parsing
- GGUF header parsing
- Path traversal protection in `unpack()`
- Basic roundtrip and manifest tests
- CI with GitHub Actions (pytest + ruff)
- MIT License
