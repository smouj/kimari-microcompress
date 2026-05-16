# Changelog

All notable changes to Kimari MicroCompress are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0-alpha] — 2026-05-18

### Added

- **Partial access indexes** (`src/kmc/index/`) — BlockIndex, FileIndex, and TensorIndex classes that map between archive blocks, files, and tensors, enabling selective extraction without decompressing the entire archive. BlockIndex reconstructs offsets from older manifests (pre-v0.7) automatically when `archive_offset` is not set.
- **KMCReader Python API** (`src/kmc/reader.py`) — Read-only partial-access interface for `.kmc` archives. Supports context manager usage, file listing, tensor listing, reading individual files (`read_file()`), reading byte ranges (`read_file_range()`), reading individual tensors (`read_tensor()`), and extracting to disk (`extract_file()`, `extract_tensor()`). All reads verify block checksums and file hashes.
- **Selective extraction CLI** — `kmc unpack --only PATTERN` for file-level selective extraction with fnmatch glob support; `kmc unpack --tensor NAME` for tensor-level selective extraction; `kmc unpack --list` to list available files and tensors without extracting. All selective extraction commands support `--json` output.
- **`kmc list` command** — Dedicated command for listing archive contents with `--files`, `--tensors`, and `--json` flags. Shows file sizes, tensor dtypes, and tensor shapes in human-readable or structured JSON format.
- **Experimental safetensors tensor loader** (`src/kmc/loaders/`) — `load_tensor_bytes()` returns raw tensor bytes without optional dependencies; `load_tensor()` returns native tensor objects (PyTorch or NumPy) with automatic dtype mapping. BF16 tensors require PyTorch since NumPy does not natively support bfloat16.
- **Manifest v6** — New `index` field at the top level recording `has_block_offsets`, `has_file_index`, and `has_tensor_index`. New `archive_offset` field on `BlockEntry` for direct block access without offset reconstruction. Backward compatible with v1 through v5 manifests (missing fields default to empty/zero).
- **Partial-access benchmarks** — `kmc bench --partial-access` flag for measuring archive open time, single-file read time, and tensor read time. Supports `--only` and `--tensor` flags for targeted benchmarks. JSON output available.
- **Improved `kmc inspect`** — Shows partial access info for `.kmc` archives: block index status (native or reconstructed), file index availability, tensor index availability, selective extraction support, and tensor extraction support.
- **Security tests for partial access** — Tests for path traversal in `--only`, absolute path rejection, block checksum verification on corrupt archives, truncated archive handling, tensor name safety, and safe path joining in `extract_file()`.
- **53 new tests** for indexes, KMCReader, selective extraction, kmc list, manifest v6, partial access security, and CLI flags.

### Changed

- **Version bumped** to 0.7.0-alpha in pyproject.toml, __init__.py, and manifest.py
- **`src/kmc/manifest.py`** — Added `index` field to KMCManifest. Added `archive_offset` field to BlockEntry. Manifest version bumped to 6. Backward compatible.
- **`src/kmc/cli.py`** — Added `--only`, `--tensor`, `--list` flags to unpack command. Added `kmc list` command with `--files`, `--tensors`, `--json` flags. Added `--partial-access` flag to bench command. Added partial access info display in inspect command.
- **`src/kmc/archive.py`** — Pack function now sets `archive_offset` on block entries and `index` metadata in the manifest.

### Warnings

- **KMC does NOT perform compressed inference.** Partial access decompresses requested data before returning it.
- **Partial tensor loading returns bytes.** The `read_tensor` method returns raw bytes. To convert to native tensor objects, use the experimental safetensors loader.
- **Tensor extraction depends on tensor metadata.** Archives must be created with `--tensor-aware` mode for tensor-level partial access.
- **Older archives may support file-level partial access but not tensor-level access.** The index module reconstructs block offsets from older manifests, but tensor indexes require tensor metadata from `--tensor-aware` mode.

## [0.6.0-alpha] — 2026-05-17

### Added

- **Streaming I/O module** (`src/kmc/io/`) — Block-based file reading and writing without loading entire files into memory. `iter_file_blocks()` yields fixed-size chunks, `sha256_stream()` computes hashes via streaming reads, `write_blocks_from_iter()` accepts generators for truly streaming writes.
- **Parallel block compression** (`src/kmc/parallel.py`) — Optional parallel compression using `ThreadPoolExecutor`. `--jobs N` flag on pack/unpack/bench commands. `--jobs 1` preserves sequential behavior. Results are always deterministic regardless of worker count.
- **Progress reporting** (`src/kmc/reporting.py`) — `--progress` flag on pack, unpack, and bench commands. `ProgressReporter` class with start/update/finish methods. Automatic suppression when `--json` is used. Graceful plain-text output without requiring external dependencies.
- **Quick verification mode** — `kmc verify --quick` checks manifest and block hashes without decompressing. Significantly faster for large archives. `kmc verify --full` (default) decompresses all blocks and verifies file hashes.
- **Benchmark job comparison** — `kmc bench --compare-jobs 1,2,4,auto` flag for comparing performance across different worker counts.
- **Kimari CLI plugin integration** (`src/kmc/integrations/kimari_plugin.py`) — Clean plugin interface for Kimari CLI. `register_kimari_commands()` function registers compress, decompress, verify-compress, bench-compress, and inspect-model commands. No circular dependencies.
- **Plugin registration example** (`examples/kimari_plugin_registration.py`) — Demonstrates how to integrate KMC into a Kimari CLI application.
- **Manifest v5** — New `parallelism` field recording `created_with_jobs` and `deterministic_order`. Backward compatible with v1/v2/v3/v4 manifests.
- **49 new tests** for streaming I/O, parallel compression, progress reporting, verify modes, CLI flags, manifest v5, robustness, and Kimari plugin integration.

### Changed

- **Version bumped** to 0.6.0-alpha in pyproject.toml, __init__.py, and manifest.py
- **`src/kmc/archive.py`** — Added `jobs` and `progress_reporter` parameters to `pack()`. Added `verify_quick()` function. Manifest records `parallelism` metadata when `jobs > 1`.
- **`src/kmc/cli.py`** — Added `--jobs` flag to pack, unpack, and bench commands. Added `--progress` flag. Added `--quick`/`--full` flags to verify. Added `--compare-jobs` to bench.
- **`src/kmc/manifest.py`** — Added `parallelism` field to KMCManifest. Version bumped to 5. Backward compatible.
- **277 tests passing** (up from 228 in v0.5.0-alpha)

### Robustness

- Empty directory packing creates valid archives
- Empty file packing creates valid archives
- Unicode filename roundtrip works correctly
- Many small files packing works correctly
- Output overwrites existing files correctly
- Truncated archive detected by verify_quick
- Corrupt manifest detected by verify_full
- Parallel compression preserves deterministic block order

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
