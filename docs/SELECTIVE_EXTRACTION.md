# Selective Extraction

## Overview

KMC v0.7.0-alpha introduces selective extraction features that allow extracting specific files or tensors from a `.kmc` archive without decompressing everything. Selective extraction is available through both the CLI (`kmc unpack` with new flags) and the Python API (`KMCReader`). This guide focuses on the CLI interface; for the Python API, see [KMC_READER_API.md](KMC_READER_API.md).

Selective extraction is useful in many practical scenarios: you might want to extract only the configuration files to inspect model metadata, pull out a single tensor for debugging, or retrieve the tokenizer without waiting for the full model to decompress. For large archives containing multi-gigabyte model files, selective extraction can save significant time and disk space.

**Important warnings:**

- **KMC does NOT perform compressed inference.** Extracted data is fully decompressed before being written to disk or returned to the caller.
- **Tensor extraction requires `--tensor-aware` packing.** Archives created without `--tensor-aware` mode do not have tensor metadata and can only support file-level selective extraction.
- **Partial tensor loading returns bytes.** Extracted tensors are written as raw `.bin` files. To convert them to native tensor objects, you need PyTorch or NumPy installed and must handle the dtype/shape conversion yourself.

## CLI Commands

### kmc unpack --only

The `--only` flag extracts only files matching a specified pattern. The pattern supports fnmatch-style glob syntax, allowing you to match specific filenames or file types. This is the primary mechanism for file-level selective extraction.

```bash
# Extract only config.json
kmc unpack model.kmc ./output --only config.json

# Extract all JSON files
kmc unpack model.kmc ./output --only "*.json"

# Extract all tokenizer files
kmc unpack model.kmc ./output --only "tokenizer*"

# Extract with JSON output for scripting
kmc unpack model.kmc ./output --only "*.json" --json
```

When `--only` is specified, KMC uses `KMCReader` internally to read only the blocks needed for the matching files. Files that do not match the pattern are skipped entirely, avoiding unnecessary decompression.

The `--only` flag matches against both the full relative path and the filename component. This means `--only "config.json"` will match `config.json` at the archive root, and `--only "*.json"` will match `config.json`, `tokenizer.json`, and any other JSON files in the archive.

When combined with `--json`, the output includes structured information about which files were extracted and which were skipped:

```json
{
  "extracted": {
    "config.json": {"size": 256, "hash": "sha256:abc..."},
    "tokenizer.json": {"size": 1024, "hash": "sha256:def..."}
  },
  "skipped": 3,
  "total_files": 5
}
```

### kmc unpack --tensor

The `--tensor` flag extracts a single tensor by name from the archive. The tensor's raw bytes are written to a file in the output directory. This flag requires that the archive was created with `--tensor-aware` mode; otherwise, tensor metadata is not available and the extraction will fail.

```bash
# Extract a specific tensor
kmc unpack model.kmc ./output --tensor "transformer.h.0.attn.c_attn.weight"

# Extract with JSON output
kmc unpack model.kmc ./output --tensor "transformer.h.0.attn.c_attn.weight" --json
```

The extracted tensor is written to a file named after the tensor with a `.bin` extension. Special characters in tensor names (slashes, backslashes, colons) are replaced with underscores. For example, `transformer.h.0/attn.weight` becomes `transformer.h.0_attn.weight.bin`.

### kmc unpack --list

The `--list` flag lists the available files and tensors in the archive without extracting anything. This is useful for discovering what is inside an archive before deciding what to extract.

```bash
# List available files and tensors
kmc unpack model.kmc ./output --list

# List with JSON output
kmc unpack model.kmc ./output --list --json
```

The `--list` flag can be combined with `--json` for programmatic consumption. The JSON output includes separate arrays for files and tensors:

```json
{
  "files": ["config.json", "tokenizer.json", "model.safetensors"],
  "tensors": ["transformer.h.0.attn.c_attn.weight", "transformer.h.0.attn.c_proj.weight"]
}
```

If the archive was not created with `--tensor-aware` mode, the `tensors` array will be empty.

### kmc list

The `kmc list` command is a dedicated command for listing archive contents. It provides more detailed output than `kmc unpack --list`, including file sizes, tensor dtypes, and shapes.

```bash
# List all contents (files and tensors)
kmc list model.kmc

# List only files
kmc list model.kmc --files

# List only tensors
kmc list model.kmc --tensors

# List with JSON output
kmc list model.kmc --json
```

The human-readable output separates files and tensors, showing file sizes for files and dtype/shape for tensors:

```
KMC Archive Contents

Files:
  config.json (23 bytes)
  tokenizer.json (1.23 KB)
  model.safetensors (2.45 GB)

Tensors:
  transformer.h.0.attn.c_attn.weight  BF16  [768, 2304]
  transformer.h.0.attn.c_proj.weight  BF16  [2304, 768]
```

The `--json` output provides structured data suitable for scripting:

```json
{
  "archive": "model.kmc",
  "version": 6,
  "files": [
    {"path": "config.json", "size": 23, "sha256": "sha256:abc..."},
    {"path": "model.safetensors", "size": 2621440000, "sha256": "sha256:def..."}
  ],
  "tensors": [
    {"name": "transformer.h.0.attn.c_attn.weight", "file_path": "model.safetensors", "dtype": "BF16", "shape": [768, 2304]}
  ]
}
```

## Common Workflows

### Inspect Before Extracting

A common workflow is to first list the contents of an archive, then selectively extract what you need:

```bash
# Step 1: See what's in the archive
kmc list model.kmc

# Step 2: Extract only the config
kmc unpack model.kmc ./output --only config.json

# Step 3: Check the config
cat ./output/config.json

# Step 4: If you need a specific tensor, extract it
kmc unpack model.kmc ./output --tensor "transformer.h.0.attn.c_attn.weight"
```

### Extract Configuration Files Only

When you only need to inspect model metadata without downloading or decompressing the full model:

```bash
kmc unpack model.kmc ./metadata --only "*.json"
```

This is especially useful for inspecting Hugging Face model configurations, tokenizer settings, or training arguments without touching the weight files.

### Extract a Specific Tensor for Debugging

When debugging a specific layer's weights:

```bash
# First, find the tensor name
kmc list model.kmc --tensors

# Then extract it
kmc unpack model.kmc ./debug --tensor "model.layers.0.mlp.down_proj.weight"
```

### Benchmark Partial Access Performance

Use `kmc bench --partial-access` to measure how fast partial reads are on your archive:

```bash
# Benchmark partial access on an existing archive
kmc bench model.kmc /dev/null --partial-access

# Benchmark reading a specific file pattern
kmc bench model.kmc /dev/null --partial-access --only "*.json"

# Benchmark reading a specific tensor
kmc bench model.kmc /dev/null --partial-access --tensor "model.layers.0.mlp.down_proj.weight"

# JSON output for scripting
kmc bench model.kmc /dev/null --partial-access --json
```

The benchmark reports archive open time (including index construction), single-file read time, and (if applicable) tensor read time. This helps you understand the performance characteristics of partial access on your specific hardware and archive sizes.

## Security Considerations

Selective extraction applies the same security protections as full unpacking:

1. **Path traversal protection.** The `--only` flag is treated as a matching pattern, not a filesystem path. Patterns containing `..` are rejected. Extracted files are validated to ensure they write within the output directory.

2. **Block checksum verification.** Every block read during selective extraction has its SHA-256 checksum verified against the manifest. If a checksum does not match, the extraction fails with an error.

3. **File hash verification.** For file-level reads, the reconstructed file's SHA-256 hash is verified against the manifest value after decompression and concatenation.

4. **No pickle deserialization.** Selective extraction reads raw bytes and writes raw bytes. It never deserializes pickle-based files, regardless of the file type.

5. **Safe tensor filenames.** Tensor names may contain characters that are unsafe for filesystem paths (slashes, colons, etc.). The `extract_tensor` method sanitizes these characters before writing to disk.

## Limitations

1. **Only one --only pattern.** The current implementation accepts a single pattern for `--only`. To extract multiple non-contiguous patterns, run the command multiple times or extract the full archive.

2. **No --only and --tensor together.** The `--only` and `--tensor` flags are mutually exclusive. If you need both file-level and tensor-level extraction, run separate commands.

3. **Tensor extraction requires tensor metadata.** Archives packed without `--tensor-aware` mode cannot use `--tensor` because the manifest lacks the tensor-to-block mapping needed to locate tensor data.

4. **Pattern matching is filename-level.** The `--only` pattern matches against both the full relative path and the filename component, but does not currently support recursive directory patterns like `**/*.json`.

5. **No progress reporting for selective extraction.** Unlike full unpacking, selective extraction does not currently support progress reporting. This may be added in a future version.

## See Also

- [KMC_READER_API.md](KMC_READER_API.md) -- Python API reference for programmatic selective extraction
- [PARTIAL_ACCESS.md](PARTIAL_ACCESS.md) -- Overview of partial access architecture
- [EXPERIMENTAL_LOADERS.md](EXPERIMENTAL_LOADERS.md) -- Safetensors tensor loader for converting bytes to native tensors
- [SECURITY_MODEL.md](SECURITY_MODEL.md) -- Security considerations for archive operations
