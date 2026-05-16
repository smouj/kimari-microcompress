"""LoRA/PEFT adapter workflow: detect, pack, unpack, and inspect.

Provides dedicated support for LoRA/PEFT adapter directories, which
typically contain:
    - adapter_model.safetensors (weights)
    - adapter_config.json (PEFT configuration)
    - README.md (optional)
    - tokenizer files (optional)
    - Reference to base model (optional, in adapter_config.json)

No pickle is used. No weights are loaded into memory. The safetensors
file is read only for metadata; actual weight data is compressed as
raw bytes using KMC's lossless pipeline.

Strict rules:
    - Never invent data. If a field is not present, use "unknown".
    - Never use pickle for inspection.
    - Never modify weights.
    - Pack/unpack must be byte-for-byte reversible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoRAAdapterInfo:
    """Detected LoRA/PEFT adapter information.

    Attributes:
        is_lora: Whether the directory appears to be a LoRA adapter.
        adapter_model_path: Path to the adapter model file, if found.
        adapter_config_path: Path to adapter_config.json, if found.
        has_adapter_model: Whether adapter_model.safetensors exists.
        has_adapter_config: Whether adapter_config.json exists.
        has_readme: Whether README.md exists.
        base_model_name_or_path: Base model reference from config.
        peft_type: PEFT type from config (e.g., "LORA").
        lora_rank: LoRA rank (r) from config.
        target_modules: Target modules from config.
        warnings: List of warnings encountered during detection.
    """

    is_lora: bool = False
    adapter_model_path: str = ""
    adapter_config_path: str = ""
    has_adapter_model: bool = False
    has_adapter_config: bool = False
    has_readme: bool = False
    base_model_name_or_path: str = "unknown"
    peft_type: str = "unknown"
    lora_rank: int | None = None
    target_modules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_lora_adapter(directory: Path) -> LoRAAdapterInfo:
    """Detect LoRA/PEFT adapter files in a directory.

    Looks for:
        - adapter_model.safetensors
        - adapter_config.json
        - README.md

    Reads adapter_config.json for metadata if available.
    Never invents data — missing fields default to "unknown".

    Args:
        directory: Directory to scan for LoRA adapter files.

    Returns:
        LoRAAdapterInfo with detection results.
    """
    directory = Path(directory)
    warnings: list[str] = []

    # Check for adapter model files
    adapter_model = directory / "adapter_model.safetensors"
    has_adapter_model = adapter_model.is_file()

    # Also check alternative names
    if not has_adapter_model:
        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() == ".safetensors" and "adapter" in f.name.lower():
                adapter_model = f
                has_adapter_model = True
                break

    # Also check for LoRA tensors in any safetensors file
    if not has_adapter_model:
        for f in directory.rglob("*.safetensors"):
            if f.is_file():
                try:
                    from ..formats.safetensors import read_safetensors_info

                    info = read_safetensors_info(f)
                    if info.is_lora:
                        adapter_model = f
                        has_adapter_model = True
                        break
                except (ValueError, OSError):
                    pass

    # Check for adapter_config.json
    adapter_config = directory / "adapter_config.json"
    has_adapter_config = adapter_config.is_file()

    # Check for README
    readme = directory / "README.md"
    has_readme = readme.is_file()

    # Read adapter_config.json if present
    base_model_name_or_path = "unknown"
    peft_type = "unknown"
    lora_rank: int | None = None
    target_modules: list[str] = []

    if has_adapter_config:
        try:
            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_model_name_or_path = config.get("base_model_name_or_path", "unknown") or "unknown"
            peft_type = config.get("peft_type", "unknown") or "unknown"
            lora_rank = config.get("r", None)
            tm = config.get("target_modules", [])
            if isinstance(tm, str):
                target_modules = [tm]
            elif isinstance(tm, list):
                target_modules = tm
        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"Could not read adapter_config.json: {e}")

    # Try to infer LoRA rank from tensor shapes if not in config
    if has_adapter_model and lora_rank is None:
        try:
            from ..formats.safetensors import read_safetensors_info

            info = read_safetensors_info(adapter_model)
            if info.is_lora and info.lora_rank is not None:
                lora_rank = info.lora_rank
            if not target_modules and info.target_modules:
                target_modules = info.target_modules
            if base_model_name_or_path == "unknown" and info.base_model_reference:
                base_model_name_or_path = info.base_model_reference
        except (ValueError, OSError):
            pass

    is_lora = has_adapter_model or has_adapter_config

    if not has_adapter_model and has_adapter_config:
        warnings.append("adapter_config.json found but no adapter model file")

    return LoRAAdapterInfo(
        is_lora=is_lora,
        adapter_model_path=str(adapter_model) if has_adapter_model else "",
        adapter_config_path=str(adapter_config) if has_adapter_config else "",
        has_adapter_model=has_adapter_model,
        has_adapter_config=has_adapter_config,
        has_readme=has_readme,
        base_model_name_or_path=base_model_name_or_path,
        peft_type=peft_type,
        lora_rank=lora_rank,
        target_modules=target_modules,
        warnings=warnings,
    )


def build_lora_manifest_metadata(adapter_info: LoRAAdapterInfo) -> dict:
    """Build manifest artifact_metadata for a LoRA adapter.

    Returns a dict suitable for the artifact_metadata field in KMCManifest.

    Args:
        adapter_info: Detected LoRA adapter information.

    Returns:
        Dict with LoRA-specific metadata.
    """
    metadata: dict = {
        "artifact_type": "lora_adapter",
        "base_model_name_or_path": adapter_info.base_model_name_or_path,
        "peft_type": adapter_info.peft_type,
    }
    if adapter_info.lora_rank is not None:
        metadata["r"] = adapter_info.lora_rank
    if adapter_info.target_modules:
        metadata["target_modules"] = adapter_info.target_modules
    return metadata
