"""Training checkpoint workflow: detect, pack, unpack, and inspect.

Provides dedicated support for Hugging Face training checkpoint
directories, which typically contain:
    - trainer_state.json
    - training_args.bin (pickle-based, NOT loaded)
    - optimizer.pt (pickle-based, NOT loaded)
    - scheduler.pt (pickle-based, NOT loaded)
    - rng_state.pth (pickle-based, NOT loaded)
    - pytorch_model.bin (pickle-based, NOT loaded)
    - model.safetensors (safe to read metadata)
    - Various config and tokenizer files

Strict rules:
    - Never use pickle for inspection.
    - Never load optimizer.pt, training_args.bin, or any pickle file.
    - Only detect presence, size, and hash of pickle-based files.
    - Compression uses normal or tensor-aware mode for safetensors.
    - Pack/unpack must be byte-for-byte reversible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Known checkpoint file patterns
CHECKPOINT_FILES = {
    "trainer_state.json": "trainer_state",
    "training_args.bin": "training_args",
    "optimizer.pt": "optimizer",
    "optimizer_state.pt": "optimizer",
    "scheduler.pt": "scheduler",
    "scaler.pt": "scaler",
    "rng_state.pth": "rng_state",
    "rng_state_0.pth": "rng_state",
    "pytorch_model.bin": "pytorch_model",
    "model.safetensors": "safetensors_model",
    "config.json": "config",
    "generation_config.json": "generation_config",
    "tokenizer_config.json": "tokenizer_config",
    "special_tokens_map.json": "special_tokens_map",
    "tokenizer.json": "tokenizer",
    "vocab.json": "vocab",
    "merges.txt": "merges",
    "tokenizer.model": "sentencepiece_tokenizer",
    "global_step.json": "global_step",
    "state.json": "state",
    "README.md": "readme",
}

# Pickle-based files that must NEVER be loaded/deserialized
PICKLE_FILES = {
    "training_args.bin",
    "optimizer.pt",
    "optimizer_state.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng_state.pth",
    "rng_state_0.pth",
    "pytorch_model.bin",
}


@dataclass
class CheckpointInfo:
    """Detected training checkpoint information.

    Attributes:
        is_checkpoint: Whether the directory appears to be a training checkpoint.
        step: Training step number (inferred from directory name or global_step.json).
        has_trainer_state: Whether trainer_state.json exists.
        has_optimizer_state: Whether optimizer.pt exists.
        has_scheduler_state: Whether scheduler.pt exists.
        has_rng_state: Whether rng_state.pth exists.
        has_safetensors_model: Whether model.safetensors exists.
        has_pytorch_model: Whether pytorch_model.bin exists.
        detected_files: Dict mapping filename to category.
        warnings: List of warnings encountered during detection.
    """

    is_checkpoint: bool = False
    step: int | None = None
    has_trainer_state: bool = False
    has_optimizer_state: bool = False
    has_scheduler_state: bool = False
    has_rng_state: bool = False
    has_safetensors_model: bool = False
    has_pytorch_model: bool = False
    detected_files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def detect_checkpoint(directory: Path) -> CheckpointInfo:
    """Detect training checkpoint files in a directory.

    Looks for files matching known checkpoint patterns. Detects the step
    number from the directory name (e.g., checkpoint-1000) or from
    global_step.json. Never loads pickle-based files.

    Args:
        directory: Directory to scan for checkpoint files.

    Returns:
        CheckpointInfo with detection results.
    """
    directory = Path(directory)
    warnings: list[str] = []
    detected_files: dict[str, str] = {}

    has_trainer_state = False
    has_optimizer_state = False
    has_scheduler_state = False
    has_rng_state = False
    has_safetensors_model = False
    has_pytorch_model = False
    step: int | None = None

    # Try to infer step from directory name (e.g., checkpoint-1000)
    dir_name = directory.name
    if dir_name.startswith("checkpoint-"):
        try:
            step = int(dir_name.split("-", 1)[1])
        except ValueError:
            pass

    # Scan directory for known checkpoint files
    if directory.is_dir():
        for f in directory.iterdir():
            if not f.is_file():
                continue
            fname = f.name
            if fname in CHECKPOINT_FILES:
                detected_files[fname] = CHECKPOINT_FILES[fname]

                if fname == "trainer_state.json":
                    has_trainer_state = True
                elif fname in ("optimizer.pt", "optimizer_state.pt"):
                    has_optimizer_state = True
                elif fname == "scheduler.pt":
                    has_scheduler_state = True
                elif fname in ("rng_state.pth", "rng_state_0.pth"):
                    has_rng_state = True
                elif fname == "model.safetensors":
                    has_safetensors_model = True
                elif fname == "pytorch_model.bin":
                    has_pytorch_model = True

        # Also check for safetensors shard files
        for f in directory.iterdir():
            if f.is_file() and f.name.startswith("model-") and f.name.endswith(".safetensors"):
                detected_files[f.name] = "safetensors_shard"
                has_safetensors_model = True

    # Try to get step from global_step.json
    if step is None:
        global_step_file = directory / "global_step.json"
        if global_step_file.is_file():
            try:
                data = json.loads(global_step_file.read_text(encoding="utf-8"))
                step = data.get("global_step", None)
                if isinstance(step, str):
                    step = int(step)
            except (json.JSONDecodeError, OSError, ValueError):
                pass

    # Try to get step from trainer_state.json
    if step is None and has_trainer_state:
        try:
            trainer_state_file = directory / "trainer_state.json"
            data = json.loads(trainer_state_file.read_text(encoding="utf-8"))
            step = data.get("global_step", None)
            if isinstance(step, str):
                step = int(step)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    # Determine if this is a checkpoint
    # A checkpoint should have at least a model file or trainer state
    is_checkpoint = (
        has_trainer_state
        or has_safetensors_model
        or has_pytorch_model
        or has_optimizer_state
        or bool(detected_files)
    )

    if has_pytorch_model:
        warnings.append(
            "pytorch_model.bin detected (pickle-based). "
            "Only size/hash will be recorded; contents will NOT be deserialized."
        )

    if has_optimizer_state:
        warnings.append(
            "optimizer.pt detected (pickle-based). "
            "Only size/hash will be recorded; contents will NOT be deserialized."
        )

    return CheckpointInfo(
        is_checkpoint=is_checkpoint,
        step=step,
        has_trainer_state=has_trainer_state,
        has_optimizer_state=has_optimizer_state,
        has_scheduler_state=has_scheduler_state,
        has_rng_state=has_rng_state,
        has_safetensors_model=has_safetensors_model,
        has_pytorch_model=has_pytorch_model,
        detected_files=detected_files,
        warnings=warnings,
    )


def build_checkpoint_manifest_metadata(ckpt_info: CheckpointInfo) -> dict:
    """Build manifest artifact_metadata for a training checkpoint.

    Returns a dict suitable for the artifact_metadata field in KMCManifest.

    Args:
        ckpt_info: Detected checkpoint information.

    Returns:
        Dict with checkpoint-specific metadata.
    """
    metadata: dict = {
        "artifact_type": "training_checkpoint",
    }
    if ckpt_info.step is not None:
        metadata["step"] = ckpt_info.step
    metadata["has_optimizer_state"] = ckpt_info.has_optimizer_state
    metadata["has_scheduler_state"] = ckpt_info.has_scheduler_state
    metadata["has_rng_state"] = ckpt_info.has_rng_state
    metadata["has_trainer_state"] = ckpt_info.has_trainer_state
    return metadata
