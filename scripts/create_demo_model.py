#!/usr/bin/env python3
"""Create a synthetic demo model for testing KMC.

This generates a directory with files that simulate an AI model's structure,
including repetitive binary data that compresses well with zstd/zlib.
"""

import json
import struct
from pathlib import Path


def create_demo_model(output_dir: str = "demo-model") -> None:
    """Create a synthetic model directory with test files."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    # 1. Create a fake safetensors file with header + data
    # Header: 8 bytes length + JSON + tensor data
    header_dict = {
        "__metadata__": {"model_type": "demo", "format": "safetensors"},
        "embed_tokens.weight": {
            "dtype": "F32",
            "shape": [32000, 4096],
            "data_offsets": [0, 524288000],
        },
        "layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [4096, 4096],
            "data_offsets": [524288000, 590559232],
        },
    }
    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))

    # Create a smaller demo file (not the full 590 MB)
    tensor_data = b"\x00\x01\x02\x03" * 10000  # 40 KB
    with open(base / "model.safetensors", "wb") as f:
        f.write(header_len)
        f.write(header_json)
        f.write(tensor_data)

    # 2. Config JSON (highly compressible)
    config = {
        "architectures": ["DemoModel"],
        "model_type": "demo",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    with open(base / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 3. Tokenizer data (text-heavy, compressible)
    added_tokens = [
        {
            "id": i,
            "content": f"<token_{i}>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
        }
        for i in range(100)
    ]
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel"},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "vocab": {},
            "merges": [],
        },
    }
    with open(base / "tokenizer.json", "w") as f:
        json.dump(tokenizer, f, indent=2)

    # 4. Binary data (simulating model weights with repetitive patterns)
    weight_data = b"\xab\xcd" * 50000  # 100 KB
    with open(base / "pytorch_model-00001-of-00001.bin", "wb") as f:
        f.write(weight_data)

    # 5. Generation config
    gen_config = {
        "_from_model_config": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "transformers_version": "4.36.0",
    }
    with open(base / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    total_size = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
    print(f"Demo model created in '{base}/' ({total_size:,} bytes)")


if __name__ == "__main__":
    create_demo_model()
