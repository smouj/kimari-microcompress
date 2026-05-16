#!/usr/bin/env python3
"""Basic roundtrip example: pack, verify, and unpack using the KMC Python API."""

import tempfile
from pathlib import Path

from kmc.archive import pack, unpack, verify


def main() -> None:
    """Demonstrate basic KMC roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test file
        source_dir = tmpdir / "source"
        source_dir.mkdir()
        (source_dir / "model.bin").write_bytes(b"\x00\x01\x02\x03" * 1000)
        (source_dir / "config.json").write_bytes(b'{"model_type": "test"}')

        # Pack
        archive = tmpdir / "output.kmc"
        print(f"Packing {source_dir} -> {archive}")
        pack(source_dir, archive)

        # Verify
        print(f"Verifying {archive}")
        errors = verify(archive)
        if errors:
            print(f"Errors found: {errors}")
            return
        print("Verification passed!")

        # Unpack
        restore_dir = tmpdir / "restored"
        print(f"Unpacking {archive} -> {restore_dir}")
        unpack(archive, restore_dir)

        # Check roundtrip
        for f in source_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(source_dir)
                restored = restore_dir / rel
                if restored.read_bytes() != f.read_bytes():
                    print(f"MISMATCH: {rel}")
                    return
                print(f"  {rel}: OK")

        print("Roundtrip complete — all files match!")


if __name__ == "__main__":
    main()
