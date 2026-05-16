"""Tests for v0.6 features: streaming I/O, parallelism, progress, verify modes, CLI flags."""

import hashlib
import os
from pathlib import Path

from kmc.archive import pack, unpack, verify_full, verify_quick
from kmc.io import iter_file_blocks, read_block_at, sha256_stream, write_blocks
from kmc.parallel import BlockWorkItem, compress_blocks_parallel, resolve_jobs
from kmc.reporting import ProgressReporter, create_reporter


# ===========================================================================
# Streaming I/O tests
# ===========================================================================


class TestStreamingIO:
    """Tests for the streaming I/O module."""

    def test_iter_file_blocks_basic(self, tmp_path: Path) -> None:
        """iter_file_blocks yields correct blocks for a small file."""
        data = b"hello world" * 100
        f = tmp_path / "test.bin"
        f.write_bytes(data)

        blocks = list(iter_file_blocks(f, block_size=256))
        assert b"".join(blocks) == data
        assert len(blocks) > 1

    def test_iter_file_blocks_single_block(self, tmp_path: Path) -> None:
        """A file smaller than block_size yields a single block."""
        data = b"short"
        f = tmp_path / "small.bin"
        f.write_bytes(data)

        blocks = list(iter_file_blocks(f, block_size=1024))
        assert blocks == [b"short"]

    def test_iter_file_blocks_empty_file(self, tmp_path: Path) -> None:
        """Empty file yields no blocks."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")

        blocks = list(iter_file_blocks(f))
        assert blocks == []

    def test_iter_file_blocks_non_multiple_size(self, tmp_path: Path) -> None:
        """File size not a multiple of block_size yields correct last block."""
        block_size = 100
        data = b"x" * 250
        f = tmp_path / "nonmult.bin"
        f.write_bytes(data)

        blocks = list(iter_file_blocks(f, block_size=block_size))
        assert len(blocks) == 3
        assert len(blocks[0]) == 100
        assert len(blocks[1]) == 100
        assert len(blocks[2]) == 50
        assert b"".join(blocks) == data

    def test_sha256_stream(self, tmp_path: Path) -> None:
        """sha256_stream matches hashlib for the same file."""
        data = b"test data for hashing" * 500
        f = tmp_path / "hash.bin"
        f.write_bytes(data)

        expected = hashlib.sha256(data).hexdigest()
        assert sha256_stream(f) == expected

    def test_sha256_stream_empty_file(self, tmp_path: Path) -> None:
        """sha256_stream works for empty files."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")

        expected = hashlib.sha256(b"").hexdigest()
        assert sha256_stream(f) == expected

    def test_read_block_at(self, tmp_path: Path) -> None:
        """read_block_at reads from the correct offset."""
        data = b"0123456789abcdef"
        f = tmp_path / "read.bin"
        f.write_bytes(data)

        assert read_block_at(f, offset=4, size=4) == b"4567"
        assert read_block_at(f, offset=0, size=8) == b"01234567"
        assert read_block_at(f, offset=12, size=10) == b"cdef"  # reads till end

    def test_write_blocks(self, tmp_path: Path) -> None:
        """write_blocks writes blocks to a file."""
        blocks = [b"hello ", b"world", b"!"]
        f = tmp_path / "output.bin"
        write_blocks(f, blocks)

        assert f.read_bytes() == b"hello world!"

    def test_write_blocks_creates_dirs(self, tmp_path: Path) -> None:
        """write_blocks creates parent directories."""
        f = tmp_path / "subdir" / "deep" / "out.bin"
        write_blocks(f, [b"data"])
        assert f.read_bytes() == b"data"

    def test_unicode_filename(self, tmp_path: Path) -> None:
        """Streaming I/O works with unicode filenames."""
        f = tmp_path / "datos_ñ.txt"
        f.write_text("hola mundo")

        blocks = list(iter_file_blocks(f, block_size=4))
        assert b"".join(blocks) == b"hola mundo"


# ===========================================================================
# Parallel compression tests
# ===========================================================================


class TestParallel:
    """Tests for parallel compression module."""

    def test_resolve_jobs_one(self) -> None:
        """resolve_jobs(1) returns 1."""
        assert resolve_jobs(1) == 1

    def test_resolve_jobs_auto(self) -> None:
        """resolve_jobs('auto') returns cpu_count (at least 1)."""
        result = resolve_jobs("auto")
        assert result >= 1

    def test_resolve_jobs_string_number(self) -> None:
        """resolve_jobs('4') returns 4."""
        assert resolve_jobs("4") == 4

    def test_resolve_jobs_invalid_string(self) -> None:
        """resolve_jobs('invalid') returns 1."""
        assert resolve_jobs("invalid") == 1

    def test_resolve_jobs_zero_becomes_one(self) -> None:
        """resolve_jobs(0) returns 1 (minimum)."""
        assert resolve_jobs(0) == 1

    def test_compress_blocks_sequential(self) -> None:
        """Sequential compression with jobs=1 produces correct results."""
        from kmc.archive import _compress_block_with_codec
        from kmc.codecs.base import CodecContext

        items = [
            BlockWorkItem(
                file_index=0,
                block_index=i,
                data=os.urandom(256),
                codec_name="raw",
                context=CodecContext(original_size=256, block_index=i),
            )
            for i in range(3)
        ]

        results = compress_blocks_parallel(items, _compress_block_with_codec, jobs=1)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.block_index == i

    def test_compress_blocks_parallel_order(self) -> None:
        """Parallel compression preserves block order."""
        from kmc.archive import _compress_block_with_codec
        from kmc.codecs.base import CodecContext

        items = [
            BlockWorkItem(
                file_index=0,
                block_index=i,
                data=bytes([i] * 100),  # deterministic data
                codec_name="raw",
                context=CodecContext(original_size=100, block_index=i),
            )
            for i in range(10)
        ]

        seq_results = compress_blocks_parallel(items, _compress_block_with_codec, jobs=1)
        par_results = compress_blocks_parallel(items, _compress_block_with_codec, jobs=2)

        # Results should be in the same order
        for s, p in zip(seq_results, par_results):
            assert s.block_index == p.block_index

    def test_compress_parallel_roundtrip(self, tmp_path: Path) -> None:
        """Archive packed with --jobs 2 roundtrips correctly."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(1000))

        output = tmp_path / "output.kmc"
        pack(source, output, jobs=2)

        restored = tmp_path / "restored"
        unpack(output, restored)

        original_data = (source / "data.bin").read_bytes()
        restored_data = (restored / "data.bin").read_bytes()
        assert original_data == restored_data

    def test_verify_parallel_archive(self, tmp_path: Path) -> None:
        """Archive created with parallel compression passes verification."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(2000))

        output = tmp_path / "output.kmc"
        pack(source, output, jobs=2)

        report = verify_full(output)
        assert report.integrity == "OK"


# ===========================================================================
# Progress reporter tests
# ===========================================================================


class TestProgressReporter:
    """Tests for progress reporting."""

    def test_reporter_creation(self) -> None:
        """ProgressReporter can be created with defaults."""
        reporter = ProgressReporter()
        assert reporter.total_blocks == 0
        assert not reporter.json_mode

    def test_create_reporter(self) -> None:
        """create_reporter factory works."""
        reporter = create_reporter(show_progress=True, total_blocks=100)
        assert reporter.total_blocks == 100
        assert reporter.show_progress

    def test_json_mode_suppresses_output(self) -> None:
        """JSON mode reporter suppresses all output."""
        reporter = create_reporter(json_mode=True, show_progress=True)
        assert reporter.json_mode
        # These should not raise even if they try to write
        reporter.start("Test")
        reporter.update(50)
        reporter.finish("done")

    def test_progress_no_crash(self) -> None:
        """Progress reporter methods don't crash."""
        reporter = ProgressReporter(total_blocks=10, show_progress=True)
        reporter.start("Test")
        reporter.update(5)
        reporter.update(10)
        reporter.finish("complete")


# ===========================================================================
# Verify --quick/--full tests
# ===========================================================================


class TestVerifyModes:
    """Tests for quick and full verification modes."""

    def test_verify_quick_passes(self, tmp_path: Path) -> None:
        """Quick verify passes for a valid archive."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(500))

        output = tmp_path / "output.kmc"
        pack(source, output)

        report = verify_quick(output)
        assert report.integrity == "OK"
        assert report.total_files == 1
        assert report.total_blocks >= 1

    def test_verify_full_passes(self, tmp_path: Path) -> None:
        """Full verify passes for a valid archive."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(500))

        output = tmp_path / "output.kmc"
        pack(source, output)

        report = verify_full(output)
        assert report.integrity == "OK"

    def test_verify_quick_faster_than_full(self, tmp_path: Path) -> None:
        """Quick verify doesn't decompress, so it should be at least as fast."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(5000))

        output = tmp_path / "output.kmc"
        pack(source, output)

        import time

        start = time.time()
        verify_quick(output)
        quick_time = time.time() - start

        start = time.time()
        verify_full(output)
        full_time = time.time() - start

        # Quick should be at most as slow as full (may be faster)
        # This is a soft check, not strict
        assert quick_time <= full_time + 0.5  # allow tolerance

    def test_verify_quick_truncated_archive(self, tmp_path: Path) -> None:
        """Quick verify detects truncated archive."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(500))

        output = tmp_path / "output.kmc"
        pack(source, output)

        # Truncate the archive
        data = output.read_bytes()
        truncated = tmp_path / "truncated.kmc"
        truncated.write_bytes(data[: len(data) // 2])

        report = verify_quick(truncated)
        assert report.integrity == "FAILED"

    def test_verify_full_corrupt_manifest(self, tmp_path: Path) -> None:
        """Full verify detects corrupted manifest."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(500))

        output = tmp_path / "output.kmc"
        pack(source, output)

        # Corrupt the manifest by modifying bytes
        data = bytearray(output.read_bytes())
        data[20] = 0xFF  # Corrupt a byte in the manifest area
        corrupt = tmp_path / "corrupt.kmc"
        corrupt.write_bytes(bytes(data))

        report = verify_full(corrupt)
        assert report.integrity == "FAILED"


# ===========================================================================
# CLI flags tests
# ===========================================================================


class TestV06CLI:
    """Tests for v0.6 CLI flags."""

    def test_pack_help_includes_jobs(self) -> None:
        """kmc pack --help shows --jobs flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "pack", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--jobs" in result.stdout

    def test_pack_help_includes_progress(self) -> None:
        """kmc pack --help shows --progress flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "pack", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--progress" in result.stdout

    def test_verify_help_includes_quick(self) -> None:
        """kmc verify --help shows --quick flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "verify", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--quick" in result.stdout

    def test_verify_help_includes_full(self) -> None:
        """kmc verify --help shows --full flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "verify", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--full" in result.stdout

    def test_bench_help_includes_compare_jobs(self) -> None:
        """kmc bench --help shows --compare-jobs flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "bench", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--compare-jobs" in result.stdout

    def test_unpack_help_includes_jobs(self) -> None:
        """kmc unpack --help shows --jobs flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "unpack", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--jobs" in result.stdout


# ===========================================================================
# Manifest v5 tests
# ===========================================================================


class TestManifestV5:
    """Tests for v0.6 manifest (v5) features."""

    def test_default_parallelism(self) -> None:
        """Default parallelism is empty dict."""
        from kmc.manifest import KMCManifest

        m = KMCManifest()
        assert m.parallelism == {}

    def test_parallelism_in_json(self) -> None:
        """Parallelism field appears in JSON serialization."""
        from kmc.manifest import KMCManifest

        m = KMCManifest(parallelism={"created_with_jobs": 4, "deterministic_order": True})
        j = m.to_json()
        import json

        data = json.loads(j)
        assert data["parallelism"]["created_with_jobs"] == 4
        assert data["parallelism"]["deterministic_order"] is True

    def test_parallelism_roundtrip(self) -> None:
        """Parallelism field survives JSON roundtrip."""
        from kmc.manifest import KMCManifest

        m = KMCManifest(parallelism={"created_with_jobs": 2, "deterministic_order": True})
        j = m.to_json()
        m2 = KMCManifest.from_json(j)
        assert m2.parallelism["created_with_jobs"] == 2

    def test_pack_with_jobs_sets_parallelism(self, tmp_path: Path) -> None:
        """Packing with jobs>1 sets parallelism in manifest."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(500))

        output = tmp_path / "output.kmc"
        pack(source, output, jobs=2)

        from kmc.archive import read_manifest_from_archive

        manifest, _ = read_manifest_from_archive(output)
        assert manifest.parallelism.get("created_with_jobs") == 2
        assert manifest.parallelism.get("deterministic_order") is True

    def test_pack_without_jobs_no_parallelism(self, tmp_path: Path) -> None:
        """Packing with jobs=1 (default) doesn't set parallelism."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(500))

        output = tmp_path / "output.kmc"
        pack(source, output, jobs=1)

        from kmc.archive import read_manifest_from_archive

        manifest, _ = read_manifest_from_archive(output)
        assert manifest.parallelism == {}

    def test_backward_compat_v4_manifest_reads(self) -> None:
        """v4 manifest (without parallelism) reads with empty parallelism."""
        import json

        from kmc.manifest import KMCManifest

        v4_json = json.dumps(
            {
                "version": 4,
                "tool": "kimari-microcompress",
                "tool_version": "0.5.0-alpha",
                "created_at": "2025-01-01",
                "total_original_size": 5000,
                "total_compressed_size": 2500,
                "artifact_type": "lora_adapter",
                "artifact_metadata": {},
                "format_metadata": {},
                "files": [],
            }
        )
        m = KMCManifest.from_json(v4_json)
        assert m.version == 4
        assert m.parallelism == {}
        assert m.artifact_type == "lora_adapter"


# ===========================================================================
# Robustness tests
# ===========================================================================


class TestRobustness:
    """Tests for edge cases and robustness."""

    def test_pack_empty_directory(self, tmp_path: Path) -> None:
        """Packing an empty directory creates a valid archive."""
        source = tmp_path / "empty_dir"
        source.mkdir()

        output = tmp_path / "output.kmc"
        pack(source, output)

        report = verify_full(output)
        assert report.integrity == "OK"
        assert report.total_files == 0

    def test_pack_empty_file(self, tmp_path: Path) -> None:
        """Packing an empty file creates a valid archive."""
        source = tmp_path / "empty.bin"
        source.write_bytes(b"")

        output = tmp_path / "output.kmc"
        pack(source, output)

        report = verify_full(output)
        assert report.integrity == "OK"

    def test_pack_many_small_files(self, tmp_path: Path) -> None:
        """Packing many small files creates a valid archive."""
        source = tmp_path / "many"
        source.mkdir()
        for i in range(50):
            (source / f"file_{i:03d}.bin").write_bytes(os.urandom(50))

        output = tmp_path / "output.kmc"
        pack(source, output)

        report = verify_full(output)
        assert report.integrity == "OK"
        assert report.total_files == 50

    def test_pack_unicode_filenames(self, tmp_path: Path) -> None:
        """Packing files with unicode names roundtrips correctly."""
        source = tmp_path / "unicode"
        source.mkdir()
        (source / "datos_ñ.txt").write_bytes(b"spanish data")
        (source / "データ.bin").write_bytes(b"japanese data")
        (source / "файл.dat").write_bytes(b"russian data")

        output = tmp_path / "output.kmc"
        pack(source, output)

        restored = tmp_path / "restored"
        unpack(output, restored)

        assert (restored / "datos_ñ.txt").read_bytes() == b"spanish data"
        assert (restored / "データ.bin").read_bytes() == b"japanese data"
        assert (restored / "файл.dat").read_bytes() == b"russian data"

    def test_output_already_exists(self, tmp_path: Path) -> None:
        """Packing over an existing output file works."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(b"original")

        output = tmp_path / "output.kmc"
        output.write_bytes(b"old data")

        pack(source, output)
        report = verify_full(output)
        assert report.integrity == "OK"

    def test_kimari_plugin_compress(self, tmp_path: Path) -> None:
        """Kimari plugin compress_model_command works."""
        from kmc.integrations.kimari_plugin import compress_model_command

        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(500))

        output = tmp_path / "output.kmc"
        result = compress_model_command(source, output)

        assert result["status"] == "ok"
        assert result["original_size"] > 0
        assert result["compressed_size"] > 0

    def test_kimari_plugin_decompress(self, tmp_path: Path) -> None:
        """Kimari plugin decompress_model_command works."""
        from kmc.integrations.kimari_plugin import (
            compress_model_command,
            decompress_model_command,
        )

        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(b"test data 123")

        archive = tmp_path / "output.kmc"
        compress_model_command(source, archive)

        restored = tmp_path / "restored"
        result = decompress_model_command(archive, restored)

        assert result["status"] == "ok"
        assert (restored / "data.bin").read_bytes() == b"test data 123"

    def test_kimari_plugin_verify(self, tmp_path: Path) -> None:
        """Kimari plugin verify_compressed_model_command works."""
        from kmc.integrations.kimari_plugin import (
            compress_model_command,
            verify_compressed_model_command,
        )

        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(500))

        archive = tmp_path / "output.kmc"
        compress_model_command(source, archive)

        result = verify_compressed_model_command(archive)
        assert result["status"] == "ok"
        assert result["integrity"] == "OK"

    def test_kimari_plugin_verify_quick(self, tmp_path: Path) -> None:
        """Kimari plugin verify with quick=True works."""
        from kmc.integrations.kimari_plugin import (
            compress_model_command,
            verify_compressed_model_command,
        )

        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(500))

        archive = tmp_path / "output.kmc"
        compress_model_command(source, archive)

        result = verify_compressed_model_command(archive, quick=True)
        assert result["status"] == "ok"
        assert result["quick"] is True
