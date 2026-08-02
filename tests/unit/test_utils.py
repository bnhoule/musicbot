"""Unit tests for shared helpers (musicbot.utils)."""

import json

import pytest

from musicbot.utils import (
    build_song_dirs,
    ensure_dir,
    parse_bpm_from_filename,
    resolve_input,
    run_id,
    sanitize_filename,
    save_json,
)

pytestmark = pytest.mark.unit


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("simple", "simple"),
            ("with space", "with_space"),
            ("slash/back\\slash", "slash_back_slash"),
            ("keep-dash_under.dot", "keep-dash_under.dot"),
            ("  trimmed  ", "trimmed"),
            ("125 - Baby Baby", "125_-_Baby_Baby"),
        ],
    )
    def test_sanitization(self, raw, expected):
        assert sanitize_filename(raw) == expected


class TestRunId:
    def test_format(self):
        rid = run_id()
        assert len(rid) == 4
        int(rid, 16)  # must be valid hex

    def test_uniqueness_over_many_draws(self):
        ids = {run_id() for _ in range(200)}
        # 4 hex chars = 65k possibilities; 200 draws colliding entirely is
        # impossible — allow a few birthday collisions.
        assert len(ids) > 190


class TestParseBpmFromFilename:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("122 - Bulletproof.mp3", 122.0),
            ("140 - Something New.mp3", 140.0),
            ("Song Name (128 BPM).flac", 128.0),
            ("artist - title 95bpm.wav", 95.0),
            ("Track - 174.mp3", 174.0),
            ("128bpm banger.mp3", 128.0),
        ],
    )
    def test_parses(self, filename, expected):
        assert parse_bpm_from_filename(filename) == expected

    @pytest.mark.parametrize(
        "filename",
        [
            "No Numbers Here.mp3",
            "300 - Out Of Range.mp3",   # 300 > max plausible BPM
            "45 - Too Slow.mp3",        # 45 < min plausible BPM
            "1999 - Prince.mp3",        # 4 digits — not a BPM pattern
        ],
    )
    def test_rejects(self, filename):
        assert parse_bpm_from_filename(filename) is None


class TestResolveInput:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_input(str(tmp_path / "nope.mp3"))

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "song.ogg"
        f.write_bytes(b"x")
        with pytest.raises(ValueError):
            resolve_input(str(f))

    def test_valid_file_resolves(self, tmp_path):
        f = tmp_path / "song.mp3"
        f.write_bytes(b"x")
        assert resolve_input(str(f)) == f.resolve()


class TestFilesystemHelpers:
    def test_ensure_dir_creates_nested(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert result.is_dir()

    def test_save_json_roundtrip(self, tmp_path):
        data = {"bpm": 125.0, "key": "F minor", "nested": {"x": [1, 2]}}
        out = tmp_path / "meta.json"
        save_json(data, out)
        assert json.loads(out.read_text()) == data

    def test_build_song_dirs_structure(self, tmp_path):
        song_dir, stems_dir = build_song_dirs(str(tmp_path), "My Song!", method_tag="madmom")
        assert stems_dir.is_dir()
        assert stems_dir.parent == song_dir
        assert stems_dir.name == "stems"
        # sanitized name + 4-hex run id + method tag
        assert song_dir.name.startswith("My_Song")
        assert song_dir.name.endswith("_madmom")

    def test_build_song_dirs_unique_per_run(self, tmp_path):
        d1, _ = build_song_dirs(str(tmp_path), "Same Song")
        d2, _ = build_song_dirs(str(tmp_path), "Same Song")
        assert d1 != d2
