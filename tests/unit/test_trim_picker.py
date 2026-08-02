"""Unit tests for trim choice logic and persistence (musicbot.processing.trim_picker)."""

import json

import pytest

from musicbot.processing.madmom_beats import TrimCandidate
from musicbot.processing.trim_picker import (
    load_trim_choices,
    pick_trim_auto,
    save_trim_choice,
)

pytestmark = pytest.mark.unit


def _cand(t: float, pct: float) -> TrimCandidate:
    return TrimCandidate(time_sec=t, energy=pct / 100.0, energy_pct=pct)


class TestPickTrimAuto:
    def test_empty_candidates_returns_zero(self):
        assert pick_trim_auto([]) == 0.0

    def test_picks_highest_energy(self):
        cands = [_cand(10.0, 40.0), _cand(43.6, 100.0), _cand(60.0, 85.0)]
        assert pick_trim_auto(cands) == 43.6

    def test_single_candidate(self):
        assert pick_trim_auto([_cand(5.5, 12.0)]) == 5.5

    def test_first_wins_on_tie(self):
        # max() keeps the first of equal keys — earlier candidate wins ties
        cands = [_cand(10.0, 100.0), _cand(20.0, 100.0)]
        assert pick_trim_auto(cands) == 10.0


class TestChoicePersistence:
    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_trim_choices(tmp_path / "nothing.json") == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "choices.json"
        save_trim_choice("song.mp3", 43.71337, method="test", path=path)

        loaded = load_trim_choices(path)
        assert "song.mp3" in loaded
        entry = loaded["song.mp3"]
        assert entry["trim_sec"] == 43.7134  # rounded to 4 decimals
        assert entry["method"] == "test"
        assert "timestamp" in entry

    def test_save_preserves_other_entries(self, tmp_path):
        path = tmp_path / "choices.json"
        save_trim_choice("a.mp3", 1.0, path=path)
        save_trim_choice("b.mp3", 2.0, path=path)
        loaded = load_trim_choices(path)
        assert set(loaded.keys()) == {"a.mp3", "b.mp3"}

    def test_save_overwrites_same_key(self, tmp_path):
        path = tmp_path / "choices.json"
        save_trim_choice("a.mp3", 1.0, path=path)
        save_trim_choice("a.mp3", 99.0, path=path)
        assert load_trim_choices(path)["a.mp3"]["trim_sec"] == 99.0

    def test_file_is_sorted_and_indented(self, tmp_path):
        path = tmp_path / "choices.json"
        save_trim_choice("zebra.mp3", 1.0, path=path)
        save_trim_choice("apple.mp3", 2.0, path=path)
        text = path.read_text()
        # sort_keys=True means apple appears before zebra in the raw file
        assert text.index("apple") < text.index("zebra")
        assert json.loads(text)  # valid JSON
