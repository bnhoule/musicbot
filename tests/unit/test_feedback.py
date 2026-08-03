"""Unit tests for the append-only feedback log and transform-limit learning."""

import pytest

from musicbot.feedback import load_events, log_event
from musicbot.processing.transform_limits import (
    aggregate_limits,
    check_transform,
    save_limits,
)

pytestmark = pytest.mark.unit


class TestFeedbackLog:
    def test_append_and_load(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        log_event("trim", path=path, song="Run Away", agreed=True)
        log_event("key", path=path, song="Run Away", verdict="correct")

        all_events = load_events(path=path)
        assert len(all_events) == 2
        assert all(e["git_sha"] for e in all_events)
        assert all("timestamp" in e for e in all_events)

        trim_only = load_events("trim", path=path)
        assert len(trim_only) == 1
        assert trim_only[0]["agreed"] is True

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_events(path=tmp_path / "nope.jsonl") == []

    def test_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        path.write_text('{"stage":"trim","ok":true}\nnot-json\n{"stage":"key"}\n')
        events = load_events(path=path)
        assert len(events) == 2


class TestTransformLimits:
    def test_no_limit_until_enough_bad_votes(self):
        events = [
            {"category": "vocals", "verdict": "good", "semitones": 2, "stretch_ratio": 1.0},
            {"category": "vocals", "verdict": "bad", "semitones": 5, "stretch_ratio": 1.0},
        ]
        limits = aggregate_limits(events)
        # Only 1 bad vote — below MIN_VOTES_FOR_LIMIT
        assert "max_semitones" not in limits.get("vocals", {})

    def test_learns_semitone_ceiling(self):
        events = [
            {"category": "vocals", "verdict": "good", "semitones": 2, "stretch_ratio": 1.0},
            {"category": "vocals", "verdict": "good", "semitones": 3, "stretch_ratio": 1.0},
            {"category": "vocals", "verdict": "bad", "semitones": 5, "stretch_ratio": 1.0},
            {"category": "vocals", "verdict": "bad", "semitones": 6, "stretch_ratio": 1.0},
        ]
        limits = aggregate_limits(events)
        # Largest good is 3; smallest bad is 5 → ceiling at 3
        assert limits["vocals"]["max_semitones"] == 3.0

    def test_learns_stretch_ceiling(self):
        events = [
            {"category": "drums", "verdict": "good", "semitones": 0, "stretch_ratio": 1.05},
            {"category": "drums", "verdict": "bad", "semitones": 0, "stretch_ratio": 1.20},
            {"category": "drums", "verdict": "bad", "semitones": 0, "stretch_ratio": 0.80},
        ]
        limits = aggregate_limits(events)
        # 5% good, 20% bad → max_stretch_pct = 5
        assert limits["drums"]["max_stretch_pct"] == 5.0

    def test_check_transform_warns_past_limit(self, tmp_path):
        limits_path = tmp_path / "limits.json"
        save_limits({"vocals": {"max_semitones": 3, "votes_good": 2, "votes_bad": 2}},
                    path=limits_path)

        # Monkeypatch via explicit limits arg
        warn = check_transform("vocals", semitones=5, stretch_ratio=1.0,
                               limits={"vocals": {"max_semitones": 3}})
        assert warn is not None
        assert "5" in warn and "vocals" in warn

        ok = check_transform("vocals", semitones=2, stretch_ratio=1.0,
                             limits={"vocals": {"max_semitones": 3}})
        assert ok is None

    def test_check_transform_silent_without_data(self):
        assert check_transform("piano", 7, 1.1, limits={}) is None
