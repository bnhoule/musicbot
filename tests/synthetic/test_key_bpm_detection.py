"""Synthetic tests for key and BPM detection (musicbot.processing.audio_analysis)."""

import pytest

from musicbot.processing.audio_analysis import CAMELOT_MAP, detect_bpm, detect_key

from .conftest import (
    MAJOR_SCALE,
    MINOR_SCALE,
    NOTE_HZ,
    SR,
    click_track,
    scale_note_mix,
)

pytestmark = pytest.mark.synthetic


class TestKeyDetection:
    @pytest.mark.parametrize(
        "root_note, scale, expected_key",
        [
            ("A3", MINOR_SCALE, "A minor"),
            ("C3", MAJOR_SCALE, "C major"),
            ("F3", MINOR_SCALE, "F minor"),
            ("G3", MAJOR_SCALE, "G major"),
        ],
    )
    def test_detects_scale_key(self, root_note, scale, expected_key):
        y = scale_note_mix(scale, NOTE_HZ[root_note])
        key, camelot = detect_key(y, SR)
        assert key == expected_key
        assert camelot == CAMELOT_MAP[expected_key]

    def test_returns_valid_camelot_for_any_input(self):
        # Even noise-ish input must map to a real Camelot code, never "unknown"
        y = scale_note_mix(MAJOR_SCALE, 200.0)  # non-tempered root
        key, camelot = detect_key(y, SR)
        assert camelot in CAMELOT_MAP.values()


class TestBpmDetection:
    @pytest.mark.parametrize("bpm", [90.0, 120.0, 128.0])
    def test_click_track_bpm(self, bpm):
        y = click_track(bpm)
        detected = detect_bpm(y, SR)
        # Accept octave errors (half/double) but demand the right tempo
        # region — librosa's tempo estimate quantizes to ~2-3% granularity.
        candidates = (detected, detected * 2, detected / 2)
        assert any(abs(c - bpm) < bpm * 0.03 for c in candidates), (
            f"expected ~{bpm}, got {detected}"
        )

    def test_bpm_is_rounded_to_two_decimals(self):
        y = click_track(120.0)
        detected = detect_bpm(y, SR)
        assert detected == round(detected, 2)
