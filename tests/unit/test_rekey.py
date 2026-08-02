"""Unit tests for key parsing and semitone distance (musicbot.processing.rekey)."""

import pytest

from musicbot.processing.rekey import parse_key, semitone_distance

pytestmark = pytest.mark.unit


class TestParseKey:
    @pytest.mark.parametrize(
        "key_str, expected_pc, expected_mode",
        [
            ("C major", 0, "major"),
            ("A minor", 9, "minor"),
            ("F# minor", 6, "minor"),
            ("B major", 11, "major"),
            ("G# minor", 8, "minor"),
        ],
    )
    def test_sharp_keys(self, key_str, expected_pc, expected_mode):
        assert parse_key(key_str) == (expected_pc, expected_mode)

    @pytest.mark.parametrize(
        "flat_key, sharp_equivalent",
        [
            ("Db major", "C# major"),
            ("Eb minor", "D# minor"),
            ("Gb major", "F# major"),
            ("Ab minor", "G# minor"),
            ("Bb major", "A# major"),
            ("Cb major", "B major"),
            ("Fb major", "E major"),
        ],
    )
    def test_enharmonic_flats_normalize_to_sharps(self, flat_key, sharp_equivalent):
        assert parse_key(flat_key) == parse_key(sharp_equivalent)

    def test_whitespace_tolerated(self):
        assert parse_key("  A minor  ") == (9, "minor")

    @pytest.mark.parametrize(
        "bad_input",
        ["", "A", "A min", "H major", "A major minor", "major A"],
    )
    def test_invalid_raises(self, bad_input):
        with pytest.raises(ValueError):
            parse_key(bad_input)


class TestSemitoneDistance:
    def test_identity_is_zero(self):
        assert semitone_distance("A minor", "A minor") == 0

    def test_mode_change_same_root_is_zero(self):
        # Distance is root-to-root; A minor -> A major is 0 semitones
        assert semitone_distance("A minor", "A major") == 0

    @pytest.mark.parametrize(
        "src, tgt, expected",
        [
            ("A minor", "C minor", 3),     # up a minor third
            ("C major", "A major", -3),    # down a minor third
            ("B major", "C major", 1),     # wraps around the top
            ("C major", "B major", -1),    # wraps around the bottom
            ("C major", "F# major", 6),    # tritone: +6 by convention
            ("F# major", "C major", 6),    # tritone the other way is also +6
            ("G minor", "D minor", -5),    # shortest path is down, not up 7
        ],
    )
    def test_shortest_path(self, src, tgt, expected):
        assert semitone_distance(src, tgt) == expected

    def test_result_always_within_six(self):
        roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for src in roots:
            for tgt in roots:
                d = semitone_distance(f"{src} major", f"{tgt} minor")
                assert -6 <= d <= 6

    def test_flats_and_sharps_are_equivalent(self):
        assert semitone_distance("Bb minor", "C minor") == semitone_distance("A# minor", "C minor")
