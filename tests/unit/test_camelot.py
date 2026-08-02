"""Consistency tests for the Camelot wheel and pitch-class tables.

These guard the musical invariants that the rekey and analysis modules
both depend on — a silent mismatch between the two tables would produce
wrong-key mixes without any crash.
"""

import re

import pytest

from musicbot.processing.audio_analysis import CAMELOT_MAP
from musicbot.processing.audio_analysis import PITCH_CLASSES as ANALYSIS_PITCHES
from musicbot.processing.rekey import PITCH_CLASSES as REKEY_PITCHES
from musicbot.processing.rekey import semitone_distance

pytestmark = pytest.mark.unit


class TestPitchClassTables:
    def test_tables_are_identical_across_modules(self):
        assert ANALYSIS_PITCHES == REKEY_PITCHES

    def test_twelve_unique_pitches(self):
        assert len(ANALYSIS_PITCHES) == 12
        assert len(set(ANALYSIS_PITCHES)) == 12


class TestCamelotMap:
    def test_covers_all_24_keys(self):
        assert len(CAMELOT_MAP) == 24
        for pitch in ANALYSIS_PITCHES:
            assert f"{pitch} major" in CAMELOT_MAP
            assert f"{pitch} minor" in CAMELOT_MAP

    def test_codes_are_valid_and_unique(self):
        codes = list(CAMELOT_MAP.values())
        assert len(set(codes)) == 24
        for code in codes:
            m = re.fullmatch(r"(\d{1,2})([AB])", code)
            assert m, f"Malformed Camelot code: {code}"
            assert 1 <= int(m.group(1)) <= 12

    def test_minor_keys_are_column_a_major_keys_column_b(self):
        for key, code in CAMELOT_MAP.items():
            expected_col = "A" if key.endswith("minor") else "B"
            assert code.endswith(expected_col), f"{key} → {code}"

    def test_relative_major_minor_share_camelot_number(self):
        """A minor / C major are relatives → both 8; this must hold for all 12 pairs.

        The relative major of a minor key is +3 semitones from its root.
        """
        for i, pitch in enumerate(ANALYSIS_PITCHES):
            minor_key = f"{pitch} minor"
            relative_major_root = ANALYSIS_PITCHES[(i + 3) % 12]
            major_key = f"{relative_major_root} major"

            minor_num = CAMELOT_MAP[minor_key][:-1]
            major_num = CAMELOT_MAP[major_key][:-1]
            assert minor_num == major_num, (
                f"{minor_key} ({CAMELOT_MAP[minor_key]}) and its relative "
                f"{major_key} ({CAMELOT_MAP[major_key]}) must share a number"
            )

    def test_adjacent_camelot_numbers_are_perfect_fifths(self):
        """Moving +1 on the wheel = +7 semitones (a perfect fifth) — the whole
        point of harmonic mixing. Verify for the B (major) column."""
        by_code = {v: k for k, v in CAMELOT_MAP.items()}
        for num in range(1, 13):
            this_key = by_code[f"{num}B"]
            next_key = by_code[f"{(num % 12) + 1}B"]
            dist = semitone_distance(this_key, next_key) % 12
            assert dist == 7, f"{this_key} → {next_key} should be a fifth, got {dist}"
