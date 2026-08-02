"""Synthetic tests for stem trimming and mixing."""

import numpy as np
import pytest
import soundfile as sf

from musicbot.processing.audio_analysis import trim_stems_to_onset
from processing.stack_engine import HEADROOM_DB, mix_stems

from .conftest import SR, sine

pytestmark = pytest.mark.synthetic


class TestTrimStems:
    def test_trims_exactly_by_offset(self, tmp_path):
        sr = SR
        y = sine(220.0, 3.0, sr)
        stem = tmp_path / "drums.wav"
        sf.write(str(stem), y, sr, subtype="PCM_24")

        trim_stems_to_onset(str(tmp_path), 0.5)

        info = sf.info(str(stem))
        expected_frames = len(y) - int(0.5 * sr)
        assert info.frames == expected_frames
        assert info.samplerate == sr
        assert info.subtype == "PCM_24"

    def test_trims_all_wavs_in_dir_identically(self, tmp_path):
        sr = SR
        for name in ("vocals", "drums", "bass", "other"):
            sf.write(str(tmp_path / f"{name}.wav"), sine(300.0, 2.0, sr), sr, subtype="PCM_24")

        trim_stems_to_onset(str(tmp_path), 0.25)

        lengths = {sf.info(str(p)).frames for p in tmp_path.glob("*.wav")}
        assert len(lengths) == 1  # all identical after trim

    def test_stereo_preserved(self, tmp_path):
        sr = SR
        left = sine(220.0, 2.0, sr)
        right = sine(330.0, 2.0, sr)
        stereo = np.stack([left, right], axis=1)
        stem = tmp_path / "stereo.wav"
        sf.write(str(stem), stereo, sr, subtype="PCM_24")

        trim_stems_to_onset(str(tmp_path), 0.5)

        info = sf.info(str(stem))
        assert info.channels == 2
        assert info.frames == len(left) - int(0.5 * sr)

    def test_zero_trim_is_noop(self, tmp_path):
        sr = SR
        stem = tmp_path / "drums.wav"
        sf.write(str(stem), sine(220.0, 1.0, sr), sr, subtype="PCM_24")
        before = sf.info(str(stem)).frames

        trim_stems_to_onset(str(tmp_path), 0.0)

        assert sf.info(str(stem)).frames == before

    def test_alignment_survives_trim(self, tmp_path):
        """Two stems with markers at the same timestamp must still align
        after trimming — this is the property the whole DAW workflow rests on."""
        sr = SR
        marker_sec = 1.0
        for name in ("a", "b"):
            y = np.zeros(int(2.0 * sr))
            y[int(marker_sec * sr)] = 1.0  # single-sample impulse marker
            sf.write(str(tmp_path / f"{name}.wav"), y, sr, subtype="PCM_24")

        trim_stems_to_onset(str(tmp_path), 0.4)

        positions = []
        for name in ("a", "b"):
            y, _ = sf.read(str(tmp_path / f"{name}.wav"))
            positions.append(int(np.argmax(np.abs(y))))
        assert positions[0] == positions[1]


class TestMixStems:
    def test_mix_length_is_shortest_input(self, tmp_path):
        sr = SR
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        sf.write(str(a), sine(220.0, 1.0, sr), sr)
        sf.write(str(b), sine(330.0, 2.0, sr), sr)

        out = tmp_path / "mix.wav"
        mix_stems([str(a), str(b)], str(out))

        assert sf.info(str(out)).frames == int(1.0 * sr)

    def test_peak_normalized_to_headroom(self, tmp_path):
        sr = SR
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        sf.write(str(a), sine(220.0, 1.0, sr, amp=0.9), sr)
        sf.write(str(b), sine(220.0, 1.0, sr, amp=0.9), sr)  # in-phase → would clip

        out = tmp_path / "mix.wav"
        mix_stems([str(a), str(b)], str(out))

        y, _ = sf.read(str(out))
        target_peak = 10.0 ** (HEADROOM_DB / 20.0)
        assert abs(np.abs(y).max() - target_peak) < 0.01

    def test_mono_upmixed_to_match_stereo(self, tmp_path):
        sr = SR
        mono = tmp_path / "mono.wav"
        stereo = tmp_path / "stereo.wav"
        sf.write(str(mono), sine(220.0, 1.0, sr), sr)
        sf.write(str(stereo), np.stack([sine(330.0, 1.0, sr)] * 2, axis=1), sr)

        out = tmp_path / "mix.wav"
        mix_stems([str(mono), str(stereo)], str(out))

        assert sf.info(str(out)).channels == 2

    def test_empty_input_writes_nothing(self, tmp_path):
        out = tmp_path / "mix.wav"
        mix_stems([], str(out))
        assert not out.exists()
