"""Synthetic tests for pitch shifting and time stretching.

Requires the ``rubberband`` CLI (brew install rubberband / apt install
rubberband-cli). These verify the audio actually changed the way the
function names promise — frequency ratio for rekey, duration ratio for
stretch.
"""

import numpy as np
import pytest
import soundfile as sf

from musicbot.processing.rekey import rekey_audio
from musicbot.processing.tempo_match import stretch_audio

from .conftest import click_track, sine

pytestmark = pytest.mark.synthetic


def dominant_freq(path: str) -> float:
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    return float(freqs[int(np.argmax(spectrum))])


class TestRekeyAudio:
    def test_shift_up_two_semitones(self, write_wav, tmp_path):
        src = write_wav(sine(440.0, 2.0))
        out = str(tmp_path / "shifted.wav")
        rekey_audio(src, +2, out)

        expected = 440.0 * (2 ** (2 / 12))  # ≈ 493.88 Hz
        assert abs(dominant_freq(out) - expected) < expected * 0.01

    def test_shift_down_three_semitones(self, write_wav, tmp_path):
        src = write_wav(sine(440.0, 2.0))
        out = str(tmp_path / "shifted.wav")
        rekey_audio(src, -3, out)

        expected = 440.0 * (2 ** (-3 / 12))  # ≈ 369.99 Hz
        assert abs(dominant_freq(out) - expected) < expected * 0.01

    def test_zero_shift_copies_unchanged(self, write_wav, tmp_path):
        src = write_wav(sine(440.0, 1.0))
        out = str(tmp_path / "copy.wav")
        rekey_audio(src, 0, out)

        assert abs(dominant_freq(out) - 440.0) < 4.4
        assert sf.info(out).subtype == "PCM_24"

    def test_duration_preserved(self, write_wav, tmp_path):
        src = write_wav(sine(440.0, 2.0))
        out = str(tmp_path / "shifted.wav")
        rekey_audio(src, +4, out)

        assert abs(sf.info(out).duration - 2.0) < 0.05  # pitch shift ≠ speed change


class TestStretchAudio:
    def test_speed_up_shortens(self, write_wav, tmp_path):
        src = write_wav(click_track(120.0, duration=10.0))
        out = str(tmp_path / "stretched.wav")
        stretch_audio(src, source_bpm=120.0, target_bpm=126.0, output_path=out)

        expected_duration = 10.0 / (126.0 / 120.0)
        assert abs(sf.info(out).duration - expected_duration) < expected_duration * 0.02

    def test_slow_down_lengthens(self, write_wav, tmp_path):
        src = write_wav(click_track(128.0, duration=10.0))
        out = str(tmp_path / "stretched.wav")
        stretch_audio(src, source_bpm=128.0, target_bpm=120.0, output_path=out)

        expected_duration = 10.0 / (120.0 / 128.0)
        assert abs(sf.info(out).duration - expected_duration) < expected_duration * 0.02

    def test_equal_bpm_skips_stretch(self, write_wav, tmp_path):
        src = write_wav(click_track(125.0, duration=5.0))
        out = str(tmp_path / "same.wav")
        stretch_audio(src, source_bpm=125.0, target_bpm=125.05, output_path=out)

        # within 0.1 BPM → copied unchanged
        assert abs(sf.info(out).duration - 5.0) < 0.01
        assert sf.info(out).subtype == "PCM_24"

    def test_pitch_unchanged_by_stretch(self, write_wav, tmp_path):
        src = write_wav(sine(440.0, 4.0))
        out = str(tmp_path / "stretched.wav")
        stretch_audio(src, source_bpm=120.0, target_bpm=132.0, output_path=out)

        freq = dominant_freq(out)
        assert abs(freq - 440.0) < 8.8  # time stretch must not shift pitch (±2%)
