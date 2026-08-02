"""Synthetic audio generators shared by the Layer 2 DSP tests.

Everything here is generated from numpy with fixed parameters — fully
deterministic, no audio files needed in the repo.
"""

import numpy as np
import pytest
import soundfile as sf

SR = 22050

# Note frequencies (Hz) for octave 3-4, equal temperament A440
NOTE_HZ = {
    "C3": 130.81, "D3": 146.83, "E3": 164.81, "F3": 174.61,
    "G3": 196.00, "A3": 220.00, "B3": 246.94,
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
    "G4": 392.00, "A4": 440.00, "B4": 493.88,
    "G#3": 207.65, "F#3": 185.00,
}


def sine(freq: float, duration: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def scale_note_mix(root_offsets: list[int], root_hz: float, duration: float = 8.0,
                   sr: int = SR) -> np.ndarray:
    """Sum sines for a scale, weighting tonic and fifth more heavily.

    root_offsets are semitone offsets from the root defining the scale.
    """
    y = np.zeros(int(duration * sr))
    for idx, offset in enumerate(root_offsets):
        freq = root_hz * (2 ** (offset / 12.0))
        # Tonic loudest, fifth next (scale degree 5 is index 4), rest quieter
        if idx == 0:
            amp = 1.0
        elif offset == 7:
            amp = 0.6
        else:
            amp = 0.3
        y += sine(freq, duration, sr, amp=amp)
    peak = np.abs(y).max()
    return (y / peak * 0.7).astype(np.float64)


MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # natural minor


def kick(sr: int = SR, duration: float = 0.4) -> np.ndarray:
    """Synthesized kick drum: 2 ms attack, brief 90→55 Hz pitch drop, then a
    steady decaying 55 Hz tone — mimics a real electronic kick's spectral
    shape (flux concentrated at the attack, not smeared across the decay).
    """
    n = int(duration * sr)
    t = np.arange(n) / sr
    freq = np.where(t < 0.03, 90.0 - (35.0 / 0.03) * t, 55.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 9)
    attack = np.minimum(t / 0.002, 1.0)
    return (0.9 * env * attack * np.sin(phase)).astype(np.float64)


def kick_pattern(bpm: float, silence_sec: float, n_kicks: int = 16,
                 sr: int = SR) -> np.ndarray:
    """Silence followed by four-on-the-floor kicks at the given BPM."""
    beat_period = 60.0 / bpm
    total = silence_sec + n_kicks * beat_period + 1.0
    y = np.zeros(int(total * sr))
    k = kick(sr)
    for i in range(n_kicks):
        start = int((silence_sec + i * beat_period) * sr)
        y[start:start + len(k)] += k
    return y.astype(np.float64)


def click_track(bpm: float, duration: float = 20.0, sr: int = SR) -> np.ndarray:
    """Short broadband clicks at the beat period — ideal beat-tracker input."""
    beat_period = 60.0 / bpm
    y = np.zeros(int(duration * sr))
    click_len = int(0.01 * sr)
    rng = np.random.default_rng(1234)          # fixed seed → deterministic
    click_burst = rng.uniform(-1, 1, click_len) * np.exp(-np.arange(click_len) / (0.002 * sr))
    t = 0.0
    while t < duration - 0.1:
        s = int(t * sr)
        y[s:s + click_len] += click_burst
        t += beat_period
    return (y * 0.8).astype(np.float64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def write_wav(tmp_path):
    """Factory fixture: write an array to a temp WAV and return its path."""
    counter = {"n": 0}

    def _write(y: np.ndarray, sr: int = SR, subtype: str = "PCM_24") -> str:
        counter["n"] += 1
        path = tmp_path / f"synth_{counter['n']}.wav"
        sf.write(str(path), y, sr, subtype=subtype)
        return str(path)

    return _write
