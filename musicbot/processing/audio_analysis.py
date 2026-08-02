"""BPM/key detection and stem preprocessing utilities.

Key detection uses the Krumhansl-Schmuckler algorithm:
  - Compute a chromagram from the audio.
  - Average chroma energy across time to get a 12-element pitch-class profile.
  - Correlate that profile against major and minor key templates (rotated for
    every root pitch), pick the key with the highest Pearson correlation.

Stem trimming (beat-grid approach):
  - Run librosa beat tracking on the drums stem to get a musically-aware
    beat grid.
  - The first beat in the grid is the trim anchor (bar 1 beat 1).
  - Optionally refine by looking for a kick-band onset in a narrow window
    around that beat.
  - Trim all stems to that point minus a small pre-roll.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

# Chromagram pitch-class ordering used by librosa (C=0 … B=11)
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler perceptual key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot Wheel – maps "Root mode" → Camelot code
CAMELOT_MAP: dict[str, str] = {
    # Major keys (B column)
    "C major":  "8B",  "G major":  "9B",  "D major": "10B",  "A major": "11B",
    "E major": "12B",  "B major":  "1B",  "F# major": "2B", "C# major":  "3B",
    "G# major": "4B", "D# major":  "5B", "A# major":  "6B",  "F major":  "7B",
    # Minor keys (A column)
    "A minor":  "8A",  "E minor":  "9A",  "B minor": "10A", "F# minor": "11A",
    "C# minor": "12A", "G# minor": "1A", "D# minor":  "2A", "A# minor":  "3A",
    "F minor":  "4A",  "C minor":  "5A",  "G minor":  "6A",  "D minor":  "7A",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def detect_bpm(y: np.ndarray, sr: int) -> float:
    """Return the estimated BPM from a pre-loaded audio signal."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(np.atleast_1d(tempo)[0]), 2)


def detect_key(y: np.ndarray, sr: int) -> tuple[str, str]:
    """Estimate the musical key from a pre-loaded audio signal.

    Returns (musical_key, camelot_code).
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)  # shape: (12,)

    best_score = -np.inf
    best_key = "C major"

    for i, pitch in enumerate(PITCH_CLASSES):
        rotated = np.roll(chroma_mean, -i)
        maj_score = float(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
        min_score = float(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])

        if maj_score > best_score:
            best_score = maj_score
            best_key = f"{pitch} major"
        if min_score > best_score:
            best_score = min_score
            best_key = f"{pitch} minor"

    camelot = CAMELOT_MAP.get(best_key, "unknown")
    return best_key, camelot


def find_beat_grid(audio_path: str) -> tuple[list[float], float]:
    """Return (beat_times_seconds, estimated_bpm) via librosa beat tracking."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 2)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    return beat_times, bpm


def analyze(file_path: str) -> dict:
    """Load an audio file once and return BPM, key, Camelot code, and beat grid.

    Keys returned: ``bpm``, ``key``, ``camelot``, ``beat_times``.
    """
    print(f"  Loading audio for analysis: {file_path}")
    y, sr = librosa.load(file_path, sr=None, mono=True)

    print("  Detecting BPM…")
    bpm = detect_bpm(y, sr)

    print("  Detecting musical key…")
    key, camelot = detect_key(y, sr)

    print("  Building beat grid…")
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    print(f"  BPM: {bpm}  |  Key: {key}  |  Camelot: {camelot}  |  Beats: {len(beat_times)}")
    return {"bpm": bpm, "key": key, "camelot": camelot, "beat_times": beat_times}


# ---------------------------------------------------------------------------
# Stem trimming (beat-grid approach)
# ---------------------------------------------------------------------------

PRE_ROLL_MS     = 2    # ms preserved before the detected transient start
KICK_CUTOFF     = 150  # Hz — lowpass to isolate kick from hats/snare
BACKTRACK_MS    = 50   # ms to walk back from onset peak to find transient start
NOISE_FLOOR_PCT = 5    # % of local peak for backtrack stop

TRIM_METHOD = "lp150-d40pct-bt50"


def find_first_kick_time(
    drums_path: str,
    *,
    pre_roll_ms: float = PRE_ROLL_MS,
    kick_cutoff: float = KICK_CUTOFF,
    delta_pct: float = 40.0,
    backtrack_ms: float = BACKTRACK_MS,
    noise_floor_pct: float = NOISE_FLOOR_PCT,
    verbose: bool = True,
) -> float:
    """Return the trim point (seconds) for bar 1 beat 1.

    All tunable parameters can be overridden via keyword arguments so the
    grid-search evaluator can sweep them without touching module constants.

    Returns 0.0 if no kick onsets are detected.
    """
    y, sr = librosa.load(drums_path, sr=None, mono=True)

    nyq = sr / 2.0
    b, a = butter(6, kick_cutoff / nyq, btype="low")
    y_kick = filtfilt(b, a, y)

    onset_env = librosa.onset.onset_strength(y=y_kick, sr=sr)
    if onset_env.max() == 0:
        if verbose:
            print("  No kick energy detected — skipping trim.")
        return 0.0

    delta = float((delta_pct / 100.0) * onset_env.max())
    # normalize=False keeps peak-picking in the same units as `delta`.
    # (With the default normalize=True the envelope is scaled to [0,1] but
    # delta stays in raw units — any envelope with max > 1/delta_pct% made
    # the threshold unreachable and silently returned 0.0.)
    onset_frames = librosa.onset.onset_detect(
        y=y_kick, sr=sr, onset_envelope=onset_env,
        backtrack=False, units="frames", delta=delta,
        normalize=False,
    )

    if len(onset_frames) == 0:
        if verbose:
            print("  No kick onsets above threshold — skipping trim.")
        return 0.0

    peak_sample = int(librosa.frames_to_samples(onset_frames[0]))
    bt_samples = int(backtrack_ms / 1000.0 * sr)
    win_start = max(0, peak_sample - bt_samples)

    y_abs = np.abs(y_kick)
    local_peak = y_abs[win_start : peak_sample + 1].max()
    threshold = (noise_floor_pct / 100.0) * local_peak

    attack_sample = peak_sample
    for s in range(peak_sample, win_start, -1):
        if y_abs[s] <= threshold:
            attack_sample = s
            break

    attack_sec = float(attack_sample) / sr
    peak_sec = float(peak_sample) / sr

    trim_sec = max(0.0, attack_sec - pre_roll_ms / 1000.0)
    if verbose:
        print(
            f"  Kick onset peak at {peak_sec * 1000:.1f} ms, "
            f"transient start at {attack_sec * 1000:.1f} ms "
            f"→ trimming to {trim_sec * 1000:.1f} ms ({pre_roll_ms} ms pre-roll)"
        )
    return trim_sec


def trim_stems_to_onset(stems_dir: str, trim_seconds: float) -> None:
    """Trim the start of every WAV in *stems_dir* by exactly *trim_seconds*.

    Files are overwritten in place as 24-bit PCM WAV at their original
    sample rate — the same format written by the download step.
    """
    if trim_seconds <= 0:
        return

    for wav_path in sorted(Path(stems_dir).glob("*.wav")):
        # Load stereo; librosa returns (channels, samples)
        y, sr = librosa.load(str(wav_path), sr=None, mono=False)

        trim_samples = int(trim_seconds * sr)

        if y.ndim == 1:
            y_trimmed = y[trim_samples:]
        else:
            y_trimmed = y[:, trim_samples:]   # (channels, samples)
            y_trimmed = y_trimmed.T            # soundfile wants (samples, channels)

        sf.write(str(wav_path), y_trimmed, sr, subtype="PCM_24")
        print(f"  {wav_path.name}  trimmed {trim_seconds * 1000:.0f} ms from start")
