"""Madmom-based downbeat detection and candidate ranking.

Uses madmom's RNN + Dynamic Bayesian Network pipeline for precise
downbeat timing, then ranks candidates by kick energy in the Demucs
drums stem.
"""

import warnings
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class TrimCandidate:
    """One potential trim point with its energy context."""
    time_sec: float
    energy: float
    energy_pct: float  # 0-100 relative to max candidate


def get_downbeats(raw_path: str) -> tuple[list[float], float]:
    """Run madmom RNN downbeat detection on the raw audio file.

    Returns (downbeat_times_sec, estimated_bpm).
    """
    import madmom
    warnings.filterwarnings("ignore")

    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[4], fps=100
    )
    act = madmom.features.downbeats.RNNDownBeatProcessor()(raw_path)
    beats = proc(act)

    downbeats = beats[beats[:, 1] == 1][:, 0].tolist()
    all_beat_times = beats[:, 0]
    bpm = 60.0 / float(np.median(np.diff(all_beat_times))) if len(all_beat_times) > 1 else 120.0

    return downbeats, round(bpm, 3)


def rank_candidates(
    downbeats: list[float],
    drums_path: str,
    bpm: float = 125.0,
    n: int = 8,
) -> list[TrimCandidate]:
    """Score each downbeat by kick energy in the following 4 bars,
    return the top *n* sorted by descending energy."""
    if not downbeats:
        return []

    y, sr = librosa.load(drums_path, sr=22050, mono=True)

    nyq = sr / 2.0
    b, a = butter(4, 150 / nyq, btype="low")
    y_kick = filtfilt(b, a, y)
    kick_env = np.abs(y_kick)

    bar_dur = 4 * 60.0 / max(bpm, 60)
    window_sec = bar_dur * 4

    scored: list[tuple[float, float]] = []
    for db in downbeats:
        start = int(db * sr)
        end = int(min((db + window_sec) * sr, len(kick_env)))
        if start >= len(kick_env):
            continue
        energy = float(np.mean(kick_env[start:end]))
        scored.append((db, energy))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:n]
    max_energy = top[0][1] if top[0][1] > 0 else 1.0

    candidates = []
    for time_sec, energy in top:
        pct = (energy / max_energy) * 100.0
        candidates.append(TrimCandidate(
            time_sec=time_sec,
            energy=energy,
            energy_pct=pct,
        ))

    candidates.sort(key=lambda c: c.time_sec)
    return candidates


def score_all_downbeats(
    downbeats: list[float],
    drums_path: str,
    bpm: float = 125.0,
) -> list[dict]:
    """Score every downbeat by kick energy, return all sorted by time.

    Each entry: {"time_sec": float, "energy_pct": float}.
    energy_pct is 0-100 relative to the global max.
    """
    if not downbeats:
        return []

    y, sr = librosa.load(drums_path, sr=22050, mono=True)

    nyq = sr / 2.0
    b, a = butter(4, 150 / nyq, btype="low")
    y_kick = filtfilt(b, a, y)
    kick_env = np.abs(y_kick)

    bar_dur = 4 * 60.0 / max(bpm, 60)
    window_sec = bar_dur * 4

    scored: list[tuple[float, float]] = []
    for db in downbeats:
        start = int(db * sr)
        end = int(min((db + window_sec) * sr, len(kick_env)))
        if start >= len(kick_env):
            continue
        energy = float(np.mean(kick_env[start:end]))
        scored.append((db, energy))

    if not scored:
        return []

    max_energy = max(e for _, e in scored) or 1.0
    return [
        {"time_sec": t, "energy_pct": round((e / max_energy) * 100.0, 1)}
        for t, e in sorted(scored)
    ]
