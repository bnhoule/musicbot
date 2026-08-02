"""Interactive and automatic trim-point selection.

Presents ranked downbeat candidates to the user, accepts a pick,
and persists choices so re-runs skip already-decided songs.
"""

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import librosa
import soundfile as sf

from .madmom_beats import TrimCandidate

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CHOICES_FILE = DATA_DIR / "trim_choices.json"
PREVIEW_DURATION = 4.0  # seconds


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_trim_choices(path: Path = CHOICES_FILE) -> dict:
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_trim_choice(
    song_key: str,
    trim_sec: float,
    method: str = "interactive",
    path: Path = CHOICES_FILE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    choices = load_trim_choices(path)
    choices[song_key] = {
        "trim_sec": round(trim_sec, 4),
        "method": method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(choices, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Audio preview
# ---------------------------------------------------------------------------

def _load_drums(drums_path: str) -> tuple:
    """Load drums audio once and cache it for repeated previews."""
    y, sr = librosa.load(drums_path, sr=None, mono=False)
    return y, sr


def _play_preview(y, sr: int, start_sec: float) -> None:
    """Play a short clip from pre-loaded drums audio."""
    start_sample = int(start_sec * sr)
    end_sample = int(min((start_sec + PREVIEW_DURATION) * sr,
                         y.shape[-1]))

    if y.ndim == 1:
        clip = y[start_sample:end_sample]
    else:
        clip = y[:, start_sample:end_sample].T

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, clip, sr, subtype="PCM_24")
        tmp_path = tmp.name

    for cmd in (
        ["ffplay", "-nodisp", "-autoexit", tmp_path],
        ["afplay", tmp_path],
    ):
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    print("    [no audio player found — install ffmpeg for previews]")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _energy_bar(pct: float, width: int = 10) -> str:
    filled = round(pct / 100 * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _format_time(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Picker modes
# ---------------------------------------------------------------------------

def pick_trim_auto(candidates: list[TrimCandidate]) -> float:
    """Return the trim time of the highest-energy candidate."""
    if not candidates:
        return 0.0
    best = max(candidates, key=lambda c: c.energy_pct)
    return best.time_sec


def pick_trim_interactive(
    candidates: list[TrimCandidate],
    song_name: str,
    bpm: float,
    drums_path: str,
) -> float:
    """Display candidates, auto-play the top pick, and let the user choose.

    Returns the chosen trim time in seconds.
    """
    if not candidates:
        print("    No downbeat candidates found — defaulting to 0.0s")
        return 0.0

    auto_idx = max(range(len(candidates)),
                   key=lambda i: candidates[i].energy_pct)

    print()
    print(f"  Trim candidates for: {song_name}  ({bpm} BPM)")
    print()
    for i, c in enumerate(candidates):
        marker = " *" if i == auto_idx else "  "
        print(f"    [{i + 1}]  {_format_time(c.time_sec):>7s}   "
              f"{_energy_bar(c.energy_pct)}  {c.energy_pct:5.1f}%{marker}")

    y, sr = _load_drums(drums_path)

    print()
    print(f"  Playing auto-pick [{auto_idx + 1}] "
          f"({_format_time(candidates[auto_idx].time_sec)})…")
    _play_preview(y, sr, candidates[auto_idx].time_sec)

    print()
    print(f"  Enter = accept [{auto_idx + 1}],  "
          f"p<N> = preview,  number = pick")

    while True:
        try:
            raw = input("  pick> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return candidates[auto_idx].time_sec

        if raw == "":
            chosen = candidates[auto_idx]
            print(f"    Accepted [{auto_idx + 1}] → {_format_time(chosen.time_sec)}")
            return chosen.time_sec

        if raw.lower().startswith("p"):
            num_str = raw[1:].strip()
            try:
                idx = int(num_str) - 1
                if 0 <= idx < len(candidates):
                    print(f"    Playing [{idx + 1}] "
                          f"({_format_time(candidates[idx].time_sec)})…")
                    _play_preview(y, sr, candidates[idx].time_sec)
                else:
                    print(f"    Invalid — enter 1-{len(candidates)}")
            except ValueError:
                print(f"    Usage: p1, p2, … p{len(candidates)}")
            continue

        try:
            idx = int(raw) - 1
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]
                print(f"    Playing [{idx + 1}] "
                      f"({_format_time(chosen.time_sec)})…")
                _play_preview(y, sr, chosen.time_sec)
                print(f"    Picked [{idx + 1}] → {_format_time(chosen.time_sec)}")
                return chosen.time_sec
            else:
                print(f"    Invalid — enter 1-{len(candidates)}")
        except ValueError:
            print(f"    Enter a number 1-{len(candidates)}, p<N>, or Enter")
