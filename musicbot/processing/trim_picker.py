"""Interactive and automatic trim-point selection.

Presents ranked downbeat candidates to the user, accepts a pick,
and persists choices so re-runs skip already-decided songs.
"""

import json
import subprocess
import tempfile
from datetime import datetime, UTC
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
        with open(path, encoding="utf-8") as f:
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
        "timestamp": datetime.now(UTC).isoformat(),
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

    clip = y[start_sample:end_sample] if y.ndim == 1 else y[:, start_sample:end_sample].T

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


# ---------------------------------------------------------------------------
# Agreement scoring — turns every human pick into an implicit vote on the ranker
# ---------------------------------------------------------------------------

AGREEMENT_TOLERANCE_MS = 50.0


def _as_pairs(candidates) -> list[tuple[float, float]]:
    """Normalise TrimCandidate objects or plain dicts to (time_sec, energy_pct)."""
    pairs = []
    for c in candidates:
        if isinstance(c, dict):
            pairs.append((float(c["time_sec"]), float(c.get("energy_pct", 0.0))))
        else:
            pairs.append((float(c.time_sec), float(c.energy_pct)))
    return pairs


def score_pick_agreement(
    candidates,
    chosen_sec: float,
    tolerance_ms: float = AGREEMENT_TOLERANCE_MS,
) -> dict:
    """Compare the auto-pick against what the human actually chose.

    Every trim the user commits is a free judgement on the ranking logic:
    accepting the top-ranked candidate is a thumbs-up, overriding it is a
    thumbs-down plus the correct answer.

    Returns a dict with the auto-pick, the signed error in milliseconds,
    whether they agree within *tolerance_ms*, and the energy rank of the
    candidate the human chose (1 = highest energy, None = off-list).
    """
    pairs = _as_pairs(candidates)
    if not pairs:
        return {
            "auto_pick_sec": None,
            "auto_pick_delta_ms": None,
            "agreed": False,
            "chosen_rank": None,
            "n_candidates": 0,
        }

    by_energy = sorted(pairs, key=lambda p: p[1], reverse=True)
    auto_sec = by_energy[0][0]
    delta_ms = (auto_sec - chosen_sec) * 1000.0

    chosen_rank = None
    for rank, (time_sec, _energy) in enumerate(by_energy, start=1):
        if abs(time_sec - chosen_sec) * 1000.0 <= tolerance_ms:
            chosen_rank = rank
            break

    return {
        "auto_pick_sec": round(auto_sec, 4),
        "auto_pick_delta_ms": round(delta_ms, 1),
        "agreed": abs(delta_ms) <= tolerance_ms,
        "chosen_rank": chosen_rank,
        "n_candidates": len(pairs),
    }


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
