"""Evaluate kick detection against hand-labeled ground truth.

Reads data/kick_labels.csv, runs Demucs (cached) on each song to get drums
stems, then compares multiple detection methods to the human labels.

Usage
-----
    python evaluate_kicks.py --songs-dir /path/to/Annotated
    python evaluate_kicks.py --songs-dir /path/to/Annotated --methods all
"""

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends.demucs_separator import separate
from processing.audio_analysis import find_first_kick_time

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = Path(__file__).resolve().parents[2] / "eval_cache"
LABELS_FILE = DATA_DIR / "kick_labels.csv"

METHODS = ["baseline", "madmom_groove", "energy_jump"]


# ---------------------------------------------------------------------------
# Label loading + name matching
# ---------------------------------------------------------------------------

def load_labels(labels_path: Path) -> list[dict]:
    rows = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "song": row["Song"].strip(),
                "kick_sec": float(row["Kick Start (seconds)"].strip()),
            })
    return rows


def match_file(song_name: str, songs_dir: Path) -> Path | None:
    name_lower = song_name.lower()
    for mp3 in songs_dir.glob("*.mp3"):
        stem = re.sub(r"^\d+\s*-\s*", "", mp3.stem)
        if name_lower in stem.lower():
            return mp3
    return None


# ---------------------------------------------------------------------------
# Demucs caching
# ---------------------------------------------------------------------------

def get_drums_stem(mp3_path: Path) -> Path:
    cache_key = mp3_path.stem
    song_cache = CACHE_DIR / cache_key / "stems"
    drums_path = song_cache / "drums.wav"
    if drums_path.is_file():
        return drums_path
    song_cache.mkdir(parents=True, exist_ok=True)
    separate(str(mp3_path), str(song_cache))
    return drums_path


# ---------------------------------------------------------------------------
# Method 1: Baseline (existing lowpass onset detection on drums stem)
# ---------------------------------------------------------------------------

def detect_baseline(drums_path: Path, raw_path: Path) -> float:
    return find_first_kick_time(str(drums_path), verbose=False)


# ---------------------------------------------------------------------------
# Method 2: madmom RNN downbeat + kick energy groove detection
# ---------------------------------------------------------------------------

def detect_madmom_groove(drums_path: Path, raw_path: Path) -> float:
    """Use madmom RNN downbeat tracking on the raw file, then find the first
    downbeat where low-frequency energy in the drums stem stays consistently
    high (groove has locked in)."""
    import madmom
    warnings.filterwarnings("ignore")

    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[4], fps=100
    )
    act = madmom.features.downbeats.RNNDownBeatProcessor()(str(raw_path))
    beats = proc(act)

    downbeats = beats[beats[:, 1] == 1][:, 0]
    if len(downbeats) == 0:
        return 0.0

    y, sr = librosa.load(str(drums_path), sr=22050, mono=True)

    from scipy.signal import butter, filtfilt
    nyq = sr / 2.0
    b, a = butter(4, 150 / nyq, btype="low")
    y_kick = filtfilt(b, a, y)
    kick_env = np.abs(y_kick)

    bpm_est = 60.0 / np.median(np.diff(downbeats)) if len(downbeats) > 1 else 125.0
    bar_dur = 4 * 60.0 / bpm_est
    window_bars = 4
    window_sec = bar_dur * window_bars

    energies = []
    for db in downbeats:
        start = int(db * sr)
        end = int(min((db + window_sec) * sr, len(kick_env)))
        if start >= len(kick_env):
            energies.append(0.0)
            continue
        energies.append(float(np.mean(kick_env[start:end])))

    energies = np.array(energies)
    if energies.max() == 0:
        return 0.0

    threshold = 0.55 * energies.max()

    for i, (db, e) in enumerate(zip(downbeats, energies, strict=True)):
        if e >= threshold:
            following = energies[i:i+3]
            if len(following) >= 2 and all(f >= threshold for f in following):
                return float(db)

    return float(downbeats[np.argmax(energies)])


# ---------------------------------------------------------------------------
# Method 3: Energy jump detection
# ---------------------------------------------------------------------------

def detect_energy_jump(drums_path: Path, raw_path: Path) -> float:
    """Find the point of maximum *increase* in kick energy — the "drop."

    Computes a rolling mean of low-frequency drum energy, then looks for the
    largest upward jump between consecutive windows. Snaps the result to the
    nearest madmom downbeat for precise timing."""
    import madmom
    warnings.filterwarnings("ignore")

    y, sr = librosa.load(str(drums_path), sr=22050, mono=True)

    from scipy.signal import butter, filtfilt
    nyq = sr / 2.0
    b, a = butter(4, 150 / nyq, btype="low")
    y_kick = filtfilt(b, a, y)
    kick_env = np.abs(y_kick)

    bpm_est = librosa.beat.beat_track(y=librosa.load(str(raw_path), sr=22050, mono=True)[0], sr=22050)[0]
    bpm_est = float(np.atleast_1d(bpm_est)[0])
    bar_dur = 4 * 60.0 / max(bpm_est, 80)
    window_sec = bar_dur * 4
    hop_sec = bar_dur

    n_windows = int((len(y) / sr - window_sec) / hop_sec)
    if n_windows < 2:
        return 0.0

    times = []
    energies = []
    for i in range(n_windows):
        t = i * hop_sec
        s = int(t * sr)
        e = int(min((t + window_sec) * sr, len(kick_env)))
        times.append(t)
        energies.append(float(np.mean(kick_env[s:e])))

    energies = np.array(energies)
    times = np.array(times)

    diffs = np.diff(energies)
    if diffs.max() <= 0:
        return 0.0

    jump_idx = int(np.argmax(diffs))
    jump_time = times[jump_idx + 1]

    # Snap to nearest madmom downbeat
    try:
        proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=[4], fps=100
        )
        act = madmom.features.downbeats.RNNDownBeatProcessor()(str(raw_path))
        beats = proc(act)
        downbeats = beats[beats[:, 1] == 1][:, 0]

        if len(downbeats) > 0:
            diffs_db = np.abs(downbeats - jump_time)
            nearest = downbeats[np.argmin(diffs_db)]
            if abs(nearest - jump_time) < bar_dur * 2:
                return float(nearest)
    except Exception:
        pass

    return float(jump_time)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

METHOD_FNS = {
    "baseline": detect_baseline,
    "madmom_groove": detect_madmom_groove,
    "energy_jump": detect_energy_jump,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(songs_dir: Path, method: str) -> list[dict]:
    labels = load_labels(LABELS_FILE)
    detect_fn = METHOD_FNS[method]
    results = []

    for entry in labels:
        mp3 = match_file(entry["song"], songs_dir)
        if mp3 is None:
            print(f"  [skip] No file found for '{entry['song']}'")
            continue

        drums = get_drums_stem(mp3)
        print(f"  {entry['song']:<35} ", end="", flush=True)

        try:
            detected_sec = detect_fn(drums, mp3)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        label_sec = entry["kick_sec"]
        error_ms = (detected_sec - label_sec) * 1000.0
        print(f"label={label_sec:.3f}  detected={detected_sec:.3f}  err={error_ms:+.0f}ms")

        results.append({
            "song": entry["song"],
            "label_sec": label_sec,
            "detected_sec": detected_sec,
            "error_ms": error_ms,
        })

    return results


def print_report(results: list[dict], method: str) -> float:
    if not results:
        print(f"  No results for {method}.")
        return float("inf")

    print(f"\n  {'Song':<35}  {'Label':>8}  {'Detected':>8}  {'Error ms':>9}")
    print("  " + "-" * 68)
    for r in sorted(results, key=lambda x: abs(x["error_ms"]), reverse=True):
        print(
            f"  {r['song']:<35}  "
            f"{r['label_sec']:>8.3f}  "
            f"{r['detected_sec']:>8.3f}  "
            f"{r['error_ms']:>+9.1f}"
        )

    errors = [r["error_ms"] for r in results]
    abs_errors = [abs(e) for e in errors]
    mae = sum(abs_errors) / len(abs_errors)
    mean = sum(errors) / len(errors)
    max_err = max(abs_errors)
    within_50 = sum(1 for e in abs_errors if e <= 50)
    within_500 = sum(1 for e in abs_errors if e <= 500)

    print("  " + "-" * 68)
    print(f"  n={len(results)}   MAE: {mae:.0f} ms   "
          f"mean: {mean:+.0f} ms   max: {max_err:.0f} ms")
    print(f"  within 50ms: {within_50}/{len(results)}   "
          f"within 500ms: {within_500}/{len(results)}\n")
    return mae


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_kicks.py",
        description="Compare kick detection methods against ground truth",
    )
    p.add_argument(
        "--songs-dir", required=True, type=Path,
        help="Directory of annotated MP3 files",
    )
    p.add_argument(
        "--methods", nargs="+", default=["all"],
        choices=METHODS + ["all"],
        help=f"Methods to evaluate (default: all). Choices: {METHODS}",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.songs_dir.is_dir():
        sys.exit(f"[error] Songs directory not found: {args.songs_dir}")
    if not LABELS_FILE.is_file():
        sys.exit(f"[error] Labels file not found: {LABELS_FILE}")

    methods = METHODS if "all" in args.methods else args.methods

    summary = []
    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")
        results = evaluate(args.songs_dir, method)
        mae = print_report(results, method)
        summary.append((method, mae, results))

    if len(summary) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*60}\n")
        print(f"  {'Method':<20}  {'MAE ms':>8}  {'<50ms':>6}  {'<500ms':>7}")
        print("  " + "-" * 48)
        for method, mae, results in sorted(summary, key=lambda x: x[1]):
            abs_errs = [abs(r["error_ms"]) for r in results]
            w50 = sum(1 for e in abs_errs if e <= 50)
            w500 = sum(1 for e in abs_errs if e <= 500)
            print(f"  {method:<20}  {mae:>8.0f}  {w50:>4}/{len(results)}  {w500:>5}/{len(results)}")
        print()


if __name__ == "__main__":
    main()
