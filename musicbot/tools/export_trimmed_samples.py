"""Export trimmed drums clips for each detection method + ground truth.

For each labeled song, outputs a short WAV clip starting at each method's
detected trim point plus the ground truth label. Drop the clips into Ableton
to compare visually and audibly.

Usage
-----
    python export_trimmed_samples.py --songs-dir /path/to/Annotated
"""

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends.demucs_separator import separate
from processing.audio_analysis import find_first_kick_time

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = Path(__file__).resolve().parents[2] / "eval_cache"
LABELS_FILE = DATA_DIR / "kick_labels.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "eval_samples"

CLIP_DURATION_SEC = 8.0


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


def get_drums_stem(mp3_path: Path) -> Path:
    cache_key = mp3_path.stem
    song_cache = CACHE_DIR / cache_key / "stems"
    drums_path = song_cache / "drums.wav"
    if drums_path.is_file():
        return drums_path
    song_cache.mkdir(parents=True, exist_ok=True)
    separate(str(mp3_path), str(song_cache))
    return drums_path


def trim_and_save(drums_path: Path, trim_sec: float, output_path: Path) -> None:
    """Load drums stem, trim to trim_sec, save first CLIP_DURATION_SEC."""
    y, sr = librosa.load(str(drums_path), sr=None, mono=False)

    trim_samples = max(0, int(trim_sec * sr))

    if y.ndim == 1:
        y_trimmed = y[trim_samples:]
    else:
        y_trimmed = y[:, trim_samples:]

    clip_samples = int(CLIP_DURATION_SEC * sr)
    if y.ndim == 1:
        y_clip = y_trimmed[:clip_samples]
    else:
        y_clip = y_trimmed[:, :clip_samples]
        y_clip = y_clip.T  # soundfile wants (samples, channels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y_clip, sr, subtype="PCM_24")


# ---------------------------------------------------------------------------
# Detection methods (same as evaluate_kicks.py)
# ---------------------------------------------------------------------------

def detect_baseline(drums_path: Path, raw_path: Path) -> float:
    return find_first_kick_time(str(drums_path), verbose=False)


def detect_madmom_groove(drums_path: Path, raw_path: Path) -> float:
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
    window_sec = bar_dur * 4

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

    for i, (db, e) in enumerate(zip(downbeats, energies)):
        if e >= threshold:
            following = energies[i:i+3]
            if len(following) >= 2 and all(f >= threshold for f in following):
                return float(db)

    return float(downbeats[np.argmax(energies)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = {
    "baseline": detect_baseline,
    "madmom": detect_madmom_groove,
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export trimmed drums clips per method for A/B comparison",
    )
    p.add_argument("--songs-dir", required=True, type=Path)
    args = p.parse_args()

    labels = load_labels(LABELS_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for entry in labels:
        mp3 = match_file(entry["song"], args.songs_dir)
        if mp3 is None:
            print(f"  [skip] {entry['song']}")
            continue

        drums = get_drums_stem(mp3)
        song_slug = re.sub(r"[^a-zA-Z0-9]", "_", entry["song"]).strip("_")
        song_dir = OUTPUT_DIR / song_slug
        label_sec = entry["kick_sec"]

        print(f"\n  {entry['song']}  (label: {label_sec:.3f}s)")

        # Ground truth clip
        out = song_dir / f"0_GROUND_TRUTH_{int(label_sec * 1000)}ms.wav"
        trim_and_save(drums, label_sec, out)
        print(f"    ground_truth  → {label_sec:.3f}s")

        # Each method
        for method_name, detect_fn in METHODS.items():
            try:
                detected = detect_fn(drums, mp3)
            except Exception as e:
                print(f"    {method_name:<12}  → ERROR: {e}")
                continue

            error_ms = (detected - label_sec) * 1000
            out = song_dir / f"{method_name}_{int(detected * 1000)}ms_err{int(error_ms):+d}ms.wav"
            trim_and_save(drums, detected, out)
            print(f"    {method_name:<12}  → {detected:.3f}s  (err: {error_ms:+.0f}ms)")

    print(f"\n  Samples written to: {OUTPUT_DIR}")
    print("  Each folder has ground truth + method clips.")
    print("  Drag into Ableton to compare alignment.\n")


if __name__ == "__main__":
    main()
