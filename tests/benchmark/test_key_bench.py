"""Layer 3: key detection benchmark against ear-confirmed labels.

Labels accumulate in data/key_labels.csv every time you thumbs-up/down a
detected key in the web UI. Once a song has both a key label and a
fixture clip, detect_key is scored against your ear.

Skips until at least one labeled fixture exists.
"""

import csv
from pathlib import Path

import librosa
import pytest

from musicbot.processing.audio_analysis import detect_key

from .conftest import load_baseline, load_manifest, save_baseline

pytestmark = pytest.mark.benchmark

KEY_LABELS = Path(__file__).resolve().parents[2] / "data" / "key_labels.csv"


def _load_key_labels() -> dict[str, str]:
    if not KEY_LABELS.is_file():
        return {}
    out = {}
    with open(KEY_LABELS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            song = (row.get("Song") or "").strip()
            key = (row.get("Key") or "").strip()
            if song and key:
                out[song] = key
    return out


def test_key_detection_does_not_regress(update_baseline):
    labels = _load_key_labels()
    if not labels:
        pytest.skip(
            "No key labels yet. After trimming a song, thumbs-up/down the "
            "detected key in the result panel — that grows data/key_labels.csv."
        )

    # Match labels to fixture clips by song name
    by_song = {e["song"]: e for e in load_manifest()}
    paired = [(song, key, by_song[song]) for song, key in labels.items() if song in by_song]
    if not paired:
        pytest.skip(
            f"{len(labels)} key label(s) but none have fixture clips yet. "
            "Promote/regenerate fixtures for labeled songs."
        )

    hits = 0
    misses = []
    for song, truth, entry in paired:
        y, sr = librosa.load(str(entry["raw"]), sr=None, mono=True)
        detected, _camelot = detect_key(y, sr)
        if detected == truth:
            hits += 1
        else:
            misses.append(f"{song}: expected {truth}, got {detected}")

    current = {"n": len(paired), "hits": hits}
    print(f"\n  [key] {hits}/{len(paired)} exact key matches")
    for m in misses:
        print(f"    miss: {m}")

    baseline = load_baseline()

    if update_baseline:
        baseline["key"] = current
        save_baseline(baseline)
        pytest.skip(f"baseline updated for key: {current}")

    floor = baseline.get("key")
    if floor is None:
        pytest.fail("No key baseline. Run: pytest tests/benchmark --update-baseline")

    assert current["hits"] >= floor["hits"], (
        f"Key hit count regressed: {current['hits']}/{current['n']} vs "
        f"baseline {floor['hits']}/{floor['n']}"
    )
