"""Layer 3: BPM detection benchmark.

The song filenames encode the true BPM ("126 - Song.mp3") — free ground
truth. A hit means detect_bpm lands within tolerance of the filename BPM,
allowing half/double-time octave folds. Ratchets on hit count.
"""

import librosa
import pytest

from musicbot.processing.audio_analysis import detect_bpm

from .conftest import load_baseline, load_manifest, save_baseline

pytestmark = pytest.mark.benchmark

BPM_TOLERANCE = 3.0  # absolute BPM, after octave folding


def bpm_hit(detected: float, truth: float) -> bool:
    return any(abs(c - truth) <= BPM_TOLERANCE for c in (detected, detected * 2, detected / 2))


def test_bpm_detection_does_not_regress(update_baseline):
    manifest = [e for e in load_manifest() if e["bpm"]]
    assert manifest, "No fixtures with filename BPM ground truth"

    hits = 0
    misses = []
    for entry in manifest:
        y, sr = librosa.load(str(entry["raw"]), sr=None, mono=True)
        detected = detect_bpm(y, sr)
        if bpm_hit(detected, entry["bpm"]):
            hits += 1
        else:
            misses.append(f"{entry['song']}: expected {entry['bpm']}, got {detected}")

    current = {"n": len(manifest), "hits": hits}
    print(f"\n  [bpm] {hits}/{len(manifest)} within ±{BPM_TOLERANCE} BPM (octave-aware)")
    for m in misses:
        print(f"    miss: {m}")

    baseline = load_baseline()

    if update_baseline:
        baseline["bpm"] = current
        save_baseline(baseline)
        pytest.skip(f"baseline updated for bpm: {current}")

    floor = baseline.get("bpm")
    if floor is None:
        pytest.fail("No BPM baseline recorded. Run: pytest tests/benchmark --update-baseline")

    assert current["hits"] >= floor["hits"], (
        f"BPM hit count regressed: {current['hits']}/{current['n']} vs "
        f"baseline {floor['hits']}/{floor['n']}"
    )
