#!/usr/bin/env python3
"""Batch pre-process a folder of songs through Demucs + madmom.

Writes stems and a cache.json manifest to the cache directory so the
trim-picker web app can load them instantly without re-processing.

Usage:
    python -m musicbot.tools.batch_preprocess /path/to/songs/
    python -m musicbot.tools.batch_preprocess /path/to/songs/ --cache-dir ./my_cache
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends.demucs_separator import separate
from processing.audio_analysis import analyze
from processing.madmom_beats import get_downbeats, rank_candidates, score_all_downbeats
from utils import parse_bpm_from_filename

SUPPORTED = {".mp3", ".wav", ".flac", ".aiff", ".ogg"}
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "cache"


def _cache_key(filename: str) -> str:
    """Normalize a filename into a stable cache key."""
    return Path(filename).stem.strip()


def process_one(song_path: Path, cache_dir: Path) -> bool:
    """Process a single song. Returns True if newly processed, False if cached."""
    key = _cache_key(song_path.name)
    song_cache = cache_dir / key
    manifest = song_cache / "cache.json"

    if manifest.is_file():
        print(f"  ✓ cached   {song_path.name}")
        return False

    print(f"  → processing {song_path.name}")
    stems_dir = song_cache / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    print("    [1/3] Demucs stem separation…")
    separate(str(song_path), str(stems_dir))

    print("    [2/3] Madmom downbeat detection…")
    downbeats, bpm_madmom = get_downbeats(str(song_path))
    title_bpm = parse_bpm_from_filename(song_path.name)
    bpm = title_bpm if title_bpm else bpm_madmom
    src = "filename" if title_bpm else "madmom"
    print(f"  Loading audio for analysis: {song_path}")
    print(f"  BPM: {bpm}  (source: {src}  madmom={bpm_madmom})")
    drums_path = str(stems_dir / "drums.wav")
    candidates = rank_candidates(downbeats, drums_path, bpm=bpm)
    all_db = score_all_downbeats(downbeats, drums_path, bpm=bpm)

    print("    [3/3] Audio analysis (BPM, key)…")
    analysis_result = analyze(str(song_path))

    data = {
        "filename": song_path.name,
        "song_path": str(song_path.resolve()),
        "stems_dir": str(stems_dir.resolve()),
        "bpm": bpm,
        "bpm_madmom": bpm_madmom,
        "analysis": {
            "bpm": analysis_result["bpm"],
            "key": analysis_result["key"],
            "camelot": analysis_result["camelot"],
        },
        "candidates": [
            {"time_sec": c.time_sec, "energy_pct": round(c.energy_pct, 1)}
            for c in candidates
        ],
        "all_downbeats": all_db,
    }

    with open(manifest, "w") as f:
        json.dump(data, f, indent=2)

    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s — {len(candidates)} candidates")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch pre-process songs for the trim picker")
    parser.add_argument("input_dir", help="Folder of MP3/WAV files to process")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="Where to store cached results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    songs = sorted(f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED)
    if not songs:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    print(f"\nBatch pre-processing {len(songs)} songs → {cache_dir}\n")

    processed = 0
    cached = 0
    errors = 0

    for i, song in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}]", end="")
        try:
            if process_one(song, cache_dir):
                processed += 1
            else:
                cached += 1
        except Exception as e:
            errors += 1
            print(f"  ✗ ERROR on {song.name}: {e}")

    print(f"\nDone: {processed} processed, {cached} already cached, {errors} errors")


if __name__ == "__main__":
    main()
