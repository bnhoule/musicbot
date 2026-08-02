#!/usr/bin/env python3
"""Batch pre-process a folder of songs through Lalal.ai + madmom.

Uses the Lalal.ai API (batch stem separator endpoint) to split each song
into 5 stems: drums, bass, vocals, piano, synth — plus the instrumental
"other" back-track from the vocals split.

Results are cached alongside existing Demucs data in:
    cache/<song>/lalal.json     (manifest)
    cache/<song>/stems_lalal/   (WAV stems)

Usage:
    python -m musicbot.tools.batch_preprocess_lalal /path/to/songs/ \\
        --api-key YOUR_KEY --cache-dir ./cache

    # Or via env var:
    export LALAL_API_KEY=YOUR_KEY
    python -m musicbot.tools.batch_preprocess_lalal /path/to/songs/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends.lalal_api import LalalClient, LALAL_STEMS_5
from processing.audio_analysis import analyze
from processing.madmom_beats import get_downbeats, rank_candidates, score_all_downbeats

SUPPORTED = {".mp3", ".wav", ".flac", ".aiff", ".ogg"}
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "cache"


def _cache_key(filename: str) -> str:
    return Path(filename).stem.strip()


def process_one(
    song_path: Path,
    cache_dir: Path,
    client: LalalClient,
    extraction_level: str,
) -> bool:
    """Process a single song via Lalal. Returns True if newly processed."""
    key = _cache_key(song_path.name)
    song_cache = cache_dir / key
    manifest = song_cache / "lalal.json"

    if manifest.is_file():
        print(f"  ✓ cached   {song_path.name}")
        return False

    print(f"  → processing {song_path.name}")
    stems_dir = song_cache / "stems_lalal"
    stems_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    print("    [1/3] Lalal.ai stem separation (5 stems)…")
    client.process_and_download_stems_batch(
        str(song_path),
        str(stems_dir),
        stems=LALAL_STEMS_5,
        extraction_level=extraction_level,
        delete_after=True,
    )

    print("    [2/3] Madmom downbeat detection…")
    downbeats, bpm_madmom = get_downbeats(str(song_path))
    drums_path = str(stems_dir / "drums.wav")
    candidates = rank_candidates(downbeats, drums_path, bpm=bpm_madmom)
    all_db = score_all_downbeats(downbeats, drums_path, bpm=bpm_madmom)

    print("    [3/3] Audio analysis (BPM, key)…")
    analysis_result = analyze(str(song_path))

    stem_files = [f.stem for f in stems_dir.iterdir() if f.suffix == ".wav"]

    data = {
        "filename": song_path.name,
        "song_path": str(song_path.resolve()),
        "stems_dir": str(stems_dir.resolve()),
        "backend": "lalal",
        "stems": sorted(stem_files),
        "bpm": bpm_madmom,
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
    print(f"    done in {elapsed:.1f}s — {len(candidates)} candidates, {len(stem_files)} stems")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch pre-process songs via Lalal.ai for the trim picker / stacker"
    )
    parser.add_argument("input_dir", help="Folder of MP3/WAV files to process")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                        help="Where to store cached results")
    parser.add_argument("--api-key", default=os.environ.get("LALAL_API_KEY", ""),
                        help="Lalal.ai license key (or set LALAL_API_KEY env var)")
    parser.add_argument("--extraction-level", default="clear_cut",
                        choices=["clear_cut", "deep_extraction"],
                        help="Lalal extraction level (default: clear_cut)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set LALAL_API_KEY env var")
        sys.exit(1)

    input_dir = Path(args.input_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    songs = sorted(
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED
    )
    if not songs:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    client = LalalClient(args.api_key)

    try:
        mins = client.minutes_left()
        print(f"\nLalal.ai account: {mins:.1f} fast minutes remaining")
    except Exception as e:
        print(f"\nWarning: could not check minutes ({e})")

    print(f"Batch pre-processing {len(songs)} songs → {cache_dir}\n")

    processed = 0
    cached = 0
    errors = 0

    for i, song in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}]", end="")
        try:
            if process_one(song, cache_dir, client, args.extraction_level):
                processed += 1
            else:
                cached += 1
        except Exception as e:
            errors += 1
            print(f"  ✗ ERROR on {song.name}: {e}")

        if processed > 0 and processed % 5 == 0:
            try:
                mins = client.minutes_left()
                print(f"  ℹ {mins:.1f} fast minutes remaining")
            except Exception:
                pass

    print(f"\nDone: {processed} processed, {cached} already cached, {errors} errors")


if __name__ == "__main__":
    main()
