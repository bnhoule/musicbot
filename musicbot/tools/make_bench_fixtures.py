"""Generate benchmark fixture clips from hand-labeled songs.

For every entry in data/kick_labels.csv this exports two small clips
covering 0 → label + TAIL_SEC seconds at 22.05 kHz mono:

  tests/fixtures/kick_bench/<slug>__raw.wav     (PCM_16 — madmom reads WAV natively)
  tests/fixtures/kick_bench/<slug>__drums.flac  (FLAC — read via soundfile/librosa)

plus a manifest ``tests/fixtures/kick_bench/labels.csv`` with the label and
the BPM parsed from the source filename.  The clips start at 0 so detection
context matches full songs exactly.

Run locally once (requires the annotated MP3s and the Demucs eval cache):

    python musicbot/tools/make_bench_fixtures.py \
        --songs-dir ~/Desktop/Annotated
"""

import argparse
import csv
import sys
from pathlib import Path

import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import parse_bpm_from_filename, sanitize_filename  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
LABELS_FILE = DATA_DIR / "kick_labels.csv"
CACHE_DIR = REPO_ROOT / "eval_cache"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "kick_bench"

FIXTURE_SR = 22050
TAIL_SEC = 15.0
MIN_CLIP_SEC = 20.0  # even kick-at-zero songs get enough context


def load_labels() -> list[dict]:
    rows = []
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "song": row["Song"].strip(),
                "kick_sec": float(row["Kick Start (seconds)"].strip()),
            })
    return rows


def match_file(song_name: str, songs_dir: Path) -> Path | None:
    import re
    name_lower = song_name.lower()
    for mp3 in sorted(songs_dir.glob("*.mp3")):
        stem = re.sub(r"^\d+\s*-\s*", "", mp3.stem)
        if name_lower in stem.lower():
            return mp3
    return None


def export_clip(src: Path, dest: Path, duration: float) -> None:
    y, _ = librosa.load(str(src), sr=FIXTURE_SR, mono=True, duration=duration)
    fmt = "FLAC" if dest.suffix == ".flac" else "WAV"
    sf.write(str(dest), y, FIXTURE_SR, subtype="PCM_16", format=fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build kick benchmark fixture clips")
    parser.add_argument("--songs-dir", type=Path, default=Path.home() / "Desktop" / "Annotated")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--song", help="Only regenerate fixtures for this song name")
    parser.add_argument("--tail-sec", type=float, default=TAIL_SEC)
    args = parser.parse_args()

    if not args.songs_dir.is_dir():
        sys.exit(f"[error] Songs directory not found: {args.songs_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "labels.csv"

    labels = load_labels()
    if args.song:
        labels = [row for row in labels if row["song"].lower() == args.song.lower()]
        if not labels:
            sys.exit(f"[error] No label found for song: {args.song}")

    # Start from the existing manifest so single-song regeneration keeps others
    manifest: dict[str, dict] = {}
    if manifest_path.is_file():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manifest[row["song"]] = row

    for entry in labels:
        song, kick_sec = entry["song"], entry["kick_sec"]
        mp3 = match_file(song, args.songs_dir)
        if mp3 is None:
            print(f"  [skip] No MP3 found for '{song}'")
            continue

        drums_src = CACHE_DIR / mp3.stem / "stems" / "drums.wav"
        if not drums_src.is_file():
            print(f"  [skip] No cached drums stem for '{song}' "
                  f"(run evaluate_kicks first to build the Demucs cache)")
            continue

        slug = sanitize_filename(song)
        duration = max(kick_sec + args.tail_sec, MIN_CLIP_SEC)

        raw_dest = args.out / f"{slug}__raw.wav"
        drums_dest = args.out / f"{slug}__drums.flac"

        print(f"  {song:<25} clip 0→{duration:.1f}s", flush=True)
        export_clip(mp3, raw_dest, duration)
        export_clip(drums_src, drums_dest, duration)

        manifest[song] = {
            "song": song,
            "slug": slug,
            "kick_sec": f"{kick_sec}",
            "bpm": f"{parse_bpm_from_filename(mp3.name) or ''}",
            "clip_sec": f"{duration:.1f}",
        }

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["song", "slug", "kick_sec", "bpm", "clip_sec"])
        writer.writeheader()
        for song in sorted(manifest):
            writer.writerow(manifest[song])

    print(f"\n  Manifest: {manifest_path}  ({len(manifest)} songs)")


if __name__ == "__main__":
    main()
