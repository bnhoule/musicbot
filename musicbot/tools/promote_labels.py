"""Promote web trim picks into the kick benchmark ground truth.

Every pick made in the web trim UI lands in data/trim_choices.json. This
tool reviews those picks and appends the ones you trust to
data/kick_labels.csv, then regenerates the fixture clip so the benchmark
grows as you use the product.

Usage
-----
    # See which web picks aren't in the benchmark yet
    python musicbot/tools/promote_labels.py --list

    # Promote one song (song name as it should appear in kick_labels.csv)
    python musicbot/tools/promote_labels.py --promote "122 - Purple Line.mp3" \
        --songs-dir ~/Desktop/Annotated

    # Promote everything not yet labeled
    python musicbot/tools/promote_labels.py --promote-all --songs-dir ~/Desktop/Annotated

Fixture regeneration needs the source MP3 (--songs-dir) and a cached
Demucs drums stem in eval_cache/ — songs without one are promoted to the
CSV but skipped for fixtures until you run the pipeline on them.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
LABELS_FILE = DATA_DIR / "kick_labels.csv"
CHOICES_FILE = DATA_DIR / "trim_choices.json"
CACHE_DIR = REPO_ROOT / "eval_cache"
FIXTURE_TOOL = Path(__file__).parent / "make_bench_fixtures.py"


def clean_song_name(filename: str) -> str:
    """'122 - Purple Line.mp3' → 'Purple Line' (matches kick_labels.csv style)."""
    stem = Path(filename).stem
    return re.sub(r"^\d+\s*[-–—]\s*", "", stem).strip()


def load_labeled_songs() -> set[str]:
    if not LABELS_FILE.is_file():
        return set()
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        return {row["Song"].strip().lower() for row in csv.DictReader(f)}


def load_choices() -> dict:
    with open(CHOICES_FILE, encoding="utf-8") as f:
        return json.load(f)


def unpromoted_choices() -> list[tuple[str, str, float]]:
    """Return (choice_key, clean_name, trim_sec) for picks not yet in the CSV."""
    labeled = load_labeled_songs()
    out = []
    for key, entry in sorted(load_choices().items()):
        name = clean_song_name(key)
        if name.lower() not in labeled:
            out.append((key, name, float(entry["trim_sec"])))
    return out


def append_label(song_name: str, kick_sec: float) -> None:
    with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([song_name, kick_sec])
    print(f"  + kick_labels.csv: {song_name},{kick_sec}")


def regen_fixture(song_name: str, songs_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(FIXTURE_TOOL),
         "--songs-dir", str(songs_dir), "--song", song_name],
        capture_output=True, text=True,
    )
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines():
        print(f"    {line}")
    if result.returncode != 0:
        print(f"    [warn] fixture generation failed for '{song_name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote web trim picks to benchmark labels")
    parser.add_argument("--list", action="store_true", help="Show unpromoted web picks")
    parser.add_argument("--promote", metavar="CHOICE_KEY",
                        help="Promote one pick (key as shown by --list)")
    parser.add_argument("--promote-all", action="store_true", help="Promote every unpromoted pick")
    parser.add_argument("--songs-dir", type=Path, default=Path.home() / "Desktop" / "Annotated",
                        help="Directory containing the source MP3s (for fixture regen)")
    args = parser.parse_args()

    pending = unpromoted_choices()

    if args.list or not (args.promote or args.promote_all):
        if not pending:
            print("  All web picks are already in kick_labels.csv")
            return
        print(f"  {len(pending)} unpromoted pick(s):\n")
        for key, name, trim in pending:
            print(f"    {key:<55} → {name:<30} {trim:>9.3f}s")
        print("\n  Promote with: --promote \"<choice key>\"  or  --promote-all")
        return

    if args.promote:
        matches = [p for p in pending if p[0] == args.promote]
        if not matches:
            sys.exit(f"[error] '{args.promote}' not found among unpromoted picks (see --list)")
        to_promote = matches
    else:
        to_promote = pending

    for _key, name, trim in to_promote:
        append_label(name, trim)
        mp3_exists = any(args.songs_dir.glob("*.mp3")) if args.songs_dir.is_dir() else False
        if mp3_exists:
            regen_fixture(name, args.songs_dir)
        else:
            print(f"    [skip fixture] songs dir not found: {args.songs_dir}")

    print("\n  Done. Refresh the benchmark floor in the same PR:")
    print("    pytest tests/benchmark --update-baseline")


if __name__ == "__main__":
    main()
