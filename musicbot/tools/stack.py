"""Interactive stem-stack shuffler.

Scans processed song folders, lets you randomly audition combinations of
stems from different songs — one per category (drums, bass, vocals, other).
All stems are rekeyed and tempo-matched to a common target before mixing.

Usage
-----
    python stack.py --input processed/ --target-bpm 128 --target-key "A minor"
    python stack.py --input processed/  # uses first-picked song's bpm/key
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

# Ensure musicbot/ is on the path when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from processing.rekey import semitone_distance, rekey_audio
from processing.tempo_match import stretch_audio

CATEGORIES = ("drums", "bass", "vocals", "other")
HEADROOM_DB = -1.0  # peak-normalize target


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StemInfo:
    """One stem file + the metadata of the song it came from."""
    path: str
    song_name: str
    bpm: float
    key: str
    camelot: str


@dataclass
class StemSlot:
    """Shuffle state for a single stem category."""
    category: str
    pool: list[StemInfo]
    history: list[int] = field(default_factory=list)
    cursor: int = -1

    @property
    def current_index(self) -> int | None:
        if not self.history:
            return None
        return self.history[self.cursor]

    @property
    def current(self) -> StemInfo | None:
        idx = self.current_index
        return self.pool[idx] if idx is not None else None

    def _unvisited(self) -> list[int]:
        visited = set(self.history)
        return [i for i in range(len(self.pool)) if i not in visited]

    def next(self) -> StemInfo | None:
        """Advance to the next random stem, or step forward in history."""
        if len(self.pool) == 0:
            return None

        if self.cursor < len(self.history) - 1:
            self.cursor += 1
            return self.current

        remaining = self._unvisited()
        if not remaining:
            self.history.clear()
            remaining = list(range(len(self.pool)))

        pick = random.choice(remaining)
        self.history.append(pick)
        self.cursor = len(self.history) - 1
        return self.current

    def prev(self) -> StemInfo | None:
        """Step backward in history."""
        if self.cursor > 0:
            self.cursor -= 1
        return self.current


# ---------------------------------------------------------------------------
# Scanning processed songs
# ---------------------------------------------------------------------------

def scan_songs(input_dir: str) -> dict[str, list[StemInfo]]:
    """Build per-category pools from processed/<song>/stems/ folders."""
    pools: dict[str, list[StemInfo]] = {cat: [] for cat in CATEGORIES}
    base = Path(input_dir)

    for meta_path in sorted(base.glob("*/metadata.json")):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        song_dir = meta_path.parent
        stems_dir = song_dir / "stems"
        song_name = song_dir.name

        for cat in CATEGORIES:
            stem_file = stems_dir / f"{cat}.wav"
            if stem_file.is_file():
                pools[cat].append(StemInfo(
                    path=str(stem_file),
                    song_name=song_name,
                    bpm=meta.get("bpm", 120.0),
                    key=meta.get("key", "C major"),
                    camelot=meta.get("camelot_key", "8B"),
                ))

    return pools


# ---------------------------------------------------------------------------
# Audio processing helpers
# ---------------------------------------------------------------------------

def _prepare_stem(
    stem: StemInfo,
    target_bpm: float,
    target_key: str,
    work_dir: str,
) -> str:
    """Rekey + tempo-stretch a stem into *work_dir*, return path to result."""
    base_name = f"{stem.song_name}_{Path(stem.path).stem}"
    out_path = os.path.join(work_dir, f"{base_name}.wav")

    semitones = semitone_distance(stem.key, target_key)

    if semitones != 0:
        rekeyed = os.path.join(work_dir, f"{base_name}_rekeyed.wav")
        rekey_audio(stem.path, semitones, rekeyed)
        src = rekeyed
    else:
        src = stem.path

    stretch_audio(src, stem.bpm, target_bpm, out_path)
    return out_path


def mix_stems(stem_paths: list[str], output_path: str) -> None:
    """Sum-and-normalize a list of WAV files to *output_path*.

    Peak-normalizes to HEADROOM_DB so the mix doesn't clip.
    """
    arrays: list[np.ndarray] = []
    sr_out: int | None = None

    for p in stem_paths:
        y, sr = sf.read(p, dtype="float64")
        if sr_out is None:
            sr_out = sr

        if y.ndim == 1:
            y = y[:, np.newaxis]
        arrays.append(y)

    if not arrays or sr_out is None:
        return

    min_len = min(a.shape[0] for a in arrays)
    max_ch = max(a.shape[1] for a in arrays)

    padded = []
    for a in arrays:
        a = a[:min_len]
        if a.shape[1] < max_ch:
            a = np.repeat(a, max_ch, axis=1)
        padded.append(a)

    mixed = sum(padded)

    peak = np.abs(mixed).max()
    if peak > 0:
        target_peak = 10.0 ** (HEADROOM_DB / 20.0)
        mixed = mixed * (target_peak / peak)

    sf.write(output_path, mixed, sr_out, subtype="PCM_24")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

_HELP = """
Commands (case-insensitive):
  d / D  — next / prev  DRUMS
  b / B  — next / prev  BASS
  v / V  — next / prev  VOCALS
  o / O  — next / prev  OTHER
  r      — re-roll ALL categories
  e      — export current stack
  ?      — show this help
  q      — quit
""".strip()

_NEXT_KEYS = {"d": "drums", "b": "bass", "v": "vocals", "o": "other"}
_PREV_KEYS = {"D": "drums", "B": "bass", "V": "vocals", "O": "other"}


def _display_current(slots: dict[str, StemSlot]) -> None:
    """Print the current stem selection."""
    print()
    print("┌─────────────────────────────────────────────┐")
    for cat in CATEGORIES:
        stem = slots[cat].current
        if stem:
            hist = slots[cat].history
            pos = slots[cat].cursor + 1
            print(f"  {cat:>6s}  │  {stem.song_name}  "
                  f"({stem.key}, {stem.bpm} BPM)  [{pos}/{len(hist)}]")
        else:
            print(f"  {cat:>6s}  │  (empty pool)")
    print("└─────────────────────────────────────────────┘")


def run_interactive(
    input_dir: str,
    target_bpm: float | None,
    target_key: str | None,
    output_base: str,
) -> None:
    """Main interactive audition loop."""
    pools = scan_songs(input_dir)

    total = sum(len(p) for p in pools.values())
    if total == 0:
        sys.exit(
            f"[error] No processed songs found in {input_dir}.\n"
            "  Run process_song.py first to generate stems + metadata."
        )

    for cat in CATEGORIES:
        n = len(pools[cat])
        print(f"  {cat}: {n} stem{'s' if n != 1 else ''}")

    slots: dict[str, StemSlot] = {
        cat: StemSlot(category=cat, pool=pools[cat]) for cat in CATEGORIES
    }

    # Initial random pick for each category
    for slot in slots.values():
        slot.next()

    # Infer target from the first drums pick if not given
    first_stem = next(
        (s.current for s in slots.values() if s.current is not None), None
    )
    if target_bpm is None:
        target_bpm = first_stem.bpm if first_stem else 120.0
    if target_key is None:
        target_key = first_stem.key if first_stem else "C major"

    print(f"\n  Target: {target_key}  @  {target_bpm} BPM")

    work_dir = tempfile.mkdtemp(prefix="musicbot_stack_")
    preview_path = os.path.join(work_dir, "preview.wav")

    def _rebuild_mix() -> None:
        prepared: list[str] = []
        for cat in CATEGORIES:
            stem = slots[cat].current
            if stem is None:
                continue
            p = _prepare_stem(stem, target_bpm, target_key, work_dir)
            prepared.append(p)
        if prepared:
            mix_stems(prepared, preview_path)
            print(f"  Preview → {preview_path}")

    _rebuild_mix()
    _display_current(slots)
    print(f"\n{_HELP}\n")

    while True:
        try:
            cmd = input("stack> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        if cmd == "q":
            break
        elif cmd == "?":
            print(_HELP)
            continue
        elif cmd == "r":
            for slot in slots.values():
                slot.next()
            print("  Re-rolled all categories.")
            _rebuild_mix()
            _display_current(slots)
        elif cmd == "e":
            _export_stack(slots, target_bpm, target_key, work_dir, output_base)
        elif cmd in _NEXT_KEYS:
            cat = _NEXT_KEYS[cmd]
            stem = slots[cat].next()
            if stem:
                print(f"  {cat} → {stem.song_name}")
            else:
                print(f"  {cat}: pool is empty")
            _rebuild_mix()
            _display_current(slots)
        elif cmd in _PREV_KEYS:
            cat = _PREV_KEYS[cmd]
            stem = slots[cat].prev()
            if stem:
                print(f"  {cat} ← {stem.song_name}")
            else:
                print(f"  {cat}: already at start of history")
            _display_current(slots)
        else:
            print(f"  Unknown command: {cmd!r}  (press ? for help)")

    # Cleanup temp files
    shutil.rmtree(work_dir, ignore_errors=True)
    print("Done.")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export_stack(
    slots: dict[str, StemSlot],
    target_bpm: float,
    target_key: str,
    work_dir: str,
    output_base: str,
) -> None:
    """Write the current stack to stacks/<timestamp>/."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    stack_dir = Path(output_base) / "stacks" / ts
    stack_dir.mkdir(parents=True, exist_ok=True)

    exported_stems: list[str] = []
    manifest: dict[str, dict] = {}

    for cat in CATEGORIES:
        stem = slots[cat].current
        if stem is None:
            continue

        out = str(stack_dir / f"{cat}.wav")
        _prepare_stem(stem, target_bpm, target_key, work_dir)

        prepared_name = f"{stem.song_name}_{cat}"
        prepared_path = os.path.join(work_dir, f"{prepared_name}.wav")
        if os.path.isfile(prepared_path):
            shutil.copy2(prepared_path, out)
        else:
            _prepare_stem(stem, target_bpm, target_key, str(stack_dir))

        exported_stems.append(out)
        manifest[cat] = {
            "source_song": stem.song_name,
            "original_key": stem.key,
            "original_bpm": stem.bpm,
        }

    if exported_stems:
        mix_path = str(stack_dir / "stack.wav")
        mix_stems(exported_stems, mix_path)

    meta = {
        "target_bpm": target_bpm,
        "target_key": target_key,
        "stems": manifest,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with open(stack_dir / "stack.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  Exported → {stack_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stack.py",
        description="Interactive stem-stack shuffler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  python stack.py --input processed/ --target-bpm 128 --target-key "A minor"\n'
            "  python stack.py --input processed/\n"
        ),
    )
    parser.add_argument(
        "--input",
        metavar="DIR",
        default="processed",
        help="Directory containing processed song folders (default: processed/)",
    )
    parser.add_argument(
        "--target-bpm",
        metavar="BPM",
        type=float,
        default=None,
        help="Target BPM for all stems (default: use first pick's BPM)",
    )
    parser.add_argument(
        "--target-key",
        metavar="KEY",
        default=None,
        help='Target key for all stems, e.g. "A minor" (default: use first pick\'s key)',
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default="processed",
        help="Base directory for exported stacks (default: processed/)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_interactive(args.input, args.target_bpm, args.target_key, args.output)


if __name__ == "__main__":
    main()
