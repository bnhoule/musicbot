"""musicbot -- remix preparation pipeline.

Splits a song into stems (via Demucs locally or Lalal.ai remotely),
uses madmom RNN downbeat detection to find trim-point candidates,
lets the user interactively pick the right downbeat (or auto-picks),
then trims all stems and writes analysis metadata.

    processed/
      <song_name>_<run_id>_madmom/
        stems/
          vocals.wav  drums.wav  bass.wav  other.wav
        metadata.json

Usage
-----
    python process_song.py track.mp3               # interactive picker
    python process_song.py track.mp3 --auto         # auto-pick highest energy
    python process_song.py track.mp3 --repick       # override a saved choice
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from processing.audio_analysis import analyze, trim_stems_to_onset
from processing.madmom_beats import get_downbeats, rank_candidates
from processing.trim_picker import (
    load_trim_choices, save_trim_choice,
    pick_trim_auto, pick_trim_interactive,
)
from utils import build_song_dirs, save_json

SUPPORTED_EXTENSIONS = {".mp3", ".wav"}
DEFAULT_OUTPUT_DIR = "processed"
TRIM_METHOD = "madmom"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run(
    song_path: str,
    output_base: str,
    backend: str = "demucs",
    auto: bool = False,
    repick: bool = False,
    api_key: str | None = None,
    extraction_level: str = "deep_extraction",
) -> None:
    song = Path(song_path).resolve()

    if not song.exists():
        sys.exit(f"[error] File not found: {song_path}")
    if song.suffix.lower() not in SUPPORTED_EXTENSIONS:
        sys.exit(
            f"[error] Unsupported format '{song.suffix}'. "
            f"Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    song_dir, stems_dir = build_song_dirs(output_base, song.stem, method_tag=TRIM_METHOD)

    _banner(f"musicbot  ·  {song.name}")
    print(f"  Run folder: {song_dir.name}")

    # ── Step 1: Stem separation ─────────────────────────────────────────────
    if backend == "demucs":
        _step(1, 5, "Separating stems via Demucs (local)")
        from backends.demucs_separator import separate
        separate(str(song), str(stems_dir))
    else:
        _step(1, 5, "Separating stems via Lalal.ai")
        if not api_key:
            sys.exit(
                "[error] Lalal.ai backend requires an API key.\n"
                "  Pass --api-key KEY  or  set the LALAL_API_KEY env var."
            )
        from backends.lalal_api import LalalClient
        client = LalalClient(api_key)
        client.process_and_download_stems(
            str(song),
            str(stems_dir),
            extraction_level=extraction_level,
        )

    # ── Step 2: Detect downbeats via madmom ─────────────────────────────────
    _step(2, 5, "Detecting downbeats via madmom RNN")
    song_key = song.name
    choices = load_trim_choices()

    if song_key in choices and not repick:
        trim_offset = choices[song_key]["trim_sec"]
        prev_method = choices[song_key].get("method", "?")
        print(f"  Using saved trim: {trim_offset * 1000:.1f} ms "
              f"(from {prev_method}, use --repick to override)")
    else:
        drums_path = str(stems_dir / "drums.wav")
        downbeats, bpm = get_downbeats(str(song))
        print(f"  Found {len(downbeats)} downbeats at ~{bpm} BPM")

        candidates = rank_candidates(downbeats, drums_path, bpm=bpm)
        print(f"  Ranked top {len(candidates)} candidates by kick energy")

        if auto:
            trim_offset = pick_trim_auto(candidates)
            method = "auto"
            print(f"  Auto-picked: {trim_offset * 1000:.1f} ms")
        else:
            trim_offset = pick_trim_interactive(
                candidates, song.name, bpm, drums_path
            )
            method = "interactive"

        save_trim_choice(song_key, trim_offset, method=method)

    # ── Step 3: Trim stems ──────────────────────────────────────────────────
    _step(3, 5, "Trimming all stems")
    if trim_offset > 0.0:
        print(f"  Trim point: {trim_offset * 1000:.1f} ms")
        trim_stems_to_onset(str(stems_dir), trim_offset)
    else:
        print("  Trim point is 0 — stems already start at beat 1.")

    # ── Step 4: Audio analysis ──────────────────────────────────────────────
    _step(4, 5, "Analysing audio (BPM + key + beat grid)")
    analysis = analyze(str(song))
    print(
        f"  BPM: {analysis['bpm']}  │  "
        f"Key: {analysis['key']}  │  "
        f"Camelot: {analysis['camelot']}  │  "
        f"Beats: {len(analysis['beat_times'])}"
    )

    # ── Step 5: Metadata ────────────────────────────────────────────────────
    _step(5, 5, "Saving metadata.json")
    metadata = {
        "original_filename":   song.name,
        "run_folder":          song_dir.name,
        "trim_method":         TRIM_METHOD,
        "bpm":                 analysis["bpm"],
        "key":                 analysis["key"],
        "camelot_key":         analysis["camelot"],
        "beat_times":          [round(t, 4) for t in analysis["beat_times"]],
        "trim_offset_seconds": round(trim_offset, 4),
        "backend":             backend,
        "timestamp_processed": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = song_dir / "metadata.json"
    save_json(metadata, meta_path)
    print(f"  Saved → {meta_path}")

    _banner(f"Done!  Output → {song_dir}", char="─")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process_song.py",
        description="musicbot – remix preparation pipeline with interactive trim picker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python process_song.py track.mp3\n"
            "  python process_song.py track.mp3 --auto\n"
            "  python process_song.py track.mp3 --repick\n"
            "  python process_song.py track.mp3 --backend lalal --api-key KEY"
        ),
    )
    parser.add_argument(
        "song",
        metavar="SONG",
        help="Path to the input .mp3 or .wav file",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output base directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-pick the highest-energy downbeat (no interactive prompt)",
    )
    parser.add_argument(
        "--repick",
        action="store_true",
        help="Override a previously saved trim choice for this song",
    )
    parser.add_argument(
        "--backend",
        choices=["demucs", "lalal"],
        default="demucs",
        help="Stem separation backend (default: demucs, runs locally)",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        default=os.environ.get("LALAL_API_KEY"),
        help="Lalal.ai API key — only needed with --backend lalal",
    )
    parser.add_argument(
        "--extraction-level",
        metavar="LEVEL",
        default="deep_extraction",
        choices=["deep_extraction", "clear_cut"],
        help=(
            "Lalal.ai extraction quality (ignored with demucs): "
            "deep_extraction (default) or clear_cut"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.backend == "lalal" and not args.api_key:
        parser.error(
            "Lalal.ai backend requires an API key.\n"
            "  Pass --api-key KEY  or  set the LALAL_API_KEY environment variable."
        )

    run(
        args.song,
        args.output,
        backend=args.backend,
        auto=args.auto,
        repick=args.repick,
        api_key=args.api_key,
        extraction_level=args.extraction_level,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _banner(message: str, width: int = 60, char: str = "=") -> None:
    print(f"\n{char * width}")
    print(f"  {message}")
    print(f"{char * width}\n")


def _step(current: int, total: int, label: str) -> None:
    print(f"[{current}/{total}] {label}…")


if __name__ == "__main__":
    main()
