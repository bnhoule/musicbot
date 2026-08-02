"""Audition the kick-label ground truth by ear.

Plays each labeled song starting exactly at its labeled kick time — the
same experience as dropping the trimmed stem into a DAW. If the label is
right, playback opens on the kick transient with zero dead air.

Commands during the session:

    Enter     label is correct — mark verified
    r         replay from the label point
    c         replay with 1.5s of context before the label
    d         replay the drums stem instead of the raw mix
    t 43.71   correct the label to a new time (then replays for confirmation)
    s         skip this song (leave unverified)
    q         quit (progress is saved after every song)

After a session, if any labels changed:

    python musicbot/tools/make_bench_fixtures.py --songs-dir ~/Desktop/Annotated
    pytest tests/benchmark --update-baseline    # in a PR

Usage:

    python musicbot/tools/audition_labels.py            # audition all
    python musicbot/tools/audition_labels.py --song "Run Away"
    python musicbot/tools/audition_labels.py --unverified-only
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import sanitize_filename  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = REPO_ROOT / "data" / "kick_labels.csv"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "kick_bench"

PREVIEW_SEC = 4.0
CONTEXT_SEC = 1.5

FIELDNAMES = ["Song", "Kick Start (seconds)", "Verified"]


def load_labels() -> list[dict]:
    rows = []
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "Song": row["Song"].strip(),
                "Kick Start (seconds)": row["Kick Start (seconds)"].strip(),
                "Verified": (row.get("Verified") or "").strip(),
            })
    return rows


def save_labels(rows: list[dict]) -> None:
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def play_snippet(audio_path: Path, start_sec: float, duration: float = PREVIEW_SEC) -> bool:
    """Cut a snippet starting at start_sec and play it. Returns False if no player."""
    y, sr = sf.read(str(audio_path), dtype="float64")
    start = max(0, int(start_sec * sr))
    end = min(len(y), start + int(duration * sr))
    if start >= len(y):
        print("    [label beyond end of clip!]")
        return True

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y[start:end], sr)
        tmp_path = tmp.name

    for cmd in (["afplay", tmp_path], ["ffplay", "-nodisp", "-autoexit", tmp_path]):
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            Path(tmp_path).unlink(missing_ok=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    Path(tmp_path).unlink(missing_ok=True)
    print("    [no audio player found — need afplay or ffplay]")
    return False


def audition(rows: list[dict], only_song: str | None, unverified_only: bool) -> None:
    todo = [
        r for r in rows
        if (not only_song or only_song.lower() in r["Song"].lower())
        and (not unverified_only or not r["Verified"])
    ]
    if not todo:
        print("  Nothing to audition.")
        return

    print(f"\n  Auditioning {len(todo)} label(s). Playback starts AT the label —")
    print("  a correct label means you hear the kick instantly, no dead air.\n")

    verified = changed = 0

    for row in todo:
        song = row["Song"]
        label = float(row["Kick Start (seconds)"])
        slug = sanitize_filename(song)
        raw = FIXTURES_DIR / f"{slug}__raw.wav"
        drums = FIXTURES_DIR / f"{slug}__drums.flac"

        if not raw.is_file():
            print(f"  [{song}] no fixture clip — run make_bench_fixtures.py first")
            continue

        status = f" (verified {row['Verified']})" if row["Verified"] else ""
        print(f"  ▶ {song}  —  label {label:.3f}s{status}")
        play_snippet(raw, label)

        while True:
            try:
                cmd = input("    [Enter=good  r=replay  c=context  d=drums  "
                            "t <sec>=fix  s=skip  q=quit] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                save_labels(rows)
                return

            if cmd == "":
                row["Verified"] = date.today().isoformat()
                verified += 1
                save_labels(rows)
                break
            elif cmd == "r":
                play_snippet(raw, label)
            elif cmd == "c":
                play_snippet(raw, max(0.0, label - CONTEXT_SEC),
                             duration=PREVIEW_SEC + CONTEXT_SEC)
            elif cmd == "d":
                if drums.is_file():
                    play_snippet(drums, label)
                else:
                    print("    [no drums fixture]")
            elif cmd.startswith("t "):
                try:
                    new_label = float(cmd[2:].strip())
                except ValueError:
                    print("    Usage: t 43.71")
                    continue
                print(f"    New label {new_label:.3f}s — replaying…")
                play_snippet(raw, new_label)
                label = new_label
                row["Kick Start (seconds)"] = f"{new_label}"
                row["Verified"] = date.today().isoformat()
                changed += 1
                save_labels(rows)
            elif cmd == "s":
                break
            elif cmd == "q":
                save_labels(rows)
                _summary(verified, changed)
                return
            else:
                print("    ?")

    save_labels(rows)
    _summary(verified, changed)


def _summary(verified: int, changed: int) -> None:
    print(f"\n  Session done: {verified} confirmed, {changed} corrected.")
    if changed:
        print("\n  Labels changed — refresh fixtures and baseline in a PR:")
        print("    python musicbot/tools/make_bench_fixtures.py --songs-dir ~/Desktop/Annotated")
        print("    pytest tests/benchmark --update-baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audition kick labels by ear")
    parser.add_argument("--song", help="Only audition songs matching this name")
    parser.add_argument("--unverified-only", action="store_true",
                        help="Skip labels already marked verified")
    parser.add_argument("--status", action="store_true",
                        help="Print verification status and exit")
    args = parser.parse_args()

    rows = load_labels()

    if args.status:
        print(f"\n  {'Song':<28} {'Label (s)':>10}  Verified")
        print("  " + "-" * 52)
        for r in rows:
            print(f"  {r['Song']:<28} {float(r['Kick Start (seconds)']):>10.3f}  "
                  f"{r['Verified'] or '—'}")
        n_ver = sum(1 for r in rows if r["Verified"])
        print(f"\n  {n_ver}/{len(rows)} verified")
        return

    audition(rows, args.song, args.unverified_only)


if __name__ == "__main__":
    main()
