"""Local stem separation via Demucs (Meta's hybrid transformer model).

Runs entirely on-device — no API key, no upload/download latency.
Uses MPS acceleration on Apple Silicon when available.

Requires: ``pip install demucs``
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STEM_NAMES = ("drums", "bass", "vocals", "other")
DEFAULT_MODEL = "htdemucs"


def separate(
    file_path: str,
    stems_dir: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, str]:
    """Separate *file_path* into stems and write WAVs to *stems_dir*.

    Returns a mapping of stem name -> output path.
    """
    song = Path(file_path).resolve()
    out = Path(stems_dir)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="demucs_") as tmp:
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", model,
            "--out", tmp,
            str(song),
        ]
        print(f"  Running Demucs ({model}) on '{song.name}'…")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            err = result.stderr.strip().splitlines()
            tail = "\n".join(err[-10:]) if err else "(no stderr)"
            raise RuntimeError(
                f"Demucs failed (exit {result.returncode}):\n{tail}"
            )

        demucs_out = Path(tmp) / model / song.stem
        if not demucs_out.is_dir():
            candidates = list(Path(tmp).rglob("vocals.wav"))
            if candidates:
                demucs_out = candidates[0].parent
            else:
                raise FileNotFoundError(
                    f"Demucs output not found. Expected: {demucs_out}\n"
                    f"Contents of tmp: {list(Path(tmp).rglob('*'))}"
                )

        downloaded: dict[str, str] = {}
        for stem in STEM_NAMES:
            src = demucs_out / f"{stem}.wav"
            dest = out / f"{stem}.wav"
            if src.is_file():
                shutil.copy2(str(src), str(dest))
                downloaded[stem] = str(dest)
                print(f"  {stem}.wav ✓")
            else:
                print(f"  {stem}.wav — not found in Demucs output, skipping")

    return downloaded
