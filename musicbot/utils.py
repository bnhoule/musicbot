"""Shared utility helpers: filesystem, JSON, name sanitization, input validation."""

import json
import re
import uuid
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp3", ".wav"}


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and all parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(name: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
    return safe.strip("_")


def run_id() -> str:
    """Return a short 4-hex-char unique run identifier."""
    return uuid.uuid4().hex[:4]


def save_json(data: dict, path: str | Path) -> None:
    """Write *data* as indented JSON to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def resolve_input(path_str: str) -> Path:
    """Resolve an input path and validate it exists and is a supported format.

    Raises FileNotFoundError or ValueError on bad input.
    """
    p = Path(path_str).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{p.suffix}'. "
            f"Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return p


_BPM_PATTERNS = [
    # "122 - Song Name"  or  "122 – Song Name"  (leading number, most common)
    re.compile(r"^(\d{2,3})\s*[-–—]"),
    # "Song Name - 122"  or  "Song Name – 122"  (trailing number)
    re.compile(r"[-–—]\s*(\d{2,3})\s*$"),
    # "Song Name (122 BPM)"  or  "Song Name [122bpm]"
    re.compile(r"[\(\[](\d{2,3})\s*bpm[\)\]]", re.IGNORECASE),
    # "Song Name 122bpm"  or  "122bpm Song Name"
    re.compile(r"\b(\d{2,3})\s*bpm\b", re.IGNORECASE),
    # "Song Name_122_"  or  "Song Name 122 "  — number anchored to start/end of stem
    re.compile(r"^(\d{2,3})\b"),
]
_BPM_MIN = 60
_BPM_MAX = 220


def parse_bpm_from_filename(filename: str) -> float | None:
    """Try to extract a BPM value from a song filename.

    Tries several patterns in order of confidence.  Returns ``None`` if no
    plausible BPM is found.

    Examples::

        "122 - Bulletproof.mp3"          → 122.0
        "140 - Something New.mp3"        → 140.0
        "Song Name (128 BPM).flac"       → 128.0
        "artist - title 95bpm.wav"       → 95.0
    """
    stem = Path(filename).stem
    for pattern in _BPM_PATTERNS:
        for m in pattern.finditer(stem):
            val = int(m.group(1))
            if _BPM_MIN <= val <= _BPM_MAX:
                return float(val)
    return None


def clean_song_name(filename: str) -> str:
    """Strip a leading BPM prefix and extension: '122 - Run Away.mp3' → 'Run Away'.

    This is the canonical song identity used across the label files
    (``kick_labels.csv``, ``key_labels.csv``) and benchmark fixtures.
    """
    stem = Path(filename).stem
    return re.sub(r"^\d{2,3}\s*[-–—]\s*", "", stem).strip()


def build_song_dirs(output_base: str, song_name: str, method_tag: str = "") -> tuple[Path, Path]:
    """Return (song_dir, stems_dir) for a given song and create them on disk.

    Each run gets a unique folder::

        processed/
          <song_name>_<4hex>_<method_tag>/
            stems/
    """
    rid = run_id()
    parts = [sanitize_filename(song_name), rid]
    if method_tag:
        parts.append(method_tag)
    folder_name = "_".join(parts)

    song_dir = Path(output_base) / folder_name
    stems_dir = song_dir / "stems"
    ensure_dir(stems_dir)
    return song_dir, stems_dir
