"""Pitch-shift audio stems to a target musical key.

Uses pyrubberband for high-quality formant-preserving pitch shifting,
which handles vocals significantly better than a phase vocoder.

Requires the ``rubberband`` CLI tool (macOS: ``brew install rubberband``).
"""

import time

import soundfile as sf
import pyrubberband as pyrb

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_PC_INDEX = {name: i for i, name in enumerate(PITCH_CLASSES)}

# Common enharmonic aliases → canonical sharp-based name
_ENHARMONIC: dict[str, str] = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B",
}


def _normalize_pitch(name: str) -> str:
    """Map a pitch name like 'Bb' or 'Db' to its sharp-based canonical form."""
    name = name.strip()
    return _ENHARMONIC.get(name, name)


def parse_key(key_str: str) -> tuple[int, str]:
    """Parse a key string like 'A minor' into (pitch_class_index, mode).

    Returns (pc_index, mode) where mode is 'major' or 'minor'.
    """
    parts = key_str.strip().split()
    if len(parts) != 2 or parts[1] not in ("major", "minor"):
        raise ValueError(f"Cannot parse key: {key_str!r}  (expected 'X major' or 'X minor')")
    root = _normalize_pitch(parts[0])
    if root not in _PC_INDEX:
        raise ValueError(f"Unknown pitch class: {parts[0]!r}")
    return _PC_INDEX[root], parts[1]


def semitone_distance(source_key: str, target_key: str) -> int:
    """Shortest pitch shift (in semitones) to move *source_key* to *target_key*.

    Both keys must include mode (e.g. ``'A minor'``).  The mode is used to
    compare root notes directly — shifting A minor → C minor is +3, not the
    relative-major distance.

    The returned value is in [-6, +6] (shortest path around the pitch-class
    circle).
    """
    src_pc, _ = parse_key(source_key)
    tgt_pc, _ = parse_key(target_key)
    diff = (tgt_pc - src_pc) % 12
    if diff > 6:
        diff -= 12
    return diff


def plausible_key_confusions(key_str: str) -> list[str]:
    """Keys a detector most often mistakes for *key_str*, most likely first.

    Krumhansl-Schmuckler correlation confuses a handful of specific
    relationships far more than random keys: the relative major/minor
    (identical pitch content), the neighbouring fifths (six of seven notes
    shared), and the parallel major/minor (same root).  Offering these as
    one-click corrections means a wrong-key vote costs no typing.
    """
    pc, mode = parse_key(key_str)
    other_mode = "minor" if mode == "major" else "major"

    def name(offset: int, m: str) -> str:
        return f"{PITCH_CLASSES[(pc + offset) % 12]} {m}"

    # Relative minor is 3 semitones down from major; relative major is 3 up from minor
    relative = name(9, "minor") if mode == "major" else name(3, "major")

    candidates = [
        relative,
        name(7, mode),           # dominant (up a fifth)
        name(5, mode),           # subdominant (down a fifth)
        name(0, other_mode),     # parallel major/minor
    ]

    # De-duplicate while preserving order, and never suggest the key itself
    seen = {key_str}
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def rekey_audio(
    wav_path: str,
    semitones: int,
    output_path: str,
) -> str:
    """Pitch-shift a WAV file by *semitones* and write to *output_path*.

    Returns *output_path* for convenience.  If semitones == 0, the file is
    simply copied unchanged.
    """
    y, sr = sf.read(wav_path, dtype="float64")

    if semitones == 0:
        sf.write(output_path, y, sr, subtype="PCM_24")
        return output_path

    duration_s = len(y) / sr
    t0 = time.perf_counter()
    y_shifted = pyrb.pitch_shift(y, sr, n_steps=semitones)
    rb_time = time.perf_counter() - t0
    print(f"[PERF]     rubberband pitch_shift  {semitones:+d} semitones  "
          f"audio={duration_s:.1f}s  elapsed={rb_time:.2f}s  "
          f"({rb_time/duration_s:.2f}x realtime)", flush=True)

    sf.write(output_path, y_shifted, sr, subtype="PCM_24")
    return output_path
