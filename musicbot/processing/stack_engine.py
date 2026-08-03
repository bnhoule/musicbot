"""Stem stacking engine — trim, rekey, tempo-match, and mix stems.

Shared by the web backend and the CLI stacker.  All functions are
stateless and operate on file paths, making them easy to test and
compose.
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from processing.rekey import semitone_distance, rekey_audio
from processing.tempo_match import stretch_audio
from processing.transform_limits import check_transform

# Preview mode: cap audio fed to rubberband at this many seconds.
# Reduces rubberband time by 50-75% for typical 3-7 min stems.
PREVIEW_MAX_SEC = 90


@contextmanager
def _timer(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[PERF]   {label}  {time.perf_counter()-t0:.2f}s", flush=True)

CATEGORIES = ("drums", "bass", "vocals", "piano", "synth", "other")
HEADROOM_DB = -1.0


def load_stem_library(lib_path: str | Path) -> dict:
    """Load the stem library index from disk."""
    p = Path(lib_path)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return {}


def prepare_stem(
    stem_wav: str,
    trim_sec: float,
    source_bpm: float,
    source_key: str,
    target_bpm: float,
    target_key: str,
    work_dir: str,
    tag: str = "",
    max_duration_sec: float | None = None,
    loop_bars: int | None = None,
) -> str:
    """Trim + rekey + tempo-stretch a single stem, return output path.

    Parameters
    ----------
    max_duration_sec : cap on audio length applied before rubberband (preview).
    loop_bars        : if set, cap audio to exactly this many bars at source_bpm.
                       Takes priority over max_duration_sec.
    tag              : optional prefix for the output filename.
    """
    basename = tag or Path(stem_wav).stem
    out_path = os.path.join(work_dir, f"{basename}.wav")

    t0 = time.perf_counter()

    # Compute the effective duration cap
    if loop_bars is not None:
        # bars × 4 beats × (60s / bpm)  — exact loop length at source tempo
        loop_duration = loop_bars * 4 * 60.0 / max(source_bpm, 40.0)
        effective_max = loop_duration
        loop_label = f"{loop_bars}-bar loop  ({loop_duration:.2f}s)"
    else:
        effective_max = max_duration_sec
        loop_label = "full"

    print(f"[PERF] ▶ prepare_stem: {Path(stem_wav).name}  "
          f"trim={trim_sec:.3f}s  {source_bpm:.3f}→{target_bpm:.3f} BPM  "
          f"{source_key}→{target_key}  [{loop_label}]", flush=True)

    with _timer("  sf.read"):
        y, sr = sf.read(stem_wav, dtype="float64")

    if trim_sec > 0:
        start_sample = int(trim_sec * sr)
        if start_sample < len(y):
            y = y[start_sample:]

    if effective_max is not None:
        max_samples = int(effective_max * sr)
        if len(y) > max_samples:
            y = y[:max_samples]

    trimmed_path = os.path.join(work_dir, f"{basename}_trimmed.wav")
    with _timer("  sf.write trimmed"):
        sf.write(trimmed_path, y, sr, subtype="PCM_24")

    semitones = semitone_distance(source_key, target_key)
    if semitones != 0:
        rekeyed_path = os.path.join(work_dir, f"{basename}_rekeyed.wav")
        with _timer(f"  rekey_audio ({semitones:+d} semitones)"):
            rekey_audio(trimmed_path, semitones, rekeyed_path)
        src = rekeyed_path
    else:
        src = trimmed_path

    with _timer("  stretch_audio"):
        stretch_audio(src, source_bpm, target_bpm, out_path)

    print(f"[PERF] ✓ prepare_stem: {Path(stem_wav).name}  total={time.perf_counter()-t0:.2f}s", flush=True)
    return out_path


def mix_stems(stem_paths: list[str], output_path: str, _label: str = "") -> None:
    """Sum-and-normalize a list of WAV files to *output_path*.

    Peak-normalizes to HEADROOM_DB so the mix doesn't clip.
    """
    t0 = time.perf_counter()
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
    print(f"[PERF]   mix_stems ({len(stem_paths)} stems)  {time.perf_counter()-t0:.2f}s", flush=True)


def _resolve_stems_dir(entry: dict, backend: str | None = None) -> str:
    """Return the stems directory for an entry, respecting backend choice.

    Supports both legacy (flat ``stems_dir``) and new (``backends`` dict).
    """
    backends = entry.get("backends", {})
    if backends:
        if backend and backend in backends:
            return backends[backend]["stems_dir"]
        for pref in ("lalal", "demucs"):
            if pref in backends:
                return backends[pref]["stems_dir"]
    return entry.get("stems_dir", "")


def build_preview(
    slot_selections: dict[str, str],
    stem_library: dict,
    target_bpm: float,
    target_key: str,
    work_dir: str,
    slot_backends: dict[str, str] | None = None,
    slot_bpms: dict[str, float] | None = None,
    slot_loops: dict[str, int] | None = None,
    slot_loop_starts: dict[str, float] | None = None,
    is_preview: bool = True,
) -> dict:
    """Build a full stack preview or export.

    Parameters
    ----------
    slot_selections  : dict mapping category -> stem library entry name
    stem_library     : the full stem library dict
    target_bpm       : target BPM for all stems
    target_key       : target key for all stems (e.g. "A minor")
    work_dir         : scratch directory for intermediate files
    slot_backends    : optional dict mapping category -> preferred backend name
    slot_loops       : optional dict mapping category -> loop length in bars
                       (4 / 8 / 16 / 32).  None or missing key = full track.
    slot_loop_starts : optional dict mapping category -> absolute loop start
                       time in seconds.  Overrides the entry's trim_sec when
                       loop mode is active for that slot.
    is_preview       : if True, cap non-looped audio at PREVIEW_MAX_SEC and
                       run stems in parallel.  Set False for a full export.

    Returns
    -------
    dict with keys: mix_path, stem_paths (dict cat->path), slots_info (dict cat->metadata)
    """
    t_stack = time.perf_counter()
    mode = "preview" if is_preview else "export"
    print(f"\n[PERF] ═══ build_preview  target={target_bpm:.1f} BPM  {target_key}  [{mode}] ═══", flush=True)

    # Collect per-slot args first so we can dispatch in parallel
    tasks: list[tuple[str, str, dict, dict]] = []  # (cat, stem_wav, prepare_kwargs, info)
    for cat in CATEGORIES:
        name = slot_selections.get(cat)
        if not name or name not in stem_library:
            continue

        entry = stem_library[name]
        backend = (slot_backends or {}).get(cat)
        stems_dir = _resolve_stems_dir(entry, backend)
        stem_wav = os.path.join(stems_dir, f"{cat}.wav")

        if not os.path.isfile(stem_wav):
            continue

        analysis = entry.get("analysis", {})
        detected_bpm = entry.get("bpm") or analysis.get("bpm", 120.0)
        source_bpm = (slot_bpms or {}).get(cat) or detected_bpm
        source_key = analysis.get("key", "C major")

        # Loop start overrides the stored trim_sec when loop mode is active
        loop_bars = (slot_loops or {}).get(cat)
        loop_start = (slot_loop_starts or {}).get(cat)
        trim_sec = loop_start if (loop_bars is not None and loop_start is not None) \
            else entry.get("trim_sec", 0)

        # For full-track (no loop) in preview mode, cap at PREVIEW_MAX_SEC
        max_dur = (PREVIEW_MAX_SEC if is_preview else None) if loop_bars is None else None

        tag = f"{name}_{cat}"

        prepare_kwargs = {
            "stem_wav": stem_wav,
            "trim_sec": trim_sec,
            "source_bpm": source_bpm,
            "source_key": source_key,
            "target_bpm": target_bpm,
            "target_key": target_key,
            "work_dir": work_dir,
            "tag": tag,
            "max_duration_sec": max_dur,
            "loop_bars": loop_bars,
        }
        semitones = semitone_distance(source_key, target_key)
        stretch_ratio = target_bpm / source_bpm if source_bpm else 1.0

        slot_meta = {
            "name": name,
            "backend": backend or "auto",
            "original_bpm": source_bpm,
            "detected_bpm": detected_bpm,
            "original_key": source_key,
            "trim_sec": trim_sec,
            "loop_bars": loop_bars,
            "loop_start_sec": trim_sec if loop_bars is not None else None,
            "semitones": semitones,
            "stretch_ratio": round(stretch_ratio, 4),
            "warning": check_transform(cat, semitones, stretch_ratio),
        }
        tasks.append((cat, stem_wav, prepare_kwargs, slot_meta))

    stem_paths: dict[str, str] = {}
    slots_info: dict[str, dict] = {}

    if is_preview and len(tasks) > 1:
        # Run all stem preparations concurrently — they're fully independent
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {
                pool.submit(prepare_stem, **kw): (cat, meta)
                for cat, _, kw, meta in tasks
            }
            for future in as_completed(futures):
                cat, meta = futures[future]
                try:
                    stem_paths[cat] = future.result()
                    slots_info[cat] = meta
                except Exception as exc:
                    print(f"[PERF] ✗ prepare_stem {cat}: {exc}", flush=True)
    else:
        # Sequential for export (avoids interleaved log output, lets rubberband use all cores)
        for cat, _, kw, meta in tasks:
            try:
                stem_paths[cat] = prepare_stem(**kw)
                slots_info[cat] = meta
            except Exception as exc:
                print(f"[PERF] ✗ prepare_stem {cat}: {exc}", flush=True)

    mix_path = ""
    if stem_paths:
        mix_path = os.path.join(work_dir, "mix.wav")
        mix_stems(list(stem_paths.values()), mix_path)

    print(f"[PERF] ═══ build_preview done  total={time.perf_counter()-t_stack:.1f}s ═══\n", flush=True)

    return {
        "mix_path": mix_path,
        "stem_paths": stem_paths,
        "slots_info": slots_info,
    }


def random_selection(
    stem_library: dict,
    exclude: dict[str, str] | None = None,
) -> dict[str, str]:
    """Pick a random stem library entry for each category.

    Returns dict mapping category -> entry name.
    """
    names = list(stem_library.keys())
    if not names:
        return {}

    selection: dict[str, str] = {}
    for cat in CATEGORIES:
        pool = [n for n in names if n != (exclude or {}).get(cat)]
        if not pool:
            pool = names
        selection[cat] = random.choice(pool)
    return selection
