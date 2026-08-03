"""FastAPI backend for the trim picker + stem stacker web app."""

import contextlib
import csv
import io
import json
import random
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends.demucs_separator import separate
from feedback import STAGE_KEY, STAGE_TRANSFORM, STAGE_TRIM, load_events, log_event
from processing.audio_analysis import CAMELOT_MAP, analyze, trim_stems_to_onset
from processing.madmom_beats import get_downbeats, rank_candidates, score_all_downbeats
from processing.rekey import plausible_key_confusions
from processing.stack_engine import (
    CATEGORIES,
    build_preview,
    mix_stems as engine_mix_stems,
    random_selection,
)
from processing.transform_limits import aggregate_limits, save_limits
from processing.trim_picker import save_trim_choice, score_pick_agreement
from utils import build_song_dirs, clean_song_name, parse_bpm_from_filename


def _random_loop_start(
    name: str,
    trim_sec: float,
    loop_bars: int,
    bpm: float,
) -> float:
    """Pick a random downbeat start for a loop slot.

    Loads downbeats from the song's cache, filters to those at or after
    *trim_sec*, and samples weighted by kick energy so dense sections get
    picked more often.  Falls back to *trim_sec* if no downbeats are found.
    """
    cache_dir = CACHE_DIR / name
    all_downbeats: list[dict] = []
    for mname in ("lalal.json", "cache.json"):
        f = cache_dir / mname
        if f.is_file():
            with contextlib.suppress(Exception):
                all_downbeats = json.loads(f.read_text()).get("all_downbeats", [])
            break

    if not all_downbeats:
        return trim_sec

    loop_duration = loop_bars * 4 * 60.0 / max(bpm, 40.0)

    # Keep downbeats that start at/after the trim point and leave room for the full loop.
    # Use the last downbeat time as a soft upper bound.
    last_db = all_downbeats[-1].get("time_sec", 0)
    valid = [
        db for db in all_downbeats
        if db.get("time_sec", 0) >= trim_sec
        and db.get("time_sec", 0) + loop_duration <= last_db + loop_duration
    ]

    if not valid:
        return trim_sec

    # Weighted random — higher kick energy = more likely to be chosen
    weights = [max(db.get("energy_pct", 1.0), 0.5) for db in valid]
    chosen = random.choices(valid, weights=weights, k=1)[0]
    return chosen["time_sec"]


def _resolve_bpm(filename: str, stored_bpm: float | None) -> float | None:
    """Return the best BPM for a song.

    Filename-derived BPM is always preferred over the madmom-detected value
    stored in cache, because the filename was curated by a human.  Falls back
    to *stored_bpm* when no BPM can be parsed from the filename.
    """
    return parse_bpm_from_filename(filename) or stored_bpm

@contextmanager
def _timer(label: str):
    """Log elapsed time for a labelled block to stdout."""
    t0 = time.perf_counter()
    print(f"[PERF] ▶ {label}", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        print(f"[PERF] ✓ {label}  {elapsed:.2f}s", flush=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
PROCESSED_DIR = PROJECT_ROOT / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"
STEM_LIBRARY_FILE = DATA_DIR / "stem_library.json"
STACKS_DIR = PROJECT_ROOT / "stacks"
STACK_WORK_DIR = PROJECT_ROOT / ".stack_work"

app = FastAPI(title="musicbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(filename: str) -> str:
    return Path(filename).stem.strip()


def _try_load_cache(filename: str) -> dict | None:
    """Load cached data for a song, preferring Lalal over Demucs.

    Checks lalal.json first, falls back to cache.json (Demucs).
    """
    key = _cache_key(filename)
    song_dir = CACHE_DIR / key

    for manifest_name, backend in [("lalal.json", "lalal"), ("cache.json", "demucs")]:
        manifest = song_dir / manifest_name
        if not manifest.is_file():
            continue
        try:
            with open(manifest) as f:
                data = json.load(f)
            data.setdefault("backend", backend)
            return data
        except Exception:
            continue
    return None


def _save_cache(filename: str, data: dict) -> None:
    """Write a cache.json manifest so future loads are instant."""
    key = _cache_key(filename)
    cache_path = CACHE_DIR / key
    cache_path.mkdir(parents=True, exist_ok=True)
    with open(cache_path / "cache.json", "w") as f:
        json.dump(data, f, indent=2)


def _job_from_cache(job_id: str, filename: str, cached: dict) -> dict:
    """Build a ready job dict from cached data."""
    return {
        "id": job_id,
        "filename": filename,
        "song_path": cached.get("song_path", ""),
        "song_dir": "",
        "stems_dir": Path(cached["stems_dir"]),
        "backend": cached.get("backend", "demucs"),
        "status": "ready",
        "candidates": cached["candidates"],
        "all_downbeats": cached.get("all_downbeats", []),
        "bpm": cached["bpm"],
        "analysis": cached["analysis"],
        "trim_sec": None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Stem library helpers
# ---------------------------------------------------------------------------

def _load_stem_library() -> dict:
    if STEM_LIBRARY_FILE.is_file():
        with open(STEM_LIBRARY_FILE) as f:
            return json.load(f)
    return {}


def _save_stem_library(lib: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STEM_LIBRARY_FILE, "w") as f:
        json.dump(lib, f, indent=2)


def _add_to_stem_library(job: dict, trim_sec: float) -> None:
    """Add a trimmed song to the stem library index.

    Supports dual backends: each song entry has a ``backends`` dict mapping
    backend name -> {stems_dir, stems[]}.  Legacy entries with a flat
    ``stems_dir`` are still readable by the stack engine.
    """
    lib = _load_stem_library()
    key = Path(job["filename"]).stem
    backend = job.get("backend", "demucs")
    stems_dir = str(job["stems_dir"])

    stem_files = sorted(
        f.stem for f in Path(stems_dir).glob("*.wav")
    )

    existing = lib.get(key, {})
    existing_backends = existing.get("backends", {})

    if not existing_backends and existing.get("stems_dir"):
        old_dir = existing["stems_dir"]
        old_stems = sorted(f.stem for f in Path(old_dir).glob("*.wav")) if Path(old_dir).is_dir() else []
        existing_backends["demucs"] = {"stems_dir": old_dir, "stems": old_stems}

    existing_backends[backend] = {"stems_dir": stems_dir, "stems": stem_files}

    lib[key] = {
        "filename": job["filename"],
        "trim_sec": round(trim_sec, 4),
        "bpm": job.get("bpm"),
        "analysis": job.get("analysis", {}),
        "stems_dir": stems_dir,
        "backends": existing_backends,
    }
    _save_stem_library(lib)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PickRequest(BaseModel):
    job_id: str
    trim_sec: float


class KeyFeedbackRequest(BaseModel):
    filename: str
    detected_key: str
    detected_camelot: str = ""
    verdict: str                        # "correct" | "wrong"
    corrected_key: str | None = None    # required when verdict == "wrong"


class TransformFeedbackRequest(BaseModel):
    category: str                       # drums | bass | vocals | …
    verdict: str                        # "good" | "bad"
    semitones: float | None = None
    stretch_ratio: float | None = None
    song: str | None = None
    stack_id: str | None = None


class StackRequest(BaseModel):
    slots: dict[str, str]                          # category -> stem library entry name
    target_bpm: float
    target_key: str
    slot_backends: dict[str, str] | None = None    # category -> "lalal" | "demucs"
    slot_bpms: dict[str, float] | None = None      # category -> source BPM override
    slot_loops: dict[str, int] | None = None       # category -> loop length in bars (4/8/16/32)


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def _process_song(job_id: str) -> None:
    """Run Demucs + madmom in background, update job state, cache result."""
    job = jobs[job_id]
    song_path = job["song_path"]
    stems_dir = job["stems_dir"]
    fname = job["filename"]

    t_total = time.perf_counter()
    print(f"\n[PERF] ═══ Processing: {fname} ═══", flush=True)

    try:
        job["status"] = "separating"
        with _timer("Demucs stem separation"):
            separate(song_path, str(stems_dir))

        job["status"] = "detecting"
        with _timer("madmom downbeat detection"):
            downbeats, madmom_bpm = get_downbeats(song_path)

        # Prefer BPM from filename (more accurate than madmom estimate)
        title_bpm = parse_bpm_from_filename(fname)
        if title_bpm:
            bpm = title_bpm
            print(f"[BPM] filename={title_bpm}  madmom={madmom_bpm}  → using filename", flush=True)
        else:
            bpm = madmom_bpm
            print(f"[BPM] madmom={madmom_bpm}  (no filename BPM found)", flush=True)

        drums_path = str(stems_dir / "drums.wav")
        with _timer("rank candidates (kick energy)"):
            candidates = rank_candidates(downbeats, drums_path, bpm=bpm)

        with _timer("score all downbeats"):
            all_db = score_all_downbeats(downbeats, drums_path, bpm=bpm)

        with _timer("audio analysis (BPM/key/librosa)"):
            analysis = analyze(song_path)

        cand_dicts = [
            {"time_sec": c.time_sec, "energy_pct": round(c.energy_pct, 1)}
            for c in candidates
        ]

        job["bpm"] = bpm
        job["analysis"] = analysis
        job["candidates"] = cand_dicts
        job["all_downbeats"] = all_db
        job["status"] = "ready"

        _save_cache(job["filename"], {
            "filename": job["filename"],
            "song_path": song_path,
            "stems_dir": str(stems_dir),
            "bpm": bpm,
            "analysis": {"bpm": analysis["bpm"], "key": analysis["key"], "camelot": analysis["camelot"]},
            "candidates": cand_dicts,
            "all_downbeats": all_db,
        })

        total = time.perf_counter() - t_total
        print(f"[PERF] ═══ Done: {fname}  total={total:.1f}s ═══\n", flush=True)

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[PERF] ✗ {fname} failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_song(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")

    job_id = uuid.uuid4().hex[:8]

    cached = _try_load_cache(file.filename)
    if cached and Path(cached["stems_dir"]).is_dir():
        jobs[job_id] = _job_from_cache(job_id, file.filename, cached)
        return {"job_id": job_id, "status": "ready"}

    upload_path = UPLOAD_DIR / job_id
    upload_path.mkdir(parents=True, exist_ok=True)

    song_file = upload_path / file.filename
    with open(song_file, "wb") as f:
        shutil.copyfileobj(file.file, f)

    song_dir, stems_dir = build_song_dirs(str(PROCESSED_DIR), song_file.stem, method_tag="madmom")

    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "song_path": str(song_file),
        "song_dir": str(song_dir),
        "stems_dir": stems_dir,
        "status": "queued",
        "candidates": [],
        "all_downbeats": [],
        "bpm": None,
        "analysis": None,
        "trim_sec": None,
        "error": None,
    }

    thread = threading.Thread(target=_process_song, args=(job_id,), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/library")
async def list_library():
    """Return all pre-processed songs in the cache, ready for instant loading."""
    if not CACHE_DIR.is_dir():
        return {"songs": []}
    songs = []
    for d in sorted(CACHE_DIR.iterdir()):
        if not d.is_dir():
            continue
        backends = []
        data = None
        for mname, bname in [("lalal.json", "lalal"), ("cache.json", "demucs")]:
            m = d / mname
            if m.is_file():
                try:
                    with open(m) as f:
                        mdata = json.load(f)
                    if Path(mdata["stems_dir"]).is_dir():
                        backends.append(bname)
                        if data is None:
                            data = mdata
                except Exception:
                    continue
        if not data:
            continue
        fname = data["filename"]
        songs.append({
            "cache_key": d.name,
            "filename": fname,
            "bpm": _resolve_bpm(fname, data.get("bpm")),
            "key": data.get("analysis", {}).get("key", ""),
            "camelot": data.get("analysis", {}).get("camelot", ""),
            "backends": backends,
        })
    return {"songs": songs}


@app.post("/api/library/load/{cache_key}")
async def load_from_library(cache_key: str):
    """Instantly load a pre-processed song from cache into a job.

    Prefers lalal.json over cache.json (Demucs).
    """
    song_dir = CACHE_DIR / cache_key
    cached = None
    for mname, bname in [("lalal.json", "lalal"), ("cache.json", "demucs")]:
        m = song_dir / mname
        if m.is_file():
            try:
                with open(m) as f:
                    data = json.load(f)
                if Path(data["stems_dir"]).is_dir():
                    data.setdefault("backend", bname)
                    cached = data
                    break
            except Exception:
                continue
    if not cached:
        raise HTTPException(404, "Not in cache")

    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = _job_from_cache(job_id, cached["filename"], cached)
    return {"job_id": job_id, "status": "ready"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "error": job["error"],
        "bpm": job["bpm"],
    }


@app.get("/api/candidates/{job_id}")
async def get_candidates(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "ready":
        raise HTTPException(409, f"Job not ready: {job['status']}")
    detected_key = job["analysis"]["key"]
    try:
        alternatives = plausible_key_confusions(detected_key)
    except ValueError:
        alternatives = []

    return {
        "candidates": job["candidates"],
        "all_downbeats": job.get("all_downbeats", []),
        "bpm": job["bpm"],
        "analysis": {
            "bpm": job["analysis"]["bpm"],
            "key": detected_key,
            "camelot": job["analysis"]["camelot"],
            "key_alternatives": alternatives,
        },
    }


@app.get("/api/audio/{job_id}/drums")
async def get_drums_audio(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    drums_path = job["stems_dir"] / "drums.wav"
    if not drums_path.is_file():
        raise HTTPException(404, "Drums stem not found")

    return FileResponse(str(drums_path), media_type="audio/wav")


@app.post("/api/pick")
async def pick_trim(req: PickRequest):
    job = jobs.get(req.job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "ready":
        raise HTTPException(409, f"Job not ready: {job['status']}")

    stems_dir = job["stems_dir"]
    trim_sec = req.trim_sec
    fname = job["filename"]

    print(f"\n[PERF] ═══ Trim: {fname}  at {trim_sec:.3f}s ═══", flush=True)
    t_total = time.perf_counter()

    # Copy stems to an output dir so the cache originals stay intact
    out_dir = PROCESSED_DIR / f"{req.job_id}_trimmed"
    out_dir.mkdir(parents=True, exist_ok=True)
    with _timer("copy stem WAVs"):
        for wav in Path(stems_dir).glob("*.wav"):
            shutil.copy2(str(wav), str(out_dir / wav.name))

    if trim_sec > 0:
        with _timer("trim_stems_to_onset"):
            trim_stems_to_onset(str(out_dir), trim_sec)

    with _timer("save trim + library index"):
        save_trim_choice(job["filename"], trim_sec, method="web")
        _add_to_stem_library(job, trim_sec)

    # Every commit is an implicit vote on the candidate ranking: accepting the
    # top-ranked pick endorses it, overriding it supplies the correct answer.
    agreement = score_pick_agreement(job.get("candidates", []), trim_sec)
    log_event(
        STAGE_TRIM,
        filename=fname,
        song=clean_song_name(fname),
        chosen_sec=round(trim_sec, 4),
        bpm=job.get("bpm"),
        **agreement,
    )
    if agreement["auto_pick_sec"] is not None:
        verdict = "agreed" if agreement["agreed"] else "OVERRIDDEN"
        print(f"[FEEDBACK] trim {verdict}: auto={agreement['auto_pick_sec']:.3f}s "
              f"human={trim_sec:.3f}s  ({agreement['auto_pick_delta_ms']:+.0f}ms, "
              f"rank={agreement['chosen_rank']})", flush=True)

    print(f"[PERF] ═══ Trim done  total={time.perf_counter()-t_total:.2f}s ═══\n", flush=True)

    job["trim_sec"] = trim_sec
    job["trimmed_dir"] = out_dir
    job["status"] = "trimmed"

    return {
        "status": "trimmed",
        "trim_sec": trim_sec,
        "download_url": f"/api/download/{req.job_id}",
    }


# ---------------------------------------------------------------------------
# Feedback — your ear, wired back into the benchmarks
# ---------------------------------------------------------------------------

KEY_LABELS_FILE = DATA_DIR / "key_labels.csv"
KEY_LABEL_FIELDS = ["Song", "Key", "Camelot", "Source"]


def _append_key_label(song: str, key: str, camelot: str, source: str) -> None:
    """Upsert a confirmed/corrected key into the ground-truth CSV."""
    rows: dict[str, dict] = {}
    if KEY_LABELS_FILE.is_file():
        with open(KEY_LABELS_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["Song"].strip()] = {
                    field: (row.get(field) or "").strip() for field in KEY_LABEL_FIELDS
                }

    rows[song] = {"Song": song, "Key": key, "Camelot": camelot, "Source": source}

    KEY_LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(KEY_LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_LABEL_FIELDS)
        writer.writeheader()
        for name in sorted(rows):
            writer.writerow(rows[name])


@app.post("/api/feedback/key")
async def feedback_key(req: KeyFeedbackRequest):
    """Record a thumbs-up/down on detected key and grow the key ground truth."""
    if req.verdict not in ("correct", "wrong"):
        raise HTTPException(400, "verdict must be 'correct' or 'wrong'")
    if req.verdict == "wrong" and not req.corrected_key:
        raise HTTPException(400, "corrected_key is required when verdict is 'wrong'")

    song = clean_song_name(req.filename)
    truth = req.detected_key if req.verdict == "correct" else req.corrected_key

    camelot = req.detected_camelot
    if req.verdict == "wrong":
        camelot = CAMELOT_MAP.get(truth, "")

    _append_key_label(song, truth, camelot, source="web")
    log_event(
        STAGE_KEY,
        filename=req.filename,
        song=song,
        detected_key=req.detected_key,
        verdict=req.verdict,
        corrected_key=req.corrected_key,
    )
    print(f"[FEEDBACK] key {req.verdict}: {song} detected={req.detected_key} "
          f"truth={truth}", flush=True)

    return {"status": "recorded", "song": song, "key": truth, "camelot": camelot}


@app.post("/api/feedback/transform")
async def feedback_transform(req: TransformFeedbackRequest):
    """Record a thumbs-up/down on a rekeyed/stretched stem and relearn limits."""
    if req.verdict not in ("good", "bad"):
        raise HTTPException(400, "verdict must be 'good' or 'bad'")

    log_event(
        STAGE_TRANSFORM,
        category=req.category,
        verdict=req.verdict,
        semitones=req.semitones,
        stretch_ratio=req.stretch_ratio,
        song=req.song,
        stack_id=req.stack_id,
    )

    limits = aggregate_limits(load_events(STAGE_TRANSFORM))
    save_limits(limits)

    print(f"[FEEDBACK] transform {req.verdict}: {req.category} "
          f"{req.semitones:+g} semitones @ {req.stretch_ratio}x"
          if req.semitones is not None else
          f"[FEEDBACK] transform {req.verdict}: {req.category}", flush=True)

    return {"status": "recorded", "limits": limits.get(req.category, {})}


@app.get("/api/feedback/summary")
async def feedback_summary():
    """Counts per stage plus the current learned transform limits."""
    events = load_events()
    counts: dict[str, int] = {}
    for event in events:
        stage = event.get("stage", "unknown")
        counts[stage] = counts.get(stage, 0) + 1

    trim_events = [e for e in events if e.get("stage") == STAGE_TRIM]
    agreed = sum(1 for e in trim_events if e.get("agreed"))

    return {
        "counts": counts,
        "trim_agreement": {"agreed": agreed, "total": len(trim_events)},
        "transform_limits": aggregate_limits(
            [e for e in events if e.get("stage") == STAGE_TRANSFORM]
        ),
    }


@app.get("/api/download/{job_id}")
async def download_stems(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    stems_dir = job.get("trimmed_dir") or job["stems_dir"]
    stem_files = list(Path(stems_dir).glob("*.wav"))
    if not stem_files:
        raise HTTPException(404, "No stems found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in stem_files:
            zf.write(f, f.name)
    buf.seek(0)

    filename = Path(job["filename"]).stem + "_stems.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/stem-library/{name}")
async def delete_stem_library_entry(name: str):
    """Remove a song from the stem library index (does not delete audio files)."""
    lib = _load_stem_library()
    if name not in lib:
        raise HTTPException(404, f"'{name}' not found in stem library")
    del lib[name]
    _save_stem_library(lib)
    return {"status": "deleted", "name": name}


@app.get("/api/stem-library")
async def get_stem_library():
    """Return all trimmed songs in the stem library."""
    lib = _load_stem_library()
    entries = []
    for key, entry in sorted(lib.items()):
        analysis = entry.get("analysis", {})
        backends_info = entry.get("backends", {})
        if not backends_info and entry.get("stems_dir"):
            sd = Path(entry["stems_dir"])
            stems = sorted(f.stem for f in sd.glob("*.wav")) if sd.is_dir() else list(CATEGORIES)
            backends_info = {"demucs": {"stems_dir": str(sd), "stems": stems}}
        efname = entry.get("filename", "")
        entries.append({
            "name": key,
            "filename": efname,
            "trim_sec": entry.get("trim_sec", 0),
            "bpm": _resolve_bpm(efname, entry.get("bpm") or analysis.get("bpm")),
            "key": analysis.get("key", ""),
            "camelot": analysis.get("camelot", ""),
            "backends": backends_info,
        })
    return {"entries": entries, "count": len(entries)}


# ---------------------------------------------------------------------------
# Stack endpoints
# ---------------------------------------------------------------------------

stack_jobs: dict[str, dict] = {}


def _resolve_loop_starts(req: "StackRequest", lib: dict) -> dict[str, float]:
    """For each slot that has a loop length set, pick a random start time.

    Returns a dict mapping category -> loop_start_sec.
    """
    if not req.slot_loops:
        return {}
    starts: dict[str, float] = {}
    for cat, loop_bars in req.slot_loops.items():
        if loop_bars is None:
            continue
        name = req.slots.get(cat)
        if not name or name not in lib:
            continue
        entry = lib[name]
        trim_sec = entry.get("trim_sec", 0)
        analysis = entry.get("analysis", {})
        bpm = (req.slot_bpms or {}).get(cat) \
            or _resolve_bpm(entry.get("filename", ""), entry.get("bpm") or analysis.get("bpm", 120.0)) \
            or 120.0
        starts[cat] = _random_loop_start(name, trim_sec, loop_bars, bpm)
    return starts


@app.post("/api/stack/shuffle")
async def stack_shuffle():
    """Return a random selection from the stem library (one per slot)."""
    lib = _load_stem_library()
    if not lib:
        raise HTTPException(400, "Stem library is empty — trim some songs first")
    selection = random_selection(lib)
    entries = {}
    for cat, name in selection.items():
        entry = lib[name]
        analysis = entry.get("analysis", {})
        sfname = entry.get("filename", "")
        entries[cat] = {
            "name": name,
            "bpm": _resolve_bpm(sfname, entry.get("bpm") or analysis.get("bpm")),
            "key": analysis.get("key", ""),
            "camelot": analysis.get("camelot", ""),
        }
    return {"slots": entries}


@app.post("/api/stack/preview")
async def stack_preview(req: StackRequest):
    """Build a preview mix from the selected slots. Runs synchronously."""
    lib = _load_stem_library()
    if not lib:
        raise HTTPException(400, "Stem library is empty")

    for _cat, name in req.slots.items():
        if name and name not in lib:
            raise HTTPException(404, f"'{name}' not in stem library")

    stack_id = uuid.uuid4().hex[:8]
    work = STACK_WORK_DIR / stack_id
    work.mkdir(parents=True, exist_ok=True)

    stack_jobs[stack_id] = {"status": "building"}

    # Resolve random loop start positions for any looped slots
    slot_loop_starts = _resolve_loop_starts(req, lib)

    try:
        result = build_preview(
            req.slots, lib,
            req.target_bpm, req.target_key,
            str(work),
            slot_backends=req.slot_backends,
            slot_bpms=req.slot_bpms,
            slot_loops=req.slot_loops,
            slot_loop_starts=slot_loop_starts,
            is_preview=True,
        )
    except Exception as e:
        stack_jobs[stack_id] = {"status": "error", "error": str(e)}
        raise HTTPException(500, f"Stack build failed: {e}") from e

    stack_jobs[stack_id] = {
        "status": "ready",
        "work_dir": str(work),
        "mix_path": result["mix_path"],
        "stem_paths": result["stem_paths"],
        "slots_info": result["slots_info"],
    }

    stem_urls = {}
    for cat, _path in result["stem_paths"].items():
        stem_urls[cat] = f"/api/stack/audio/{stack_id}/{cat}"

    return {
        "stack_id": stack_id,
        "mix_url": f"/api/stack/audio/{stack_id}/mix",
        "stem_urls": stem_urls,
        "slots_info": result["slots_info"],
    }


@app.get("/api/stack/audio/{stack_id}/{stem}")
async def stack_audio(stack_id: str, stem: str):
    """Serve a prepared stem or the mix for a stack preview."""
    sj = stack_jobs.get(stack_id)
    if not sj or sj["status"] != "ready":
        raise HTTPException(404, "Stack not found or not ready")

    path = sj["mix_path"] if stem == "mix" else sj.get("stem_paths", {}).get(stem)

    if not path or not Path(path).is_file():
        raise HTTPException(404, f"Audio not found: {stem}")

    return FileResponse(str(path), media_type="audio/wav")


@app.post("/api/stack/export")
async def stack_export(req: StackRequest):
    """Build the stack and return a zip with individual stems + mix + manifest."""
    lib = _load_stem_library()
    if not lib:
        raise HTTPException(400, "Stem library is empty")

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    export_dir = STACKS_DIR / ts
    export_dir.mkdir(parents=True, exist_ok=True)

    slot_loop_starts = _resolve_loop_starts(req, lib)

    with tempfile.TemporaryDirectory(prefix="musicbot_export_") as work:
        result = build_preview(
            req.slots, lib,
            req.target_bpm, req.target_key,
            work,
            slot_backends=req.slot_backends,
            slot_bpms=req.slot_bpms,
            slot_loops=req.slot_loops,
            slot_loop_starts=slot_loop_starts,
            is_preview=False,
        )

        exported: list[str] = []
        manifest_stems: dict[str, dict] = {}

        for cat, src_path in result["stem_paths"].items():
            dst = str(export_dir / f"{cat}.wav")
            shutil.copy2(src_path, dst)
            exported.append(dst)
            info = result["slots_info"].get(cat, {})
            manifest_stems[cat] = {
                "source_song": info.get("name", ""),
                "original_bpm": info.get("original_bpm"),
                "original_key": info.get("original_key", ""),
            }

        if exported:
            mix_dst = str(export_dir / "stack.wav")
            engine_mix_stems(exported, mix_dst)

        manifest = {
            "target_bpm": req.target_bpm,
            "target_key": req.target_key,
            "stems": manifest_stems,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with open(export_dir / "stack.json", "w") as f:
            json.dump(manifest, f, indent=2)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in export_dir.iterdir():
            zf.write(fp, fp.name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="stack_{ts}.zip"'},
    )


# ---------------------------------------------------------------------------
# Serve frontend build in production
# ---------------------------------------------------------------------------

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("\n  musicbot trim picker → http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
