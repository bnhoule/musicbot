"""Lalal.ai API client (v1.1.0).

Endpoint reference
------------------
POST /api/v1/upload/                       – Upload binary audio; returns ``source_id``.
POST /api/v1/split/stem_separator/         – Start a single split task; returns ``task_id``.
POST /api/v1/split/batch/stem_separator/   – Start multiple split tasks in one call.
POST /api/v1/check/                        – Poll one or more ``task_id``s.
POST /api/v1/delete/                       – Delete source file from storage.

Authentication
--------------
Pass the license key in the ``X-License-Key`` header.

Workflow
--------
1. upload()              → source_id
2. split() or batch_split() → task_id(s)
3. poll_all_until_done() → waits for every task, returns results keyed by task_id
4. download_file()       → streams each stem URL to disk

The high-level helper ``process_and_download_stems()`` runs all four steps.

Check-response structure (success)
-----------------------------------
{
  "result": {
    "<task_id>": {
      "status": "success",
      "source_id": "...",
      "result": {
        "duration": 180,
        "tracks": [
          {"label": "vocals",    "type": "stem", "url": "https://..."},
          {"label": "no_vocals", "type": "back", "url": "https://..."}
        ]
      }
    }
  }
}

Track types:  ``stem`` = isolated stem,  ``back`` = everything else.

Stem names accepted by the API
-------------------------------
vocals, drum, bass, piano, electric_guitar, acoustic_guitar,
synthesizer (phoenix only), strings (phoenix only), wind (phoenix only).

Splitter models: andromeda, perseus, orion, phoenix, lyra, lynx, auto.
``auto`` selects the best model per stem type.
"""

import tempfile
import time
import requests
import soundfile as sf
import librosa
from pathlib import Path

LALAL_BASE_URL = "https://www.lalal.ai"

# (lalal_stem_name, output_filename) — processed in this order
DEFAULT_STEMS: list[tuple[str, str]] = [
    ("vocals", "vocals.wav"),
    ("drum",   "drums.wav"),
    ("bass",   "bass.wav"),
]

# 5-stem preset: drums, bass, vocals, piano, synth.
# Splitter is per-stem: "auto" for most, "phoenix" required for synthesizer.
LALAL_STEMS_5: list[tuple[str, str, str]] = [
    # (lalal_api_name, output_filename, splitter)
    ("vocals",      "vocals.wav", "auto"),
    ("drum",        "drums.wav",  "auto"),
    ("bass",        "bass.wav",   "auto"),
    ("piano",       "piano.wav",  "auto"),
    ("synthesizer", "synth.wav",  "phoenix"),
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LalalClient:
    def __init__(self, api_key: str) -> None:
        self.session = requests.Session()
        self.session.headers.update({"X-License-Key": api_key})

    # ------------------------------------------------------------------
    # Low-level API calls
    # ------------------------------------------------------------------

    def upload(self, file_path: str) -> str:
        """Upload *file_path* as a binary octet-stream; return ``source_id``."""
        path = Path(file_path)
        print(f"  Uploading '{path.name}'…")
        with open(path, "rb") as fh:
            resp = self.session.post(
                f"{LALAL_BASE_URL}/api/v1/upload/",
                headers={
                    "Content-Disposition": f"attachment; filename={path.name}",
                    "Content-Type": "application/octet-stream",
                },
                data=fh,
            )
        resp.raise_for_status()
        source_id: str = resp.json()["id"]
        print(f"  Upload complete. Source ID: {source_id}")
        return source_id

    def split(
        self,
        source_id: str,
        stem: str,
        splitter: str = "auto",
        extraction_level: str = "clear_cut",
    ) -> str:
        """Queue a stem-separation task; return the ``task_id``.

        Parameters
        ----------
        extraction_level:
            ``"deep_extraction"`` – more detail, slight bleed-through.
            ``"clear_cut"`` (default) – cleaner separation, less detail.
        """
        presets: dict = {
            "stem": stem,
            "extraction_level": extraction_level,
        }
        if splitter:
            presets["splitter"] = splitter

        payload = {
            "source_id": source_id,
            "presets": presets,
        }
        resp = self.session.post(
            f"{LALAL_BASE_URL}/api/v1/split/stem_separator/",
            json=payload,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Split request failed {resp.status_code}: {resp.text}"
            )
        task_id: str = resp.json()["task_id"]
        return task_id

    def batch_split(
        self,
        source_id: str,
        stems: list[tuple[str, str, str]],
        extraction_level: str = "clear_cut",
    ) -> dict[str, tuple[str, str]]:
        """Queue multiple stem splits in a single HTTP call.

        Parameters
        ----------
        stems:
            List of ``(lalal_api_name, output_filename, splitter)`` tuples.
        extraction_level:
            Applied to all items.

        Returns
        -------
        dict mapping task_id -> (lalal_api_name, output_filename).
        """
        items = []
        for api_name, _filename, splitter in stems:
            presets: dict = {
                "stem": api_name,
                "extraction_level": extraction_level,
            }
            if splitter:
                presets["splitter"] = splitter
            items.append({"source_id": source_id, "presets": presets})

        resp = self.session.post(
            f"{LALAL_BASE_URL}/api/v1/split/batch/stem_separator/",
            json={"items": items},
        )
        if not resp.ok:
            raise RuntimeError(
                f"Batch split failed {resp.status_code}: {resp.text}"
            )

        task_map: dict[str, tuple[str, str]] = {}
        for i, result in enumerate(resp.json()["results"]):
            if result["status"] == "error":
                api_name = stems[i][0]
                raise RuntimeError(
                    f"Batch split error for '{api_name}': {result.get('error')}"
                )
            task_id = result["task_id"]
            task_map[task_id] = (stems[i][0], stems[i][1])

        return task_map

    def check(self, task_ids: list[str]) -> dict:
        """Return raw check response for the given task IDs."""
        resp = self.session.post(
            f"{LALAL_BASE_URL}/api/v1/check/",
            json={"task_ids": task_ids},
        )
        resp.raise_for_status()
        return resp.json()

    def poll_all_until_done(
        self,
        task_ids: list[str],
        poll_interval: int = 10,
        timeout: int = 3600,
    ) -> dict[str, dict]:
        """Block until every task in *task_ids* reaches a terminal state.

        Polls all pending tasks in a single request per interval (efficient).

        Returns
        -------
        dict mapping task_id → completed result entry.

        Raises
        ------
        RuntimeError  – if any task enters ``error`` or ``server_error``.
        TimeoutError  – if tasks are still running after *timeout* seconds.
        """
        pending: set[str] = set(task_ids)
        results: dict[str, dict] = {}
        elapsed = 0

        while pending and elapsed < timeout:
            data = self.check(list(pending))
            entries: dict = data.get("result", {})

            for task_id, entry in entries.items():
                status = entry.get("status", "unknown")

                if status == "success":
                    results[task_id] = entry
                    pending.discard(task_id)

                elif status in ("error", "server_error"):
                    err = entry.get("error", "unknown error")
                    raise RuntimeError(f"Lalal task {task_id} failed: {err}")

            if pending:
                summary = ", ".join(
                    f"{tid[:8]}…={entries.get(tid, {}).get('progress', '?')}%"
                    for tid in pending
                )
                print(f"    [{elapsed:>4}s] pending: {summary}")
                time.sleep(poll_interval)
                elapsed += poll_interval

        if pending:
            raise TimeoutError(
                f"Tasks did not complete within {timeout}s: {pending}"
            )

        return results

    def download_file(self, url: str, dest: str) -> None:
        """Download *url* and save as a proper WAV file at *dest*.

        Lalal.ai returns MP3-encoded audio regardless of the requested
        filename extension. This method downloads to a temp file, decodes
        via librosa, then writes uncompressed PCM WAV — the format that
        DAWs expect and that the ``.wav`` extension promises.
        """
        resp = self.session.get(url, stream=True)
        resp.raise_for_status()

        suffix = _ext_from_content_type(resp.headers.get("Content-Type", ""))
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in resp.iter_content(chunk_size=65_536):
                tmp.write(chunk)

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=False)
            # librosa returns (channels, samples) for stereo; sf wants (samples, channels)
            if y.ndim == 2:
                y = y.T
            sf.write(dest, y, sr, subtype="PCM_24")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def delete_source(self, source_id: str) -> None:
        """Remove the source file from Lalal storage (optional cleanup)."""
        resp = self.session.post(
            f"{LALAL_BASE_URL}/api/v1/delete/",
            json={"source_id": source_id},
        )
        resp.raise_for_status()

    def minutes_left(self) -> float:
        """Return the number of processing minutes remaining on the account."""
        resp = self.session.post(f"{LALAL_BASE_URL}/api/v1/limits/minutes_left/")
        resp.raise_for_status()
        return float(resp.json()["minutes_left"])

    # ------------------------------------------------------------------
    # High-level helper
    # ------------------------------------------------------------------

    def process_and_download_stems(
        self,
        file_path: str,
        stems_dir: str,
        stems: list[tuple[str, str]] | None = None,
        splitter: str = "auto",
        extraction_level: str = "clear_cut",
        delete_after: bool = False,
    ) -> dict[str, str]:
        """Upload, split all stems concurrently, poll together, download.

        Uses individual split calls. For the batch API variant see
        :meth:`process_and_download_stems_batch`.

        Parameters
        ----------
        file_path:
            Path to the source .mp3 or .wav.
        stems_dir:
            Directory where stem files will be written.
        stems:
            ``(lalal_stem_name, output_filename)`` pairs.
            Defaults to vocals, drums, and bass.
        splitter:
            Lalal.ai model name. ``"auto"`` (default) picks best per stem.
        extraction_level:
            ``"clear_cut"`` (default) – cleaner stems, less fine detail.
            ``"deep_extraction"``     – more detail, some bleed-through.
        delete_after:
            If True, delete the source file from Lalal storage after all
            stems are downloaded (saves storage quota).

        Returns
        -------
        Mapping of stem name → local file path of downloaded stem.
        """
        if stems is None:
            stems = DEFAULT_STEMS

        out = Path(stems_dir)
        source_id = self.upload(file_path)

        task_map: dict[str, tuple[str, str]] = {}
        for stem_name, filename in stems:
            print(f"  Queuing '{stem_name}' split task…")
            task_id = self.split(source_id, stem_name, splitter, extraction_level)
            task_map[task_id] = (stem_name, filename)

        print(f"\n  Polling {len(task_map)} task(s) until complete…")
        results = self.poll_all_until_done(list(task_map.keys()))

        downloaded = self._download_results(results, task_map, out)

        if delete_after:
            print(f"  Cleaning up source file {source_id}…")
            self.delete_source(source_id)

        return downloaded

    def process_and_download_stems_batch(
        self,
        file_path: str,
        stems_dir: str,
        stems: list[tuple[str, str, str]] | None = None,
        extraction_level: str = "clear_cut",
        delete_after: bool = True,
    ) -> dict[str, str]:
        """Upload, batch-split all stems in one API call, poll, download.

        Parameters
        ----------
        stems:
            ``(lalal_api_name, output_filename, splitter)`` triples.
            Defaults to :data:`LALAL_STEMS_5`.
        extraction_level:
            ``"clear_cut"`` (default) – cleaner stems, less bleed.
        delete_after:
            If True, delete the source from Lalal storage after download.

        Returns
        -------
        Mapping of stem name → local file path.
        """
        if stems is None:
            stems = LALAL_STEMS_5

        out = Path(stems_dir)
        out.mkdir(parents=True, exist_ok=True)

        source_id = self.upload(file_path)

        print(f"  Batch-queuing {len(stems)} stem splits…")
        task_map = self.batch_split(source_id, stems, extraction_level)
        print(f"  {len(task_map)} tasks queued.")

        print("  Polling until complete…")
        results = self.poll_all_until_done(list(task_map.keys()))

        downloaded = self._download_results(results, task_map, out)

        if delete_after:
            print(f"  Cleaning up source file {source_id}…")
            self.delete_source(source_id)

        return downloaded

    def _download_results(
        self,
        results: dict[str, dict],
        task_map: dict[str, tuple[str, str]],
        out: Path,
    ) -> dict[str, str]:
        """Download stems from completed Lalal results.

        Saves the vocal "back" track as ``other.wav`` (full instrumental).
        """
        downloaded: dict[str, str] = {}
        print()
        for task_id, entry in results.items():
            stem_name, filename = task_map[task_id]
            tracks: list[dict] = entry.get("result", {}).get("tracks", [])

            stem_url = next((t["url"] for t in tracks if t["type"] == "stem"), None)
            back_url = next((t["url"] for t in tracks if t["type"] == "back"), None)

            if not stem_url:
                raise RuntimeError(
                    f"No stem URL in Lalal result for task {task_id}. "
                    f"Tracks returned: {tracks}"
                )

            dest = str(out / filename)
            print(f"  Downloading {filename}…")
            self.download_file(stem_url, dest)
            downloaded[stem_name] = dest

            if stem_name == "vocals" and back_url:
                other_dest = str(out / "other.wav")
                print("  Downloading other.wav (instrumental)…")
                self.download_file(back_url, other_dest)
                downloaded["other"] = other_dest

        return downloaded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ext_from_content_type(content_type: str) -> str:
    """Return a temp-file extension inferred from the HTTP Content-Type header."""
    ct = content_type.lower()
    if "mpeg" in ct or "mp3" in ct:
        return ".mp3"
    if "ogg" in ct:
        return ".ogg"
    if "flac" in ct:
        return ".flac"
    return ".audio"
