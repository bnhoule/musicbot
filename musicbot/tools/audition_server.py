"""Local click-through UI for ear-auditing kick labels.

    python musicbot/tools/audition_server.py
    # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import urllib.parse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = REPO_ROOT / "data" / "kick_labels.csv"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "kick_bench"
HOST, PORT = "127.0.0.1", 8765
FIELDNAMES = ["Song", "Kick Start (seconds)", "Verified"]
PREVIEW_SEC = 4.0
CONTEXT_SEC = 1.5


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name).strip("_")


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


def fixture_paths(song: str) -> tuple[Path, Path]:
    slug = sanitize(song)
    return FIXTURES_DIR / f"{slug}__raw.wav", FIXTURES_DIR / f"{slug}__drums.flac"


def cut_snippet(path: Path, start_sec: float, duration: float) -> bytes:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    start = max(0, int(start_sec * sr))
    end = min(len(y), start + int(duration * sr))
    buf = io.BytesIO()
    sf.write(buf, y[start:end], sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Kick label audition</title>
<style>
  :root {
    --bg: #141414; --panel: #1e1e1e; --text: #ececec; --muted: #9a9a9a;
    --line: #333; --good: #2f6f4e; --fix: #6b5a2e; --skip: #3a3a3a; --accent: #c8c8c8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); min-height: 100vh;
  }
  main { max-width: 720px; margin: 0 auto; padding: 40px 24px 80px; }
  h1 { font-size: 28px; font-weight: 500; margin: 0 0 8px; letter-spacing: -0.02em; }
  .sub { color: var(--muted); margin: 0 0 28px; line-height: 1.45; }
  .progress { color: var(--muted); font-size: 13px; margin-bottom: 18px; }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 22px 22px 18px;
  }
  .song { font-size: 22px; margin: 0 0 6px; }
  .meta { color: var(--muted); margin: 0 0 18px; font-size: 14px; }
  .hint {
    border-left: 3px solid var(--accent); padding: 8px 12px; margin: 0 0 20px;
    color: var(--muted); font-size: 14px; line-height: 1.4;
  }
  .row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }
  button, .btn {
    appearance: none; border: 1px solid var(--line); background: #2a2a2a; color: var(--text);
    border-radius: 8px; padding: 12px 16px; font: inherit; font-size: 15px; cursor: pointer;
  }
  button:hover { filter: brightness(1.12); }
  button.primary { background: var(--good); border-color: #3d8a62; }
  button.secondary { background: var(--fix); border-color: #8a763d; }
  button.ghost { background: var(--skip); }
  input[type="number"] {
    width: 120px; background: #111; color: var(--text); border: 1px solid var(--line);
    border-radius: 8px; padding: 12px 12px; font: inherit;
  }
  .status { min-height: 22px; color: var(--muted); font-size: 13px; margin-top: 8px; }
  .done { text-align: center; padding: 40px 10px; }
  .done h2 { margin: 0 0 10px; font-weight: 500; }
  audio { width: 100%; margin: 8px 0 16px; }
</style>
</head>
<body>
<main>
  <h1>Kick label audition</h1>
  <p class="sub">Playback starts at the labeled kick. If the label is right, you should hear the kick immediately — no dead air.</p>
  <div id="app"></div>
</main>
<script>
let songs = [];
let idx = 0;
let audio = new Audio();

async function load() {
  const res = await fetch("/api/songs");
  songs = await res.json();
  // start on first unverified, else first
  idx = songs.findIndex(s => !s.verified);
  if (idx < 0) idx = 0;
  render();
  if (songs.length) play("raw");
}

function current() { return songs[idx]; }

async function play(mode) {
  const s = current();
  if (!s) return;
  const start = mode === "context" ? Math.max(0, s.kick_sec - 1.5) : s.kick_sec;
  const duration = mode === "context" ? 5.5 : 4.0;
  const stem = mode === "drums" ? "drums" : "raw";
  const url = `/api/snippet?song=${encodeURIComponent(s.song)}&stem=${stem}&start=${start}&duration=${duration}&_=${Date.now()}`;
  audio.pause();
  audio.src = url;
  try { await audio.play(); setStatus(`Playing ${stem} from ${start.toFixed(3)}s`); }
  catch (e) { setStatus("Click Replay to start audio (browser blocked autoplay)"); }
}

async function act(action, kickSec) {
  const s = current();
  const body = { song: s.song, action, kick_sec: kickSec };
  const res = await fetch("/api/act", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });
  const data = await res.json();
  songs = data.songs;
  if (action === "good" || action === "skip" || action === "fix") {
    // advance to next unverified after current
    let next = idx + 1;
    while (next < songs.length && songs[next].verified) next++;
    if (next >= songs.length) {
      // maybe earlier unverified
      next = songs.findIndex(x => !x.verified);
      if (next < 0) next = songs.length; // done
    }
    idx = next;
  }
  render();
  if (idx < songs.length) play(action === "fix" ? "raw" : "raw");
}

function setStatus(msg) {
  const el = document.getElementById("status");
  if (el) el.textContent = msg;
}

function render() {
  const app = document.getElementById("app");
  const verified = songs.filter(s => s.verified).length;
  if (idx >= songs.length) {
    app.innerHTML = `<div class="card done">
      <h2>All done</h2>
      <p class="sub">${verified}/${songs.length} verified. Labels saved to data/kick_labels.csv.</p>
      <p class="sub">If you fixed any times, next step is regenerate fixtures + update baseline in a PR.</p>
    </div>`;
    return;
  }
  const s = current();
  app.innerHTML = `
    <div class="progress">${idx + 1} / ${songs.length} · ${verified} verified</div>
    <div class="card">
      <p class="song">${escapeHtml(s.song)}</p>
      <p class="meta">Label: <strong>${s.kick_sec.toFixed(3)}s</strong>
        ${s.verified ? ` · verified ${escapeHtml(s.verified)}` : " · not verified yet"}</p>
      <p class="hint">Correct label = kick hits immediately. Early = dead air first. Late = you miss the transient / start mid-kick.</p>
      <audio controls id="player"></audio>
      <div class="row">
        <button class="primary" id="good">Good — keep this</button>
        <button class="ghost" id="replay">Replay</button>
        <button class="ghost" id="context">+1.5s context</button>
        <button class="ghost" id="drums">Drums only</button>
        <button class="ghost" id="skip">Skip</button>
      </div>
      <div class="row">
        <input id="fixVal" type="number" step="0.001" min="0" value="${s.kick_sec.toFixed(3)}" />
        <button class="secondary" id="fixBtn">Set new time & confirm</button>
      </div>
      <div class="status" id="status"></div>
    </div>`;
  document.getElementById("good").onclick = () => act("good");
  document.getElementById("replay").onclick = () => play("raw");
  document.getElementById("context").onclick = () => play("context");
  document.getElementById("drums").onclick = () => play("drums");
  document.getElementById("skip").onclick = () => act("skip");
  document.getElementById("fixBtn").onclick = () => {
    const v = parseFloat(document.getElementById("fixVal").value);
    if (Number.isFinite(v)) act("fix", v);
  };
  // wire visible audio element to shared Audio for controls
  const player = document.getElementById("player");
  player.src = audio.src || "";
  audio = player;
}

function escapeHtml(t) {
  return String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

load();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter
        if args and str(args[0]).startswith("GET /api/snippet"):
            return
        super().log_message(fmt, *args)

    def _json(self, code: int, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, code: int, data: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._bytes(200, HTML.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/songs":
            rows = load_labels()
            payload = []
            for r in rows:
                raw, drums = fixture_paths(r["Song"])
                payload.append({
                    "song": r["Song"],
                    "kick_sec": float(r["Kick Start (seconds)"]),
                    "verified": r["Verified"],
                    "has_raw": raw.is_file(),
                    "has_drums": drums.is_file(),
                })
            self._json(200, payload)
            return
        if parsed.path == "/api/snippet":
            qs = urllib.parse.parse_qs(parsed.query)
            song = qs.get("song", [""])[0]
            stem = qs.get("stem", ["raw"])[0]
            start = float(qs.get("start", ["0"])[0])
            duration = float(qs.get("duration", [str(PREVIEW_SEC)])[0])
            raw, drums = fixture_paths(song)
            path = drums if stem == "drums" else raw
            if not path.is_file():
                self._json(404, {"error": f"missing {path.name}"})
                return
            data = cut_snippet(path, start, duration)
            self._bytes(200, data, "audio/wav")
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/act":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        song = body.get("song")
        action = body.get("action")
        rows = load_labels()
        for r in rows:
            if r["Song"] != song:
                continue
            if action == "good":
                r["Verified"] = date.today().isoformat()
            elif action == "fix":
                kick = float(body["kick_sec"])
                r["Kick Start (seconds)"] = f"{kick}"
                r["Verified"] = date.today().isoformat()
            elif action == "skip":
                pass
            break
        save_labels(rows)
        payload = [{
            "song": r["Song"],
            "kick_sec": float(r["Kick Start (seconds)"]),
            "verified": r["Verified"],
        } for r in rows]
        self._json(200, {"ok": True, "songs": payload})


def main() -> None:
    mimetypes.add_type("audio/wav", ".wav")
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\n  Kick audition UI → http://{HOST}:{PORT}\n  Ctrl+C to stop.\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
