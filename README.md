# musicbot

Automated remix preparation pipeline and stem-stack shuffler. Feed it songs, get back separated stems, BPM, musical key, and beat grids — then mix-and-match stems from different songs with automatic rekeying and tempo matching.

## What it does

### 1. Process songs

1. Separates stems using **Demucs** (local, default) or **Lalal.ai** (remote).
2. Downloads/writes `vocals.wav`, `drums.wav`, `bass.wav`, and `other.wav`.
3. Beat-grid trims all stems so they start exactly on bar 1 beat 1.
4. Analyses the original file with **librosa** to detect BPM, musical key, and a full beat grid.
5. Converts the key to its **Camelot Wheel** code.
6. Writes everything into a structured output folder with `metadata.json`.

### 2. Stack & shuffle

Once you've processed a few songs, the **stack shuffler** lets you:

- Randomly pick one stem per category (drums, bass, vocals, other) from different songs.
- Auto-rekey and tempo-match every stem to a common target.
- Layer them into a preview mix.
- Swap any category forward/back with no repeats until the pool is exhausted.
- Export the current stack as individual aligned stems + a mixed `stack.wav`.

## Output structure

```
processed/
  <song_name>/
    stems/
      vocals.wav
      drums.wav
      bass.wav
      other.wav
    metadata.json
  stacks/
    <timestamp>/
      drums.wav
      bass.wav
      vocals.wav
      other.wav
      stack.wav
      stack.json
```

`metadata.json` example:

```json
{
  "original_filename": "my_track.mp3",
  "bpm": 128.0,
  "key": "A minor",
  "camelot_key": "8A",
  "beat_times": [0.42, 0.88, 1.35, 1.81],
  "trim_offset_seconds": 0.41,
  "timestamp_processed": "2026-03-14T12:00:00+00:00"
}
```

## Requirements

- Python 3.10+
- **rubberband** CLI tool (for rekeying and tempo matching)
- *(Optional)* A [Lalal.ai](https://www.lalal.ai) API key — only if using `--backend lalal`

```bash
# macOS
brew install rubberband

# Ubuntu / Debian
sudo apt install rubberband-cli
```

## Setup

```bash
# 1. Clone / copy the project
cd MusicGenerator

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Processing songs

```bash
# Default: uses Demucs (runs locally, no API key needed)
python musicbot/process_song.py path/to/song.mp3

# Custom output directory
python musicbot/process_song.py path/to/song.wav --output ~/stems

# Use Lalal.ai instead (requires API key)
python musicbot/process_song.py path/to/song.mp3 --backend lalal --api-key YOUR_KEY
```

### Stack shuffler

```bash
# Launch the interactive shuffler (uses first pick's key/BPM by default)
python musicbot/stack.py --input processed/

# Or pin a target key and BPM
python musicbot/stack.py --input processed/ --target-bpm 128 --target-key "A minor"
```

Interactive commands:

| Key | Action |
|-----|--------|
| `d` / `D` | Next / prev **drums** |
| `b` / `B` | Next / prev **bass** |
| `v` / `V` | Next / prev **vocals** |
| `o` / `O` | Next / prev **other** |
| `r` | Re-roll all categories |
| `e` | Export current stack |
| `?` | Help |
| `q` | Quit |

## Project structure

```
MusicGenerator/
  musicbot/
    process_song.py       ← CLI entry point and pipeline orchestrator
    demucs_separator.py   ← Local stem separation via Demucs (default)
    lalal_api.py          ← Remote stem separation via Lalal.ai (optional)
    audio_analysis.py     ← BPM + key + beat-grid detection via librosa
    rekey.py           ← Pitch-shift stems to a target key (pyrubberband)
    tempo_match.py     ← Time-stretch stems to a target BPM (pyrubberband)
    stack.py           ← Interactive stem-stack shuffler
    utils.py           ← Shared helpers (paths, JSON, filename sanitization)
  requirements.txt
  README.md
```

## Getting a Lalal.ai API key (optional)

Only needed if you want to use `--backend lalal` instead of the default Demucs.

1. Create an account at <https://www.lalal.ai>.
2. Go to **Account → API** to find or generate your license key.
3. The free tier processes a 30-second preview; a paid plan unlocks full tracks.

## Testing & quality gates

Three test layers, all enforced by CI on every pull request:

```bash
# Layer 1+2 — unit + synthetic audio tests (~3s)
pytest tests/unit tests/synthetic

# Layer 3 — ear benchmark on real labeled songs (~2min, needs LFS fixtures)
pytest tests/benchmark

# Lint
ruff check .
```

**AI review:** every PR is reviewed by CodeRabbit (required `CodeRabbit`
status check + comment threads must be resolved before merge) and Cursor
Bugbot. Review rules live in `.coderabbit.yaml` and `.cursor/BUGBOT.md`.

**The ratchet:** `tests/benchmark/baseline.json` records the current kick
detection accuracy (MAE, within-50ms, within-500ms per method) and BPM
hit-rate against the hand-labeled ground truth in `data/kick_labels.csv`.
Any change that makes these numbers worse fails CI. When you genuinely
improve detection, raise the floor in the same PR:

```bash
pytest tests/benchmark --update-baseline
```

**Growing the benchmark from use:** every time you use the web UI, your ear
feeds the gates — no separate labeling session required:

| When you… | What gets logged | What it gates |
|---|---|---|
| Commit a trim pick | Implicit vote: did you accept the auto-pick? (`feedback.jsonl`) | Trim-ranker agreement ratchet |
| Thumbs-up/down the detected key | Confirmed/corrected key → `key_labels.csv` | Key-detection accuracy ratchet |
| Rate a stacked stem Clean/Artifacts | Transform params + verdict → learns `transform_limits.json` | Stacker warns before exceeding your limits |

Promote hard trim cases (where you overrode the auto-pick) into the kick benchmark:

```bash
python musicbot/tools/promote_labels.py --list   # sorted by disagreement
python musicbot/tools/promote_labels.py --promote "122 - Purple Line.mp3"
```

After enough new labels, refresh the floor in the same PR:

```bash
pytest tests/benchmark --update-baseline
```

## Extending for DAW integration

`metadata.json` is designed to be machine-readable so you can later:
- Auto-import stems into Ableton/Logic via ALS/XML project scripts.
- Match BPM/key with a track library for automatic playlist building.
- Feed Camelot codes into a harmonic mixing engine.


<!-- fast-path verification -->
