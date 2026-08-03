"""Append-only feedback log — every judgement the tool collects lands here.

One JSON object per line in ``data/feedback.jsonl``, versioned in git next
to the code that produced it.  Each event records the commit SHA, so a
quality dip can be traced back to the change that caused it.

Raw events are deliberately *not* ground truth.  They are curated into
labels (``kick_labels.csv``, ``key_labels.csv``) and aggregates
(``transform_limits.json``) by the promotion tools, which is what the
benchmarks actually gate on.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"

# Stages that produce feedback
STAGE_TRIM = "trim"            # trim point chosen (implicit vote on the ranking)
STAGE_KEY = "key"              # explicit vote on detected musical key
STAGE_TRANSFORM = "transform"  # explicit vote on rekey/stretch artefacts

_git_sha: str | None = None


def git_sha() -> str:
    """Short commit SHA of the working tree, or 'unknown' outside a repo."""
    global _git_sha
    if _git_sha is None:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(DATA_DIR.parent),
                capture_output=True, text=True, timeout=5,
            )
            _git_sha = out.stdout.strip() if out.returncode == 0 else "unknown"
        except (OSError, subprocess.SubprocessError):
            _git_sha = "unknown"
    return _git_sha


def log_event(stage: str, path: Path = FEEDBACK_FILE, **payload) -> dict:
    """Append one feedback event and return the stored record."""
    record = {
        "stage": stage,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def load_events(stage: str | None = None, path: Path = FEEDBACK_FILE) -> list[dict]:
    """Read back events, optionally filtered to one stage.

    Malformed lines are skipped rather than raising — a corrupt tail should
    never take down the app or the benchmarks.
    """
    if not path.is_file():
        return []

    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if stage is None or event.get("stage") == stage:
                events.append(event)
    return events
