"""Shared machinery for the real-audio ear benchmark (Layer 3).

The gate is a ratchet: current scores live in baseline.json and every PR
must match or beat them. After a genuine improvement, refresh the floor
in the same PR with:

    pytest tests/benchmark --update-baseline
"""

import csv
import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "kick_bench"
MANIFEST = FIXTURES_DIR / "labels.csv"
BASELINE_FILE = Path(__file__).parent / "baseline.json"

# Ratchet tolerances — generous enough to absorb float jitter across
# platforms, tight enough that a real regression trips the gate.
MAE_EPSILON_MS = 5.0


def pytest_addoption(parser):
    parser.addoption(
        "--update-baseline",
        action="store_true",
        default=False,
        help="Rewrite tests/benchmark/baseline.json with current scores instead of gating",
    )


@pytest.fixture(scope="session")
def update_baseline(request) -> bool:
    return request.config.getoption("--update-baseline")


def load_manifest() -> list[dict]:
    if not MANIFEST.is_file():
        pytest.skip(
            "Benchmark fixtures missing — run musicbot/tools/make_bench_fixtures.py "
            "(or `git lfs pull` if you cloned without LFS)"
        )
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = FIXTURES_DIR / f"{row['slug']}__raw.wav"
            drums = FIXTURES_DIR / f"{row['slug']}__drums.flac"
            if not (raw.is_file() and drums.is_file()):
                continue
            rows.append({
                "song": row["song"],
                "kick_sec": float(row["kick_sec"]),
                "bpm": float(row["bpm"]) if row["bpm"] else None,
                "raw": raw,
                "drums": drums,
            })
    if not rows:
        pytest.skip("Benchmark manifest present but no fixture audio found — git lfs pull?")
    return rows


def load_baseline() -> dict:
    if BASELINE_FILE.is_file():
        with open(BASELINE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_baseline(baseline: dict) -> None:
    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
        f.write("\n")


def kick_metrics(results: list[dict]) -> dict:
    """Aggregate per-song errors into the gated metric set."""
    abs_errors = [abs(r["error_ms"]) for r in results]
    return {
        "n": len(results),
        "mae_ms": round(sum(abs_errors) / len(abs_errors), 1),
        "within_50ms": sum(1 for e in abs_errors if e <= 50),
        "within_500ms": sum(1 for e in abs_errors if e <= 500),
    }
