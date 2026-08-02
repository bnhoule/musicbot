"""Layer 3: kick detection benchmark against hand-labeled ground truth.

Reuses the exact method implementations from musicbot.tools.evaluate_kicks
on the committed fixture clips, then ratchets each metric against
baseline.json. Your ear defined the labels; this test makes sure no
refactor quietly drifts away from them.
"""

import pytest

from musicbot.tools.evaluate_kicks import METHOD_FNS

from .conftest import (
    MAE_EPSILON_MS,
    kick_metrics,
    load_baseline,
    load_manifest,
    save_baseline,
)

pytestmark = pytest.mark.benchmark

METHODS = list(METHOD_FNS.keys())  # baseline, madmom_groove, energy_jump


@pytest.fixture(scope="session")
def bench_results() -> dict[str, list[dict]]:
    """Run every detection method over every fixture clip, once per session."""
    manifest = load_manifest()
    results: dict[str, list[dict]] = {}

    for method in METHODS:
        detect = METHOD_FNS[method]
        rows = []
        for entry in manifest:
            detected = detect(entry["drums"], entry["raw"])
            rows.append({
                "song": entry["song"],
                "label_sec": entry["kick_sec"],
                "detected_sec": detected,
                "error_ms": (detected - entry["kick_sec"]) * 1000.0,
            })
        results[method] = rows
    return results


@pytest.mark.parametrize("method", METHODS)
def test_kick_detection_does_not_regress(method, bench_results, update_baseline):
    results = bench_results[method]
    current = kick_metrics(results)

    # Always print the scoreboard — visible with pytest -s / in CI logs
    print(f"\n  [{method}] n={current['n']}  MAE={current['mae_ms']}ms  "
          f"<50ms: {current['within_50ms']}/{current['n']}  "
          f"<500ms: {current['within_500ms']}/{current['n']}")
    worst = sorted(results, key=lambda r: abs(r["error_ms"]), reverse=True)[:3]
    for r in worst:
        print(f"    worst: {r['song']:<25} err={r['error_ms']:+.0f}ms")

    baseline = load_baseline()

    if update_baseline:
        baseline.setdefault("kick", {})[method] = current
        save_baseline(baseline)
        pytest.skip(f"baseline updated for {method}: {current}")

    floor = baseline.get("kick", {}).get(method)
    if floor is None:
        pytest.fail(
            f"No baseline recorded for method '{method}'. "
            f"Run: pytest tests/benchmark --update-baseline"
        )

    assert current["n"] >= floor["n"], (
        f"Benchmark shrank: {current['n']} songs vs baseline {floor['n']} — "
        f"missing fixtures?"
    )
    assert current["mae_ms"] <= floor["mae_ms"] + MAE_EPSILON_MS, (
        f"{method} MAE regressed: {current['mae_ms']}ms vs "
        f"baseline {floor['mae_ms']}ms (+{MAE_EPSILON_MS}ms allowed)"
    )
    assert current["within_50ms"] >= floor["within_50ms"], (
        f"{method} within-50ms regressed: {current['within_50ms']} vs "
        f"baseline {floor['within_50ms']}"
    )
    assert current["within_500ms"] >= floor["within_500ms"], (
        f"{method} within-500ms regressed: {current['within_500ms']} vs "
        f"baseline {floor['within_500ms']}"
    )
