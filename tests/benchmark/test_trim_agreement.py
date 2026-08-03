"""Layer 3: trim auto-pick agreement ratchet.

Every web trim pick logs whether the human accepted the top-ranked
candidate. This test gates the ranking logic (madmom energy ranking +
pick_trim_auto) — a surface the kick-onset benchmark never sees.

Skips until enough feedback events exist (cold start). Once the floor
is set, agreement rate can only go up.
"""

import pytest

from musicbot.feedback import STAGE_TRIM, load_events

from .conftest import load_baseline, save_baseline

pytestmark = pytest.mark.benchmark

# Don't ratchet on a handful of early votes — noise dominates.
MIN_EVENTS = 5


def test_trim_auto_pick_agreement_does_not_regress(update_baseline):
    events = [
        e for e in load_events(STAGE_TRIM)
        if e.get("auto_pick_sec") is not None and "agreed" in e
    ]
    if len(events) < MIN_EVENTS:
        pytest.skip(
            f"Need ≥{MIN_EVENTS} trim feedback events to ratchet "
            f"(have {len(events)}). Use the web trim picker — every pick logs one."
        )

    agreed = sum(1 for e in events if e["agreed"])
    current = {"n": len(events), "agreed": agreed}
    rate = 100.0 * agreed / len(events)
    print(f"\n  [trim-agreement] {agreed}/{len(events)} auto-picks accepted ({rate:.0f}%)")

    baseline = load_baseline()

    if update_baseline:
        baseline["trim_agreement"] = current
        save_baseline(baseline)
        pytest.skip(f"baseline updated for trim_agreement: {current}")

    floor = baseline.get("trim_agreement")
    if floor is None:
        pytest.fail(
            "No trim-agreement baseline. Run: pytest tests/benchmark --update-baseline"
        )

    # Absolute agreed count must not drop; rate is allowed to dip when n grows
    # with hard cases (promote disagreements preferentially).
    assert current["agreed"] >= floor["agreed"], (
        f"Trim auto-pick agreement regressed: {current['agreed']}/{current['n']} vs "
        f"baseline {floor['agreed']}/{floor['n']}"
    )
