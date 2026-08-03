"""Empirical limits on how far a stem can be rekeyed or stretched.

There is no universal answer to "how many semitones is too many" — it
depends on the material and on whose ears are judging.  So the limits are
learned: every thumbs-up/down on a previewed stem records the transform
that produced it, and those votes aggregate into a per-category table.

The stack engine consults the table and warns before building something
the user has already rated as artefact-ridden.  Until enough votes exist
for a category, no limit is asserted — the table starts empty on purpose
rather than shipping invented thresholds.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LIMITS_FILE = DATA_DIR / "transform_limits.json"

# A category needs at least this many negative votes before its limit is
# trusted enough to warn on — one bad-sounding preview could be the source
# material's fault, not the transform's.
MIN_VOTES_FOR_LIMIT = 2

VERDICT_GOOD = "good"
VERDICT_BAD = "bad"


def load_limits(path: Path = LIMITS_FILE) -> dict:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_limits(limits: dict, path: Path = LIMITS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(limits, f, indent=2, sort_keys=True)
        f.write("\n")


def aggregate_limits(events: list[dict]) -> dict:
    """Derive per-category transform limits from transform feedback events.

    Each event needs ``category``, ``verdict`` and the transform that was
    applied (``semitones``, ``stretch_ratio``).

    For each category the limit is the largest magnitude still rated good,
    bounded below by the smallest magnitude rated bad — i.e. the most
    permissive setting the ear has actually endorsed, never exceeding a
    setting it has rejected.
    """
    by_cat: dict[str, dict] = {}

    for event in events:
        cat = event.get("category")
        verdict = event.get("verdict")
        if not cat or verdict not in (VERDICT_GOOD, VERDICT_BAD):
            continue

        bucket = by_cat.setdefault(cat, {
            "good_semitones": [], "bad_semitones": [],
            "good_stretch": [], "bad_stretch": [],
            "votes_good": 0, "votes_bad": 0,
        })

        semitones = event.get("semitones")
        if semitones is not None:
            key = "good_semitones" if verdict == VERDICT_GOOD else "bad_semitones"
            bucket[key].append(abs(float(semitones)))

        ratio = event.get("stretch_ratio")
        if ratio is not None:
            # Express stretch as % deviation from unity so 0.9 and 1.1 compare
            deviation = abs(float(ratio) - 1.0) * 100.0
            key = "good_stretch" if verdict == VERDICT_GOOD else "bad_stretch"
            bucket[key].append(deviation)

        bucket["votes_good" if verdict == VERDICT_GOOD else "votes_bad"] += 1

    limits: dict[str, dict] = {}
    for cat, b in by_cat.items():
        entry: dict = {"votes_good": b["votes_good"], "votes_bad": b["votes_bad"]}

        if b["bad_semitones"] and b["votes_bad"] >= MIN_VOTES_FOR_LIMIT:
            worst_ok = max(b["good_semitones"], default=0.0)
            first_bad = min(b["bad_semitones"])
            entry["max_semitones"] = round(min(worst_ok, first_bad - 1) if worst_ok >= first_bad
                                           else worst_ok, 2)

        if b["bad_stretch"] and b["votes_bad"] >= MIN_VOTES_FOR_LIMIT:
            worst_ok = max(b["good_stretch"], default=0.0)
            first_bad = min(b["bad_stretch"])
            entry["max_stretch_pct"] = round(min(worst_ok, first_bad) if worst_ok >= first_bad
                                             else worst_ok, 2)

        limits[cat] = entry

    return limits


def check_transform(
    category: str,
    semitones: float,
    stretch_ratio: float,
    limits: dict | None = None,
) -> str | None:
    """Return a warning if this transform exceeds the learned limit, else None."""
    limits = load_limits() if limits is None else limits
    entry = limits.get(category)
    if not entry:
        return None

    warnings = []

    max_semi = entry.get("max_semitones")
    if max_semi is not None and abs(semitones) > max_semi:
        warnings.append(
            f"{semitones:+.0f} semitones exceeds your rated limit of ±{max_semi:g} for {category}"
        )

    max_stretch = entry.get("max_stretch_pct")
    if max_stretch is not None:
        deviation = abs(stretch_ratio - 1.0) * 100.0
        if deviation > max_stretch:
            warnings.append(
                f"{deviation:.1f}% tempo change exceeds your rated limit of "
                f"{max_stretch:g}% for {category}"
            )

    return "; ".join(warnings) if warnings else None
