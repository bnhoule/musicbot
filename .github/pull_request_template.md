## Summary

<!-- What does this change and why? -->

## Checklist

- [ ] `pytest tests/unit tests/synthetic` passes locally
- [ ] If detection behavior changed: `pytest tests/benchmark` run locally, and
      `baseline.json` updated via `--update-baseline` **only if scores improved**
- [ ] No audio fixtures modified by hand (regenerate via `make_bench_fixtures.py`)
