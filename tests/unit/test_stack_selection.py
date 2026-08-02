"""Unit tests for stem stack random selection (musicbot.processing.stack_engine)."""

import random

import pytest

from processing.stack_engine import CATEGORIES, random_selection

pytestmark = pytest.mark.unit

LIBRARY = {name: {} for name in ["Song A", "Song B", "Song C"]}


class TestRandomSelection:
    def test_empty_library_returns_empty(self):
        assert random_selection({}) == {}

    def test_selects_every_category(self):
        random.seed(42)
        sel = random_selection(LIBRARY)
        assert set(sel.keys()) == set(CATEGORIES)
        assert all(name in LIBRARY for name in sel.values())

    def test_deterministic_with_seed(self):
        random.seed(7)
        first = random_selection(LIBRARY)
        random.seed(7)
        second = random_selection(LIBRARY)
        assert first == second

    def test_exclude_is_honored(self):
        random.seed(1)
        exclude = dict.fromkeys(CATEGORIES, "Song A")
        for _ in range(20):
            sel = random_selection(LIBRARY, exclude=exclude)
            assert all(name != "Song A" for name in sel.values())

    def test_exclude_falls_back_when_pool_empty(self):
        # Only one entry and it's excluded — must still return something
        single = {"Only Song": {}}
        exclude = dict.fromkeys(CATEGORIES, "Only Song")
        sel = random_selection(single, exclude=exclude)
        assert all(name == "Only Song" for name in sel.values())
