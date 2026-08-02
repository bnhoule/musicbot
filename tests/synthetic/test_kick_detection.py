"""Synthetic tests for kick transient detection (find_first_kick_time)."""

import numpy as np
import pytest

from musicbot.processing.audio_analysis import find_first_kick_time

from .conftest import SR, kick_pattern

pytestmark = pytest.mark.synthetic


class TestFindFirstKick:
    @pytest.mark.parametrize("silence_sec", [1.0, 5.0, 15.0])
    def test_finds_kick_after_silence(self, silence_sec, write_wav):
        y = kick_pattern(bpm=128.0, silence_sec=silence_sec)
        path = write_wav(y)

        trim = find_first_kick_time(path, verbose=False)

        # Synthetic accuracy contract: within one analysis frame (~23ms).
        # A pure-tone kick's amplitude has zero crossings every ~9ms, so the
        # noise-floor backtrack can stop slightly after the true attack —
        # real-kick accuracy is measured by the Layer 3 benchmark instead.
        assert silence_sec - 0.030 <= trim <= silence_sec + 0.025, (
            f"kick at {silence_sec}s, trim at {trim}s"
        )

    def test_pure_silence_returns_zero(self, write_wav):
        y = np.zeros(int(5.0 * SR))
        path = write_wav(y)
        assert find_first_kick_time(path, verbose=False) == 0.0

    def test_kick_at_start_returns_near_zero(self, write_wav):
        y = kick_pattern(bpm=128.0, silence_sec=0.05)
        path = write_wav(y)
        trim = find_first_kick_time(path, verbose=False)
        assert 0.0 <= trim <= 0.08

    def test_high_frequency_noise_is_not_a_kick(self, write_wav):
        """Hi-hat-like noise (all energy above the kick cutoff) must not
        trigger detection earlier than the real kick."""
        rng = np.random.default_rng(99)
        silence_sec = 4.0
        y = kick_pattern(bpm=128.0, silence_sec=silence_sec)

        # Overlay filtered high-frequency noise across the *whole* intro
        noise = rng.uniform(-0.3, 0.3, int(silence_sec * SR))
        # crude high-pass: difference filter removes low-frequency content
        noise = np.diff(noise, prepend=0.0)
        y[: len(noise)] += noise

        path = write_wav(y)
        trim = find_first_kick_time(path, verbose=False)
        assert trim >= silence_sec - 0.05, (
            f"hi-hat noise fooled the detector: trim={trim}s, kick={silence_sec}s"
        )
