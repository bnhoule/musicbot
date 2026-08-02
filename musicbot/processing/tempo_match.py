"""Time-stretch audio stems to a target BPM.

Uses pyrubberband for high-quality time stretching that preserves pitch
and minimises artefacts (phase-locked vocoder under the hood).

Requires the ``rubberband`` CLI tool (macOS: ``brew install rubberband``).
"""

import time

import soundfile as sf
import pyrubberband as pyrb


def stretch_audio(
    wav_path: str,
    source_bpm: float,
    target_bpm: float,
    output_path: str,
) -> str:
    """Time-stretch a WAV file from *source_bpm* to *target_bpm*.

    The stretch ratio is ``target_bpm / source_bpm`` — a ratio > 1 speeds
    the audio up, < 1 slows it down.

    Returns *output_path* for convenience.  If the BPMs already match
    (within 0.1 BPM), the file is copied unchanged.
    """
    if abs(source_bpm - target_bpm) < 0.1:
        y, sr = sf.read(wav_path, dtype="float64")
        sf.write(output_path, y, sr, subtype="PCM_24")
        return output_path

    rate = target_bpm / source_bpm
    duration_s = None

    y, sr = sf.read(wav_path, dtype="float64")
    duration_s = len(y) / sr

    t0 = time.perf_counter()
    y_stretched = pyrb.time_stretch(y, sr, rate=rate)
    rb_time = time.perf_counter() - t0
    print(f"[PERF]     rubberband stretch  {source_bpm:.3f}→{target_bpm:.3f} BPM  "
          f"audio={duration_s:.1f}s  elapsed={rb_time:.2f}s  "
          f"({rb_time/duration_s:.2f}x realtime)", flush=True)

    sf.write(output_path, y_stretched, sr, subtype="PCM_24")
    return output_path
