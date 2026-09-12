"""Reference-free signal-quality measures for released clips.

Contributors recorded on unrestricted personal devices with no signal-to-noise
threshold, so the corpus needs a description of what actually arrived. Every
measure here is computed from the released 16 kHz PCM audio alone; none of them
requires a clean reference or a noise-only recording.

Frames are 25 ms with a 10 ms hop, matching the analysis window of the acoustic
front ends evaluated in the paper.
"""

from __future__ import annotations

import numpy as np

FRAME = 400  # 25 ms at 16 kHz
HOP = 160  # 10 ms at 16 kHz
EPS = 1e-12

# A sample is treated as clipped when it sits on the 16-bit full-scale rail.
FULL_SCALE = 32767.0 / 32768.0
# Runs shorter than this are ordinary peaks rather than a flattened waveform.
MIN_CLIP_RUN = 3
# Frames this far below the clip's loud level carry no speech.
SILENCE_TOP_DB = 30.0
# An absolute floor as well: on a clip of constant level the relative rule alone
# degenerates and would report a fully silent recording as fully voiced.
SILENCE_FLOOR_DBFS = -70.0
# Percentiles of frame power standing for the noise floor and the speech level.
# A fixed dB band below the peak cannot be used instead: once noise lifts the
# floor, no frame falls inside it and precisely the noisy clips go unmeasured.
NOISE_PERCENTILE = 5
SPEECH_PERCENTILE = 95
# Fewer frames than this cannot separate speech from a pause.
MIN_FRAMES = 10
# A pause at the 16-bit quantisation floor means the capture chain gated it, so
# the estimate describes that gate rather than the room.
GATED_NOISE_DBFS = -80.0


def frame_power(signal: np.ndarray) -> np.ndarray:
    """Mean square power per analysis frame."""
    if signal.size < FRAME:
        return np.array([float(np.mean(signal.astype(np.float64) ** 2))])
    count = 1 + (signal.size - FRAME) // HOP
    strides = (signal.strides[0] * HOP, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, (count, FRAME), strides)
    return np.mean(frames.astype(np.float64) ** 2, axis=1)


def to_db(power: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.asarray(power, dtype=np.float64) + EPS)


def clipping(signal: np.ndarray) -> tuple[float, bool]:
    """Fraction of full-scale samples, and whether any flattened run occurs.

    An isolated full-scale sample is a legitimate peak; consecutive ones mean the
    waveform was truncated by the capture chain.
    """
    railed = np.abs(signal) >= FULL_SCALE
    fraction = float(np.mean(railed)) if signal.size else 0.0
    if not railed.any():
        return fraction, False
    # Longest run of consecutive full-scale samples.
    padded = np.concatenate(([False], railed, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    longest = int((edges[1::2] - edges[::2]).max())
    return fraction, longest >= MIN_CLIP_RUN


def measure(signal: np.ndarray, sample_rate: int) -> dict:
    """Per-clip signal-quality measures.

    `snr_db` compares the clip's speech level with its own noise floor, both read
    off the frame-power distribution. Against additive white noise at known
    levels it recovers 0-30 dB within about 1 dB (see the acoustics tests).

    Where `gated_silence` is set, the pause frames sit at the quantisation floor:
    the capture chain suppressed them, so the estimate describes that gate and
    not the recording environment. Those clips are reported separately rather
    than averaged into the corpus figure.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    result = {
        "samples": int(signal.size),
        "sample_rate": int(sample_rate),
        "duration_s": float(signal.size / sample_rate) if sample_rate else 0.0,
    }
    if signal.size == 0:
        return {**result, "empty": True}

    clipped_fraction, flattened = clipping(signal)
    result["clipped_fraction"] = clipped_fraction
    result["clipped_run"] = flattened
    result["peak_dbfs"] = float(to_db(float(np.max(np.abs(signal))) ** 2))
    result["dc_offset"] = float(np.mean(signal.astype(np.float64)))

    power = frame_power(signal)
    db = to_db(power)
    result["rms_dbfs"] = float(to_db(float(np.mean(signal.astype(np.float64) ** 2))))

    # "Loud level" is a high percentile rather than the maximum, so a single
    # transient cannot shift every threshold derived from it.
    loud_db = float(np.percentile(db, 95))
    silent = (db < (loud_db - SILENCE_TOP_DB)) | (db < SILENCE_FLOOR_DBFS)
    result["silence_ratio"] = float(np.mean(silent))

    voiced = np.flatnonzero(~silent)
    if voiced.size:
        result["leading_silence_s"] = float(voiced[0] * HOP / sample_rate)
        result["trailing_silence_s"] = float(
            max(signal.size - (voiced[-1] * HOP + FRAME), 0) / sample_rate
        )
        result["speech_rms_dbfs"] = float(to_db(float(np.mean(power[~silent]))))
    else:
        result["leading_silence_s"] = result["duration_s"]
        result["trailing_silence_s"] = result["duration_s"]
        result["speech_rms_dbfs"] = None

    if power.size >= MIN_FRAMES:
        noise_power = float(np.percentile(power, NOISE_PERCENTILE))
        speech_power = float(np.percentile(power, SPEECH_PERCENTILE))
        noise_floor = float(to_db(noise_power))
        result["noise_floor_dbfs"] = noise_floor
        result["gated_silence"] = noise_floor < GATED_NOISE_DBFS
        result["snr_db"] = float(to_db(max(speech_power - noise_power, EPS) / (noise_power + EPS)))
    else:
        result["noise_floor_dbfs"] = None
        result["gated_silence"] = None
        result["snr_db"] = None
    return result


def summarise(values) -> dict:
    """Median, IQR and range over the finite entries of one measure."""
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0}
    q1, median, q3 = (float(x) for x in np.percentile(finite, [25, 50, 75]))
    return {
        "n": int(finite.size),
        "min": float(finite.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "iqr": q3 - q1,
    }
