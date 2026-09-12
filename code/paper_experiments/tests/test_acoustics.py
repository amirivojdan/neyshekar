"""The signal-quality measures must hold up on signals with a known answer."""

import unittest

import numpy as np

from neyshekar_experiments.acoustics import FULL_SCALE, clipping, measure, summarise

RATE = 16_000


def speech_like(seconds=4.0, seed=0, level=0.2, pause=True):
    """Amplitude-modulated tone bursts separated by pauses, as read speech is."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    signal = np.sin(2 * np.pi * 140 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    signal += 0.3 * np.sin(2 * np.pi * 430 * t)
    signal *= rng.uniform(0.8, 1.2, n)
    if pause:
        # Silence the first and last fifth, leaving speech in between.
        signal[: n // 5] = 0.0
        signal[-n // 5 :] = 0.0
    return (level * signal / np.max(np.abs(signal))).astype(np.float32)


class Clipping(unittest.TestCase):
    def test_isolated_peaks_are_not_clipping(self):
        signal = np.zeros(1000, dtype=np.float32)
        signal[[10, 200, 700]] = 1.0
        fraction, flattened = clipping(signal)
        self.assertAlmostEqual(fraction, 3 / 1000)
        self.assertFalse(flattened, "single full-scale samples are peaks, not a flattened waveform")

    def test_flattened_run_is_clipping(self):
        signal = np.zeros(1000, dtype=np.float32)
        signal[100:110] = 1.0
        fraction, flattened = clipping(signal)
        self.assertAlmostEqual(fraction, 10 / 1000)
        self.assertTrue(flattened)

    def test_just_below_full_scale_is_clean(self):
        signal = np.full(1000, FULL_SCALE - 1e-4, dtype=np.float32)
        fraction, flattened = clipping(signal)
        self.assertEqual(fraction, 0.0)
        self.assertFalse(flattened)


class SilenceAndLevel(unittest.TestCase):
    def test_leading_and_trailing_pauses_are_located(self):
        result = measure(speech_like(seconds=5.0), RATE)
        self.assertGreater(result["leading_silence_s"], 0.7)
        self.assertLess(result["leading_silence_s"], 1.3)
        self.assertGreater(result["trailing_silence_s"], 0.7)
        self.assertLess(result["trailing_silence_s"], 1.3)
        # Two fifths of the clip are silent by construction.
        self.assertGreater(result["silence_ratio"], 0.3)
        self.assertLess(result["silence_ratio"], 0.5)

    def test_continuous_speech_has_little_silence(self):
        result = measure(speech_like(pause=False), RATE)
        self.assertLess(result["silence_ratio"], 0.1)

    def test_level_tracks_amplitude(self):
        quiet = measure(speech_like(level=0.02), RATE)
        loud = measure(speech_like(level=0.5), RATE)
        self.assertLess(quiet["rms_dbfs"], loud["rms_dbfs"] - 15)
        self.assertLess(loud["peak_dbfs"], 0.1)

    def test_silence_is_reported_without_crashing(self):
        result = measure(np.zeros(RATE, dtype=np.float32), RATE)
        self.assertEqual(result["silence_ratio"], 1.0)
        self.assertIsNone(result["speech_rms_dbfs"])

    def test_empty_signal(self):
        self.assertTrue(measure(np.zeros(0, dtype=np.float32), RATE)["empty"])


class SnrRecovery(unittest.TestCase):
    """Additive white noise at a known level must come back out again.

    This is the property the corpus figure rests on, so it is asserted across the
    range that matters rather than spot-checked at one level.
    """

    def test_known_snr_is_recovered(self):
        rng = np.random.default_rng(7)
        clean = speech_like(seconds=6.0)
        active = clean[np.abs(clean) > np.percentile(np.abs(clean), 75)]
        speech_power = float(np.mean(active.astype(np.float64) ** 2))
        for target in (0, 5, 10, 15, 20, 30):
            noise = rng.normal(0, np.sqrt(speech_power / 10 ** (target / 10)), clean.size)
            estimate = measure((clean + noise).astype(np.float32), RATE)["snr_db"]
            self.assertIsNotNone(estimate, f"no estimate produced at {target} dB")
            self.assertLess(
                abs(estimate - target), 3.0, f"{target} dB recovered as {estimate:.1f} dB"
            )

    def test_noise_raising_the_floor_still_yields_an_estimate(self):
        """The earlier fixed-band rule returned None here, hiding noisy clips."""
        rng = np.random.default_rng(3)
        clean = speech_like(seconds=6.0)
        noisy = clean + rng.normal(0, 0.05, clean.size)
        self.assertIsNotNone(measure(noisy.astype(np.float32), RATE)["snr_db"])

    def test_digital_silence_is_flagged_as_gated(self):
        gated = measure(speech_like(seconds=6.0), RATE)
        self.assertTrue(gated["gated_silence"])
        rng = np.random.default_rng(1)
        room = speech_like(seconds=6.0) + rng.normal(0, 0.002, int(6.0 * RATE))
        self.assertFalse(measure(room.astype(np.float32), RATE)["gated_silence"])


class Summary(unittest.TestCase):
    def test_ignores_missing_values(self):
        summary = summarise([1.0, None, 2.0, float("nan"), 3.0, 4.0])
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)
        self.assertAlmostEqual(summary["iqr"], 1.5)

    def test_empty(self):
        self.assertEqual(summarise([None, float("nan")]), {"n": 0})


if __name__ == "__main__":
    unittest.main()
