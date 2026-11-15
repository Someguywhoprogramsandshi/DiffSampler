import os
import sys
import unittest
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import diffsampler


class OutputGainTests(unittest.TestCase):
    def test_quiet_signal_is_matched_to_target_level(self):
        t = np.arange(44100, dtype=np.float32) / 44100.0
        src = 0.04 * np.sin(2 * np.pi * 220 * t)
        out = diffsampler.apply_output_gain(src, 100)
        self.assertAlmostEqual(float(np.quantile(np.abs(out), 0.995)), 0.50, delta=0.005)

    def test_volume_is_applied_after_auto_gain(self):
        t = np.arange(44100, dtype=np.float32) / 44100.0
        src = 0.04 * np.sin(2 * np.pi * 220 * t)
        full = diffsampler.apply_output_gain(src, 100)
        half = diffsampler.apply_output_gain(src, 50)
        ratio = float(np.sqrt(np.mean(half * half)) / np.sqrt(np.mean(full * full)))
        self.assertAlmostEqual(ratio, 0.5, delta=0.002)

    def test_auto_gain_can_be_disabled(self):
        t = np.arange(44100, dtype=np.float32) / 44100.0
        src = 0.04 * np.sin(2 * np.pi * 220 * t)
        out = diffsampler.apply_output_gain(src, 100, auto_gain=False)
        self.assertAlmostEqual(float(np.quantile(np.abs(out), 0.995)), 0.04, delta=0.001)

    def test_vocoder_padding_repeats_the_last_phoneme_frame(self):
        class CaptureSession:
            def run(self, _outputs, inputs):
                self.inputs = inputs
                frames = inputs["mel"].shape[1]
                return [np.zeros((1, frames * diffsampler.HOP), np.float32)]

        engine = diffsampler.Engine.__new__(diffsampler.Engine)
        engine.sess = CaptureSession()
        engine.gpu_lock = threading.Lock()
        mel = np.arange(5 * diffsampler.N_MELS, dtype=np.float32).reshape(5, diffsampler.N_MELS)
        f0 = np.array([180, 190, 200, 210, 220], dtype=np.float32)
        audio = engine.vocode(mel, f0)
        self.assertEqual(len(audio), 5 * diffsampler.HOP)
        padded_mel = engine.sess.inputs["mel"][0]
        padded_f0 = engine.sess.inputs["f0"][0]
        np.testing.assert_array_equal(padded_mel[5:], np.repeat(mel[-1][None, :], len(padded_mel) - 5, axis=0))
        np.testing.assert_array_equal(padded_f0[5:], np.full(len(padded_f0) - 5, f0[-1], dtype=np.float32))

    def test_silence_stays_silent(self):
        out = diffsampler.apply_output_gain(np.zeros(1024, np.float32), 100)
        self.assertEqual(float(np.max(np.abs(out))), 0.0)

    def test_non_finite_samples_are_sanitized(self):
        out = diffsampler.apply_output_gain(np.array([0.0, np.nan, np.inf, -np.inf], np.float32), 100)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(float(np.max(np.abs(out))), 0.0)


if __name__ == "__main__":
    unittest.main()
