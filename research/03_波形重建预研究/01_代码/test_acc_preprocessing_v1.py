"""ACC 重采样的数值回归；不读取真实数据、不加载模型。"""
import unittest

import numpy as np
from scipy import signal

from acc_preprocessing_v1 import resample_acc
from prepare_ppg_reconstruction_pairs_v2 import acc_features


class TestAccPreprocessing(unittest.TestCase):
    def setUp(self):
        self.offset = np.array([0.25, -0.5, 1.0])
        t = np.arange(512) / 32
        self.motion = 0.003 * np.column_stack((
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * 0.75 * t),
            np.sin(2 * np.pi * 1.25 * t),
        ))
        self.raw = self.motion + self.offset
        self.interior = slice(64, -64)

    def test_legacy_is_exact_scipy_result(self):
        for raw in (self.raw, self.raw.astype(np.float32)):
            with self.subTest(dtype=raw.dtype):
                expected = signal.resample_poly(
                    np.asarray(raw, dtype=float), 2, 1, axis=0)
                np.testing.assert_array_equal(
                    resample_acc(raw, dc_normalize=False), expected)

    def test_constant_and_zero_include_boundaries(self):
        for value in (np.zeros(3), self.offset):
            raw = np.tile(value, (512, 1))
            with self.subTest(value=value.tolist()):
                corrected = resample_acc(raw, dc_normalize=True)
                np.testing.assert_allclose(
                    corrected, np.tile(value, (1024, 1)),
                    rtol=0, atol=1e-14)

    def test_constant_legacy_ripple_and_corrected_features(self):
        raw = np.tile(self.offset, (512, 1))
        legacy = resample_acc(raw, dc_normalize=False)
        corrected = resample_acc(raw, dc_normalize=True)
        phase_gap = legacy[64:-64:2].mean(0) - legacy[65:-64:2].mean(0)
        self.assertGreater(np.max(np.abs(phase_gap)), 1e-4)
        self.assertGreater(
            np.max(np.abs(acc_features(legacy[256:768]))), 0.9)
        self.assertLess(
            np.max(np.abs(acc_features(corrected[256:768]))), 1e-8)

    def test_small_real_oscillation_is_preserved(self):
        corrected = resample_acc(self.raw, dc_normalize=True)
        amplitude = np.sqrt(2) * np.std(corrected[self.interior, 0])
        self.assertAlmostEqual(amplitude / 0.003, 1.0, delta=0.01)

    def test_low_frequency_motion_is_offset_equivariant(self):
        plain = resample_acc(self.motion, dc_normalize=True)
        shifted = resample_acc(self.raw, dc_normalize=True)
        expected = np.broadcast_to(self.offset, plain[self.interior].shape)
        np.testing.assert_allclose(
            (shifted - plain)[self.interior], expected,
            rtol=0, atol=2e-14)

    def test_dynamic_lengths_finiteness_and_input_preservation(self):
        for count in (2, 3, 17, 128, 512):
            raw = self.raw[:count].copy()
            before = raw.copy()
            for mode in (False, True):
                with self.subTest(count=count, mode=mode):
                    result = resample_acc(raw, dc_normalize=mode)
                    self.assertEqual(result.shape, (2 * count, 3))
                    self.assertTrue(np.isfinite(result).all())
                    np.testing.assert_array_equal(raw, before)

    def test_mode_must_be_explicit_boolean(self):
        with self.assertRaises(TypeError):
            resample_acc(self.raw)
        for mode in (None, 0, 1, "true"):
            with self.subTest(mode=mode):
                with self.assertRaises((TypeError, ValueError)):
                    resample_acc(self.raw, dc_normalize=mode)

    def test_invalid_input_is_rejected(self):
        invalid = [
            np.zeros(3), np.zeros((4, 2)), np.zeros((2, 3, 1)),
            np.zeros((0, 3)), np.zeros((1, 3)),
        ]
        for value in (np.nan, np.inf, -np.inf):
            bad = self.raw.copy()
            bad[20, 1] = value
            invalid.append(bad)
        for raw in invalid:
            for mode in (False, True):
                with self.subTest(shape=raw.shape, mode=mode):
                    with self.assertRaises((TypeError, ValueError)):
                        resample_acc(raw, dc_normalize=mode)


if __name__ == "__main__":
    unittest.main()
