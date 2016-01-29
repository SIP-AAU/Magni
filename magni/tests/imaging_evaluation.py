"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests for `magni.imaging.evaluation`.

**Testing Strategy**

Three cases are considered for the calculation of the evaluation metrics:

* An identical array
* A constant offset to all elements of an array
* An array subject to additive white gaussian noise

For all three cases flat, 1D, and 2D arrays are tested.

"""

from __future__ import division
import unittest

import numpy as np

from magni.imaging.evaluation import (calculate_mse, calculate_psnr,
                                      calculate_retained_energy)


class TestMetrics(unittest.TestCase):
    """
    Test of the evaluation metrics.

    """

    def setUp(self):
        n = 10
        N = n ** 2

        np.random.seed(seed=6021)

        self.flat_array = np.random.permutation(1 / N * np.arange(N))
        self.d1_array = self.flat_array.copy().reshape(N, 1)
        self.d2_array = self.flat_array.copy().reshape(n, n)

        c = 0.27
        self.flat_const_offset = c * np.ones_like(self.flat_array)
        self.d1_const_offset = c * np.ones_like(self.d1_array)
        self.d2_const_offset = c * np.ones_like(self.d2_array)

        self.flat_noise = np.random.normal(scale=1/n, size=(N,))
        self.d1_noise = self.flat_noise.copy().reshape(N, 1)
        self.d2_noise = self.flat_noise.copy().reshape(n, n)

    def test_MSE(self):
        k = len(self.flat_array)
        x_zero = np.zeros(k)
        x_flat = np.arange(k)
        x_1d = np.arange(k).reshape(*self.d1_array.shape)
        x_2d = np.arange(k).reshape(*self.d2_array.shape)

        mse_const = np.mean(self.flat_const_offset ** 2)
        mse_noise = np.mean(self.flat_noise ** 2)

        # Integrity check of test setup
        self.assertGreater(mse_const, 1e-3)
        self.assertGreater(mse_noise, 1e-3)
        self.assertNotAlmostEqual(mse_const, mse_noise)

        # Identical arrays
        self.assertEqual(calculate_mse(x_zero, x_zero), 0)
        self.assertEqual(calculate_mse(x_2d, x_2d), 0)

        # Additve offset
        self.assertAlmostEqual(
            calculate_mse(x_flat, x_flat + self.flat_const_offset), mse_const)
        self.assertAlmostEqual(
            calculate_mse(x_1d, x_1d - self.d1_const_offset), mse_const)
        self.assertAlmostEqual(
            calculate_mse(x_2d, x_2d + self.d2_const_offset), mse_const)

        # Additive noise
        self.assertAlmostEqual(
            calculate_mse(x_flat, x_flat - self.flat_noise), mse_noise)
        self.assertAlmostEqual(
            calculate_mse(x_1d, x_1d + self.d1_noise), mse_noise)
        self.assertAlmostEqual(
            calculate_mse(x_2d, x_2d - self.d2_noise), mse_noise)

        # Fails
        self.assertNotAlmostEqual(
            calculate_mse(x_flat, x_flat + self.flat_noise), mse_const)
        self.assertNotAlmostEqual(
            calculate_mse(x_flat, x_flat - self.flat_const_offset), mse_noise)

    def test_PSNR(self):
        k = len(self.flat_array)
        x_zero = np.zeros(k)
        x_flat = np.arange(k)
        x_1d = np.arange(k).reshape(*self.d1_array.shape)
        x_2d = np.arange(k).reshape(*self.d2_array.shape)

        p = 1.42
        psnr_const = 10 * np.log10(p**2 / np.mean(self.flat_const_offset ** 2))
        psnr_noise = 10 * np.log10(p**2 / np.mean(self.flat_noise ** 2))

        # Integrity check of test setup
        self.assertGreater(psnr_const, 1e-3)
        self.assertGreater(psnr_noise, 1e-3)
        self.assertNotAlmostEqual(psnr_const, psnr_noise)

        # Identical arrays
        self.assertEqual(calculate_psnr(x_zero, x_zero, p), np.inf)
        self.assertEqual(calculate_psnr(x_2d, x_2d, p), np.inf)

        # Additve offset
        self.assertAlmostEqual(
            calculate_psnr(x_flat, x_flat + self.flat_const_offset, p),
            psnr_const)
        self.assertAlmostEqual(
            calculate_psnr(x_1d, x_1d - self.d1_const_offset, p), psnr_const)
        self.assertAlmostEqual(
            calculate_psnr(x_2d, x_2d + self.d2_const_offset, p), psnr_const)

        # Additive noise
        self.assertAlmostEqual(
            calculate_psnr(x_flat, x_flat - self.flat_noise, p), psnr_noise)
        self.assertAlmostEqual(
            calculate_psnr(x_1d, x_1d + self.d1_noise, p), psnr_noise)
        self.assertAlmostEqual(
            calculate_psnr(x_2d, x_2d - self.d2_noise, p), psnr_noise)

        # Fails
        self.assertNotAlmostEqual(
            calculate_psnr(x_flat, x_flat + self.flat_noise, p), psnr_const)
        self.assertNotAlmostEqual(
            calculate_psnr(x_flat, x_flat - self.flat_const_offset, p),
            psnr_noise)
        self.assertNotAlmostEqual(
            calculate_psnr(x_flat, x_flat + self.flat_noise, 25), psnr_noise)
        self.assertNotAlmostEqual(
            calculate_psnr(x_flat, x_flat + self.flat_noise, p*1.01),
            psnr_noise)

    def test_retained_energy(self):
        k = len(self.flat_array)
        x_zero = np.zeros(k)
        x_flat = np.ones(k)
        x_1d = np.ones(k).reshape(*self.d1_array.shape)
        x_2d = np.ones(k).reshape(*self.d2_array.shape)

        energy_const = 1/k * np.sum(self.flat_const_offset ** 2) * 100
        energy_noise = 1/k * np.sum(self.flat_noise ** 2) * 100

        # Integrity check of test setup
        self.assertGreater(energy_const, 1e-3)
        self.assertGreater(energy_noise, 1e-3)
        self.assertNotAlmostEqual(energy_const, energy_noise)

        # Identical arrays and zero case
        with self.assertRaises(ValueError):
            calculate_retained_energy(x_zero, x_zero)

        self.assertEqual(calculate_retained_energy(x_flat, x_flat), 100)
        self.assertEqual(calculate_retained_energy(x_flat, x_zero), 0)

        # Additve offset
        self.assertAlmostEqual(
            calculate_retained_energy(
                x_flat, self.flat_const_offset), energy_const)
        self.assertAlmostEqual(
            calculate_retained_energy(
                x_1d, self.d1_const_offset), energy_const)
        self.assertAlmostEqual(
            calculate_retained_energy(
                x_2d, self.d2_const_offset), energy_const)

        # Additive noise
        self.assertAlmostEqual(
            calculate_retained_energy(x_flat, self.flat_noise), energy_noise)
        self.assertAlmostEqual(
            calculate_retained_energy(x_1d, self.d1_noise), energy_noise)
        self.assertAlmostEqual(
            calculate_retained_energy(x_2d, self.d2_noise), energy_noise)

        # Fails
        self.assertNotAlmostEqual(
            calculate_retained_energy(x_flat, self.flat_noise), energy_const)
        self.assertNotAlmostEqual(
            calculate_mse(x_flat, self.flat_const_offset), energy_noise)
        with self.assertRaises(ValueError):
            calculate_retained_energy(x_zero, x_flat)
