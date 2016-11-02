"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests for `magni.cs.reconstruction`.

**Testing Strategy**
The usage of FastOps is tested along with reconstructions in various points in
the phase space.

**Phase Space Tests Overview**
A set of :math:`(\delta, \rho)` points in the phase space is selected. For each
algorithm, the reconstruction capabilities in each point has been determined
for a given problem suite. The tests are based on a comparison with these
reference results. Specifically, a comparison based on `np.allclose` is done.
Also some border cases (extremes) like :math:`k = 0` are tested.


Points where it is likely to have positive results
+-----+------+------+------+------+------+------+------+
|  no.|   1  |   2  |   3  |   4  |   5  |   6  |   7  |
+-----+------+------+------+------+------+------+------+
|delta| 0.08 | 0.24 | 0.38 | 0.62 | 0.78 | 0.84 | 0.96 |
+-----+------+------+------+------+------+------+------+
|  rho| 0.05 | 0.01 | 0.12 | 0.38 | 0.22 | 0.08 | 0.91 |
+-----+------+------+------+------+------+------+------+


Points where it is unlikely to have positive results
+-----+------+------+------+
|  no.|   A  |   B  |   C  |
+-----+------+------+------+
|delta| 0.06 | 0.19 | 0.29 |
+-----+------+------+------+
|  rho| 0.92 | 0.84 | 0.94 |
+-----+------+------+------+


**Functions Tested**

See the docstrings of the below listed classes.

Routine listings
----------------
ComparisonGAMPTests(unittest.TestCase)
    Comparison of magni.cs.reconstruction.gamp to a reference implementation.
FastOpsTests(unittest.TestCase)
    Tests of FastOp input, i.e. function based measurements and FFT dictionary.
FeatureTest(object)
    Reconstruction algorithm feature test base class.
FeaturePrecisionFloatTest(FeatureTest, unittest.TestCase)
    Test of the precision float feature in reconstruction algorithms.
FeatureReportHistoryTest(FeatureTest, unittest.TestCase)
    Test the report history feature in reconstruction algorithms.
FeatureStopCriterionTest(FeatureTest, unittest.TestCase)
    Test of the stop criterion feature in reconstruction algorithms.
FeatureWarmStartTest(FeatureTest, unittest.TestCase)
    Test of the warm_start feature in reconstruction algorithms.
PhaseSpaceExtremesTest(unittest.TestCase):
    Tests of border case (extreme) phase space values.
PhaseSpaceTest(object)
    Phase space test base class.
PhaseSpaceTest1(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.08, 0.05)
PhaseSpaceTest2(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.24, 0.01)
PhaseSpaceTest3(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.38, 0.12)
PhaseSpaceTest4(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.62, 0.38)
PhaseSpaceTest5(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.78, 0.22)
PhaseSpaceTest6(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.84, 0.08)
PhaseSpaceTest7(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.96, 0.91)
PhaseSpaceTestA(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.06, 0.92)
PhaseSpaceTestB(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.19, 0.84)
PhaseSpaceTestC(PhaseSpaceTest, unittest.TestCase)
    Test of reconstruction capabilities at Phase Space point (0.29, 0.94)
TestUSERademacher(unittest.TestCase)
    Test of the use_rademacher test fixture function.
use_rademacher(n, m, k, seed)
    Prepare an instance of the USE/Rademacher problem suite

"""

from __future__ import division
import os
import unittest
import warnings

import numpy as np

import magni
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


class ComparisonGAMPTests(unittest.TestCase):
    """
    Comparison of magni.cs.reconstruction.gamp to a reference implementation.

    **Reference implementation**
    "run_amp" from
    https://github.com/eric-tramel/SwAMP-Demo/blob/master/python/amp.py
    commit b32755caa8d6b59929174e2a06cc685bae5849b6

    """

    def setUp(self):
        self.ns = [1024, 2048, 2048, 2000, 1000]
        self.ms = [770, 780, 1024, 800, 500]
        self.ks = [440, 440, 1024, 126, 88]
        self.sigma_sqs = [0.0, 0.0, 0.0, 1e-3, 1e-2]
        self.sigma_sqs_init = [1e-6, 1e-6, 1e-6, 1, 1]
        self.theta_bars = [0.0, 0.0, 0.0, 0.2, 0.0]
        self.theta_tildes = [1.0, 1.0, 1.0, 0.3, 1.0]
        self.taus = [float(k) / n for (k, n) in zip(self.ks, self.ns)]

        self.tolerance = 1e-16
        self.iterations = 500

        np.random.seed(6021)
        self.seeds = np.random.randint(1000, 80000, size=len(self.ns))

        file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
        fixed_sigma_sq_sol_file = np.load(
            file_path + 'gamp_fixed_sigma_sq_sols.npz')
        em_sigma_sq_sol_file = np.load(
            file_path + 'gamp_em_sigma_sq_sols.npz')
        self.comparison_solutions_fixed_sigma_sq = [
            fixed_sigma_sq_sol_file[arr] for arr in sorted(
                fixed_sigma_sq_sol_file.files)]
        self.comparison_solutions_em_sigma_sq = [
            em_sigma_sq_sol_file[arr] for arr in sorted(
                em_sigma_sq_sol_file.files)]

    def tearDown(self):
        magni.cs.reconstruction.gamp.config.reset()

    def testFixedGAMPComparison(self):
        for ix in range(len(self.ns)):
            # Setup GAMP solver
            input_channel_params = {'tau': self.taus[ix],
                                    'theta_bar': self.theta_bars[ix],
                                    'theta_tilde': self.theta_tildes[ix],
                                    'use_em': False}
            output_channel_params = {
                'sigma_sq': self.sigma_sqs_init[ix],
                'noise_level_estimation': 'fixed'}
            gamp_config = {'tolerance': self.tolerance,
                           'iterations': self.iterations,
                           'input_channel_parameters': input_channel_params,
                           'output_channel_parameters': output_channel_params}
            magni.cs.reconstruction.gamp.config.update(gamp_config)

            # Generate problem instance
            z, A, alpha = use_gaussian(
                self.ns[ix], self.ms[ix], self.ks[ix], self.seeds[ix])
            A_asq = np.abs(A)**2
            if self.sigma_sqs[ix] > 0:
                y = z + np.random.normal(
                    size=z.shape, loc=0.0, scale=np.sqrt(self.sigma_sqs[ix]))
            else:
                y = z

            # Run solver
            alpha_hat = magni.cs.reconstruction.gamp.run(y, A, A_asq)

            # Compare result
            self.assertTrue(
                np.allclose(
                    self.comparison_solutions_fixed_sigma_sq[ix],
                    alpha_hat.flatten()))

    def testEMGAMPComparison(self):
        for ix in range(len(self.ns)):
            # Setup GAMP solver
            input_channel_params = {'tau': self.taus[ix],
                                    'theta_bar': self.theta_bars[ix],
                                    'theta_tilde': self.theta_tildes[ix],
                                    'use_em': False}
            output_channel_params = {
                'sigma_sq': self.sigma_sqs_init[ix],
                'noise_level_estimation': 'em'}
            gamp_config = {'tolerance': self.tolerance,
                           'iterations': self.iterations,
                           'input_channel_parameters': input_channel_params,
                           'output_channel_parameters': output_channel_params}
            magni.cs.reconstruction.gamp.config.update(gamp_config)

            # Generate problem instance
            z, A, alpha = use_gaussian(
                self.ns[ix], self.ms[ix], self.ks[ix], self.seeds[ix])
            A_asq = np.abs(A)**2
            if self.sigma_sqs[ix] > 0:
                y = z + np.random.normal(
                    size=z.shape, loc=0.0, scale=np.sqrt(self.sigma_sqs[ix]))
            else:
                y = z

            # Run solver
            alpha_hat = magni.cs.reconstruction.gamp.run(y, A, A_asq)

            # Compare result
            # The difference in EM learning for AMP vs Symmetric GAMP with AWGN
            # output channel makes it difficult to compare the results.
            # Thus, we only compare the non-zeros up to atol=1e-7.
            self.assertTrue(
                np.allclose(
                    self.comparison_solutions_em_sigma_sq[ix][:self.ks[ix]],
                    alpha_hat.flatten()[:self.ks[ix]],
                    atol=1e-7))


class FastOpsTests(unittest.TestCase):
    """
    Tests of FastOp input, i.e. function based measurements and FFT dictionary.

    The following tests are implemented:

    - *test_AMP_with_DCT_FFT_vs_Separable_2D*
    - *test_GAMP_with_DCT_FFT_vs_Separable_2D_rangan_sum_approx*
    - *test_GAMP_with_DCT_FFT_vs_Separable_2D_krzakala_sum_approx*
    - *test_GAMP_with_DCT_Separable_full_transform_and_precision*
    - *test_IT_with_DCT*
    - *test_IT_with_DFT*
    - *test_IT_with_DCT_and_precision*

    """

    def setUp(self):
        h = 25
        w = 25
        n = h * w
        k = 15

        self.problem_dim = (h, w)

        # Spiral scan pattern
        scan_length = 0.30 * 2 * h * w
        num_points = 10 * int(scan_length)
        img_coords = magni.imaging.measurements.spiral_sample_image(
            h, w, scan_length, num_points, rect_area=True)
        self.Phi = magni.imaging.measurements.construct_measurement_matrix(
            img_coords, h, w)

        np.random.seed(6021)
        self.alpha_real = np.zeros((n, 1))
        self.alpha_real[:k, 0] = np.random.normal(size=k, loc=2, scale=2.0)
        self.alpha_complex = np.zeros((n, 1), dtype=np.complex128)
        self.alpha_complex[:k, 0] = (np.random.randn(k) +
                                     1j * np.random.randn(k))

        self.noise = np.random.normal(size=(self.Phi.shape[0], 1), scale=0.01)

        magni.cs.reconstruction.it.config.update(
            {'iterations': 200, 'threshold': 'fixed', 'threshold_fixed': k})
        magni.cs.reconstruction.gamp.config.update(
            {'iterations': 500,
             'tolerance': 1e-6,
             'input_channel_parameters': {'tau': k/n,
                                          'theta_bar': 2.0,
                                          'theta_tilde': 4.0,
                                          'use_em': False},
             'output_channel_parameters': {'sigma_sq': 1.0,
                                           'noise_level_estimation': 'median'}}
        )

    def tearDown(self):
        magni.cs.reconstruction.it.config.reset()
        magni.cs.reconstruction.amp.config.reset()
        magni.cs.reconstruction.gamp.config.reset()

    def test_AMP_with_DCT_FFT_vs_Separable_2D(self):
        Psi_fft = magni.imaging.dictionaries.get_DCT(self.problem_dim)
        A_fft = magni.utils.matrices.MatrixCollection((self.Phi, Psi_fft))
        y_fft = A_fft.dot(self.alpha_real) + self.noise

        iDCT_mtx = magni.imaging.dictionaries.get_DCT_transform_matrix(
            self.problem_dim[0]).T
        Psi_sep = magni.utils.matrices.Separable2DTransform(iDCT_mtx, iDCT_mtx)
        A_sep = magni.utils.matrices.MatrixCollection((self.Phi, Psi_sep))
        y_sep = A_sep.dot(self.alpha_real) + self.noise

        self.assertTrue(np.allclose(y_fft, y_sep))
        alpha_hat_fft = self._amp_run(y_fft, A_fft, self.alpha_real,
                                      success=True)
        alpha_hat_sep = self._amp_run(y_sep, A_sep, self.alpha_real,
                                      success=True)
        self.assertTrue(np.allclose(alpha_hat_fft, alpha_hat_sep))

    def test_GAMP_with_DCT_FFT_vs_Separable_2D_rangan_sum_approx(self):
        Psi_fft = magni.imaging.dictionaries.get_DCT(self.problem_dim)
        A_fft = magni.utils.matrices.MatrixCollection((self.Phi, Psi_fft))
        y_fft = A_fft.dot(self.alpha_real) + self.noise

        iDCT_mtx = magni.imaging.dictionaries.get_DCT_transform_matrix(
            self.problem_dim[0]).T
        Psi_sep = magni.utils.matrices.Separable2DTransform(iDCT_mtx, iDCT_mtx)
        A_sep = magni.utils.matrices.MatrixCollection((self.Phi, Psi_sep))
        y_sep = A_sep.dot(self.alpha_real) + self.noise

        self.assertEqual(
            magni.cs.reconstruction.gamp.config['sum_approximation_constant'],
            {'rangan': 1.0})
        self.assertTrue(np.allclose(y_fft, y_sep))
        alpha_hat_fft = self._gamp_run(
            y_fft, A_fft, None, self.alpha_real, success=True)
        alpha_hat_sep = self._gamp_run(
            y_sep, A_sep, None, self.alpha_real, success=True)
        self.assertTrue(np.allclose(alpha_hat_fft, alpha_hat_sep))

    def test_GAMP_with_DCT_FFT_vs_Separable_2D_krzakala_sum_approx(self):
        magni.cs.reconstruction.gamp.config['sum_approximation_constant'] = {
            'krzakala': 1.0 / (self.problem_dim[0] * self.problem_dim[1])}
        Psi_fft = magni.imaging.dictionaries.get_DCT(self.problem_dim)
        A_fft = magni.utils.matrices.MatrixCollection((self.Phi, Psi_fft))
        y_fft = A_fft.dot(self.alpha_real) + self.noise

        iDCT_mtx = magni.imaging.dictionaries.get_DCT_transform_matrix(
            self.problem_dim[0]).T
        Psi_sep = magni.utils.matrices.Separable2DTransform(iDCT_mtx, iDCT_mtx)
        A_sep = magni.utils.matrices.MatrixCollection((self.Phi, Psi_sep))
        y_sep = A_sep.dot(self.alpha_real) + self.noise

        self.assertEqual(
            magni.cs.reconstruction.gamp.config['sum_approximation_constant'],
            {'krzakala': 1.0 / (self.problem_dim[0] * self.problem_dim[1])})
        self.assertTrue(np.allclose(y_fft, y_sep))
        alpha_hat_fft = self._gamp_run(
            y_fft, A_fft, None, self.alpha_real, success=True)
        alpha_hat_sep = self._gamp_run(
            y_sep, A_sep, None, self.alpha_real, success=True)
        self.assertTrue(np.allclose(alpha_hat_fft, alpha_hat_sep))

    def test_GAMP_with_DCT_Separable_full_transform_and_precision(self):
        # Float 32
        magni.cs.reconstruction.gamp.config['precision_float'] = np.float32
        self.assertEqual(
            magni.cs.reconstruction.gamp.config['precision_float'], np.float32)
        iDCT_mtx = np.float32(
            magni.imaging.dictionaries.get_DCT_transform_matrix(
                self.problem_dim[0]).T)
        iDCT_mtx_asq = np.abs(iDCT_mtx) ** 2
        Psi = magni.utils.matrices.Separable2DTransform(iDCT_mtx, iDCT_mtx)
        Psi_asq = magni.utils.matrices.Separable2DTransform(iDCT_mtx_asq,
                                                            iDCT_mtx_asq)
        A = magni.utils.matrices.MatrixCollection((self.Phi, Psi))
        A_asq = magni.utils.matrices.MatrixCollection((self.Phi, Psi_asq))
        y = A.dot(np.float32(self.alpha_real)) + np.float32(self.noise)

        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(A.T.dot(y).dtype, np.float32)
        self.assertEqual(A_asq.T.dot(y).dtype, np.float32)
        self.assertEqual(A_asq.A.dtype, (A.A**2).dtype)
        self.assertTrue(np.allclose(A_asq.A, A.A**2))
        alpha_hat = self._gamp_run(y, A, A_asq, self.alpha_real)
        self.assertEqual(alpha_hat.dtype, np.float32)

        # Float64
        magni.cs.reconstruction.gamp.config['precision_float'] = np.float64
        self.assertEqual(
            magni.cs.reconstruction.gamp.config['precision_float'], np.float64)
        iDCT_mtx = np.float64(
            magni.imaging.dictionaries.get_DCT_transform_matrix(
                self.problem_dim[0]).T)
        iDCT_mtx_asq = np.abs(iDCT_mtx) ** 2
        Psi = magni.utils.matrices.Separable2DTransform(iDCT_mtx, iDCT_mtx)
        Psi_asq = magni.utils.matrices.Separable2DTransform(iDCT_mtx_asq,
                                                            iDCT_mtx_asq)
        A = magni.utils.matrices.MatrixCollection((self.Phi, Psi))
        A_asq = magni.utils.matrices.MatrixCollection((self.Phi, Psi_asq))
        y = A.dot(np.float64(self.alpha_real)) + np.float64(self.noise)

        self.assertEqual(y.dtype, np.float64)
        self.assertEqual(A.T.dot(y).dtype, np.float64)
        self.assertEqual(A_asq.T.dot(y).dtype, np.float64)
        self.assertEqual(A_asq.A.dtype, (A.A**2).dtype)
        self.assertTrue(np.allclose(A_asq.A, A.A**2))
        alpha_hat = self._gamp_run(y, A, A_asq, self.alpha_real)
        self.assertEqual(alpha_hat.dtype, np.float64)

        # Float128
        if not hasattr(np, 'float128'):
            return

        magni.cs.reconstruction.gamp.config['precision_float'] = np.float128
        self.assertEqual(
            magni.cs.reconstruction.gamp.config['precision_float'],
            np.float128)
        iDCT_mtx = np.float128(
            magni.imaging.dictionaries.get_DCT_transform_matrix(
                self.problem_dim[0]).T)
        iDCT_mtx_asq = np.abs(iDCT_mtx) ** 2
        Psi = magni.utils.matrices.Separable2DTransform(iDCT_mtx, iDCT_mtx)
        Psi_asq = magni.utils.matrices.Separable2DTransform(iDCT_mtx_asq,
                                                            iDCT_mtx_asq)
        A = magni.utils.matrices.MatrixCollection((self.Phi, Psi))
        A_asq = magni.utils.matrices.MatrixCollection((self.Phi, Psi_asq))
        y = A.dot(np.float128(self.alpha_real)) + np.float64(self.noise)

        self.assertEqual(y.dtype, np.float128)
        self.assertEqual(A.T.dot(y).dtype, np.float128)
        self.assertEqual(A_asq.T.dot(y).dtype, np.float128)
        self.assertEqual(A_asq.A.dtype, (A.A**2).dtype)
        self.assertTrue(np.allclose(A_asq.A, A.A**2))
        alpha_hat = self._gamp_run(y, A, A_asq, self.alpha_real)
        self.assertEqual(alpha_hat.dtype, np.float128)

    def test_IT_with_DCT_and_precision(self):
        Psi = magni.imaging.dictionaries.get_DCT(self.problem_dim)
        A = magni.utils.matrices.MatrixCollection((self.Phi, Psi))

        # Float 32
        magni.cs.reconstruction.it.config.update(
            {'precision_float': np.float32})
        self.assertEqual(
            magni.cs.reconstruction.it.config['precision_float'], np.float32)
        y = A.dot(np.float32(self.alpha_real))
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(A.T.dot(y).dtype, np.float32)
        alpha_hat = self._iht_run(y, A, self.alpha_real)
        self.assertEqual(alpha_hat.dtype, np.float32)
        alpha_hat = self._ist_run(y, A, self.alpha_real)
        self.assertEqual(alpha_hat.dtype, np.float32)

        # Float 64
        magni.cs.reconstruction.it.config.update(
            {'precision_float': np.float64})
        self.assertEqual(
            magni.cs.reconstruction.it.config['precision_float'], np.float64)
        y = A.dot(np.float64(self.alpha_real))
        self.assertEqual(y.dtype, np.float64)
        self.assertEqual(A.T.dot(y).dtype, np.float64)
        alpha_hat = self._iht_run(y, A, self.alpha_real)
        self.assertEqual(alpha_hat.dtype, np.float64)
        alpha_hat = self._ist_run(y, A, self.alpha_real)
        self.assertEqual(alpha_hat.dtype, np.float64)

        # Scipy DCT does not support float128

    def test_IT_with_DCT(self):
        Psi = magni.imaging.dictionaries.get_DCT(self.problem_dim)
        A = magni.utils.matrices.MatrixCollection((self.Phi, Psi))
        y = A.dot(self.alpha_real)

        self._iht_run(y, A, self.alpha_real)
        self._ist_run(y, A, self.alpha_real)

    def test_IT_with_DFT(self):
        Psi = magni.imaging.dictionaries.get_DFT(self.problem_dim)
        A = magni.utils.matrices.MatrixCollection((self.Phi, Psi))
        y_real = A.dot(self.alpha_real)
        y_complex = A.dot(self.alpha_complex)

        self._iht_run(y_real, A, self.alpha_real)
        self._ist_run(y_real, A, self.alpha_real)
        self._iht_run(y_complex, A, self.alpha_complex)
        self._ist_run(y_complex, A, self.alpha_complex)

    def _amp_run(self, y, A, a, success=True):
        threshold_params = {
            'theta': magni.cs.reconstruction.amp.util.theta_mm(
                float(A.shape[0]) / A.shape[1]), 'tau_hat_sq': 1.0,
            'threshold_level_update_method': 'residual'}
        magni.cs.reconstruction.amp.config['threshold_parameters'].update(
            threshold_params)
        a_hat = magni.cs.reconstruction.amp.run(y, A)

        if success:
            self.assertTrue(np.allclose(a_hat, a, atol=1e-1))
        else:
            self.assertFalse(np.allclose(a_hat, a, atol=1e-1))

        return a_hat

    def _gamp_run(self, y, F, F_sq, a, success=True):
        a_hat = magni.cs.reconstruction.gamp.run(y, F, F_sq)
        if success:
            self.assertTrue(np.allclose(a_hat, a, atol=1e-1))
        else:
            self.assertFalse(np.allclose(a_hat, a, atol=1e-1))

        return a_hat

    def _iht_run(self, y, A, alpha, success=True):
        iht_config = {'threshold_operator': 'hard'}
        magni.cs.reconstruction.it.config.update(iht_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'], 'hard')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)
        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))

        return alpha_hat

    def _ist_run(self, y, A, alpha, success=True):
        ist_config = {'threshold_operator': 'soft'}
        magni.cs.reconstruction.it.config.update(ist_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'], 'soft')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)

        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))

        return alpha_hat


class FeatureTest(object):
    """
    Reconstruction algorithm feature test base class.

    This class defines a reconstruction problem which may be used as the base
    for testing features of reconstruction algorithms such as warm start or
    different stop criteria.

    See the individual feature test classes for further information.

    """

    def setUp(self):
        seed = 6021
        n = 500
        delta = 0.68
        rho = 0.17
        m = int(delta * n)

        self.k = int(rho * m)
        self.tau = delta * rho

        self.y, self.A, self.alpha = use_rademacher(n, m, self.k, seed=seed)
        self.oracle_support = self.alpha != 0

        self.z, self.F, self.a = use_gaussian(n, m, self.k, seed=seed)
        self.F_sq = self.F**2

        magni.cs.reconstruction.it.config.update(iterations=200)
        magni.cs.reconstruction.gamp.config.update(iterations=200)

    def tearDown(self):
        magni.cs.reconstruction.it.config.reset()
        magni.cs.reconstruction.amp.config.reset()
        magni.cs.reconstruction.gamp.config.reset()

    def _amp_run(self, y, A, a, success=True):
        threshold_params = {
            'theta': magni.cs.reconstruction.amp.util.theta_mm(
                float(A.shape[0]) / A.shape[1]), 'tau_hat_sq': 1.0,
            'threshold_level_update_method': 'residual'}
        magni.cs.reconstruction.amp.config['threshold_parameters'].update(
            threshold_params)
        a_hat = magni.cs.reconstruction.amp.run(y, A)

        if success:
            self.assertTrue(np.allclose(a_hat, a, atol=1e-2))
        else:
            self.assertFalse(np.allclose(a_hat, a, atol=1e-2))

        return a_hat

    def _amp_history_run(self, y, A, a, success=True):
        threshold_params = {
            'theta': magni.cs.reconstruction.amp.util.theta_mm(
                float(A.shape[0]) / A.shape[1]), 'tau_hat_sq': 1.0,
            'threshold_level_update_method': 'residual'}
        magni.cs.reconstruction.amp.config['threshold_parameters'].update(
            threshold_params)
        a_hat, history = magni.cs.reconstruction.amp.run(y, A)

        if success:
            self.assertTrue(np.allclose(a_hat, a, atol=1e-2))
        else:
            self.assertFalse(np.allclose(a_hat, a, atol=1e-2))

        return history

    def _gamp_run(self, z, F, F_sq, a, success=True):
        a_hat = magni.cs.reconstruction.gamp.run(z, F, F_sq)

        if success:
            self.assertTrue(np.allclose(a_hat, a, atol=1e-2))
        else:
            self.assertFalse(np.allclose(a_hat, a, atol=1e-2))

        return a_hat

    def _gamp_history_run(self, z, F, F_sq, a, success=True):
        a_hat, history = magni.cs.reconstruction.gamp.run(z, F, F_sq)

        if success:
            self.assertTrue(np.allclose(a_hat, a, atol=1e-2))
        else:
            self.assertFalse(np.allclose(a_hat, a, atol=1e-2))

        return history

    def _iht_run(self, y, A, alpha, success=True):
        iht_config = {'threshold_operator': 'hard'}
        magni.cs.reconstruction.it.config.update(iht_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'], 'hard')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)

        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))

        return alpha_hat

    def _ist_run(self, y, A, alpha, success=True):
        ist_config = {'threshold_operator': 'soft'}
        magni.cs.reconstruction.it.config.update(ist_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'], 'soft')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)

        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))

        return alpha_hat

    def _ist_history_run(self, y, A, alpha, success=True):
        ist_config = {'threshold_operator': 'soft'}
        magni.cs.reconstruction.it.config.update(ist_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'], 'soft')

        alpha_hat, history = magni.cs.reconstruction.it.run(y, A)

        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))

        return history


class FeaturePrecisionFloatTest(FeatureTest, unittest.TestCase):
    """
    Test of the precision float feature in reconstruction algorithms.

    The following tests are implemented:

    - *test_float32_AMP*
    - *test_float32_GAMP*
    - *test_float64_AMP*
    - *test_float64_GAMP*
    - *test_float128_AMP*
    - *test_float128_GAMP*
    - *test_float32_IST*
    - *test_float64_IST*
    - *test_float128_IST*

    """

    def test_float32_AMP(self, success=True):
        magni.cs.reconstruction.amp.config['precision_float'] = np.float32
        self.y = np.float32(self.y)
        self.A = np.float32(self.A)
        self.assertEqual(
            magni.cs.reconstruction.amp.config['precision_float'], np.float32)

        a_hat = self._amp_run(self.y, self.A, self.alpha, success=success)
        self.assertEqual(a_hat.dtype, np.float32)

    def test_float32_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        sc = magni.cs.reconstruction.gamp.stop_criterion
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'stop_criterion': sc.Residual,
             'precision_float': np.float32})
        self.z = np.float32(self.z)
        self.F = np.float32(self.F)
        self.F_sq = np.float32(self.F_sq)
        self.assertEqual(
            magni.cs.reconstruction.gamp.config['precision_float'], np.float32)

        a_hat = self._gamp_run(
            self.z, self.F, self.F_sq, self.a, success=success)
        self.assertEqual(a_hat.dtype, np.float32)

    def test_float32_GAMP_EM(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': True}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'em'}
        sc = magni.cs.reconstruction.gamp.stop_criterion
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'stop_criterion': sc.Residual,
             'precision_float': np.float32})
        self.z = np.float32(self.z)
        self.F = np.float32(self.F)
        self.F_sq = np.float32(self.F_sq)
        self.assertEqual(
            magni.cs.reconstruction.gamp.config['precision_float'], np.float32)

        a_hat = self._gamp_run(
            self.z, self.F, self.F_sq, self.a, success=success)
        self.assertEqual(a_hat.dtype, np.float32)

    def test_float64_AMP(self, success=True):
        magni.cs.reconstruction.amp.config['precision_float'] = np.float64
        self.y = np.float64(self.y)
        self.A = np.float64(self.A)
        self.assertEqual(
            magni.cs.reconstruction.amp.config['precision_float'], np.float64)

        a_hat = self._amp_run(self.y, self.A, self.alpha, success=success)
        self.assertEqual(a_hat.dtype, np.float64)

    def test_float64_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        sc = magni.cs.reconstruction.gamp.stop_criterion
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'stop_criterion': sc.Residual,
             'precision_float': np.float64})
        self.z = np.float64(self.z)
        self.F = np.float64(self.F)
        self.F_sq = np.float64(self.F_sq)
        self.assertEqual(
            magni.cs.reconstruction.gamp.config['precision_float'], np.float64)

        a_hat = self._gamp_run(
            self.z, self.F, self.F_sq, self.a, success=success)
        self.assertEqual(a_hat.dtype, np.float64)

    def test_float64_GAMP_EM(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': True}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'em'}
        sc = magni.cs.reconstruction.gamp.stop_criterion
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'stop_criterion': sc.Residual,
             'precision_float': np.float64})
        self.z = np.float64(self.z)
        self.F = np.float64(self.F)
        self.F_sq = np.float64(self.F_sq)
        self.assertEqual(
            magni.cs.reconstruction.gamp.config['precision_float'], np.float64)

        a_hat = self._gamp_run(
            self.z, self.F, self.F_sq, self.a, success=success)
        self.assertEqual(a_hat.dtype, np.float64)

    @unittest.skipIf(not hasattr(np, 'float128'), 'precision is not available')
    def test_float128_AMP(self, success=True):
        magni.cs.reconstruction.amp.config['precision_float'] = np.float128
        self.y = np.float128(self.y)
        self.A = np.float128(self.A)
        self.assertEqual(
            magni.cs.reconstruction.amp.config['precision_float'], np.float128)

        a_hat = self._amp_run(self.y, self.A, self.alpha, success=success)
        self.assertEqual(a_hat.dtype, np.float128)

    @unittest.skipIf(not hasattr(np, 'float128'), 'precision is not available')
    def test_float128_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        sc = magni.cs.reconstruction.gamp.stop_criterion
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'stop_criterion': sc.Residual,
             'precision_float': np.float128})
        self.z = np.float128(self.z)
        self.F = np.float128(self.F)
        self.F_sq = np.float128(self.F_sq)
        self.assertEqual(
            magni.cs.reconstruction.gamp.config['precision_float'],
            np.float128)

        a_hat = self._gamp_run(
            self.z, self.F, self.F_sq, self.a, success=success)
        self.assertEqual(a_hat.dtype, np.float128)

    @unittest.skipIf(not hasattr(np, 'float128'), 'precision is not available')
    def test_float128_GAMP_EM(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': True}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'em'}
        sc = magni.cs.reconstruction.gamp.stop_criterion
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'stop_criterion': sc.Residual,
             'precision_float': np.float128})
        self.z = np.float128(self.z)
        self.F = np.float128(self.F)
        self.F_sq = np.float128(self.F_sq)
        self.assertEqual(
            magni.cs.reconstruction.gamp.config['precision_float'],
            np.float128)

        a_hat = self._gamp_run(
            self.z, self.F, self.F_sq, self.a, success=success)
        self.assertEqual(a_hat.dtype, np.float128)

    def test_float32_IST(self, success=True):
        magni.cs.reconstruction.it.config.update(
            {'precision_float': np.float32})
        self.A = np.float32(self.A)
        self.y = np.float32(self.y)
        self.assertEqual(
            magni.cs.reconstruction.it.config['precision_float'], np.float32)

        alpha_hat = self._ist_run(self.y, self.A, self.alpha, success=success)
        self.assertEqual(alpha_hat.dtype, np.float32)

    def test_float64_IST(self, success=True):
        magni.cs.reconstruction.it.config.update(
            {'precision_float': np.float64})
        self.A = np.float64(self.A)
        self.y = np.float64(self.y)
        self.assertEqual(
            magni.cs.reconstruction.it.config['precision_float'], np.float64)

        alpha_hat = self._ist_run(self.y, self.A, self.alpha, success=success)
        self.assertEqual(alpha_hat.dtype, np.float64)

    @unittest.skipIf(not hasattr(np, 'float128'), 'precision is not available')
    def test_float128_IST(self, success=True):
        magni.cs.reconstruction.it.config.update(
            {'precision_float': np.float128})
        self.A = np.float128(self.A)
        self.y = np.float128(self.y)
        self.assertEqual(
            magni.cs.reconstruction.it.config['precision_float'], np.float128)

        alpha_hat = self._ist_run(self.y, self.A, self.alpha, success=success)
        self.assertEqual(alpha_hat.dtype, np.float128)


class FeatureReportHistoryTest(FeatureTest, unittest.TestCase):
    """
    Test the report history feature in reconstruction algorithms.

    The following tests are implemented:

    - *test_MSE_CONVERGENCE_AMP* (stop based on MSE)
    - *test_MAX_INTERATIONS_AMP* (stop based on max iterations)
    - *test_MSE_CONVERGENCE_GAMP* (stop based on MSE)
    - *test_MAX_INTERATIONS_GAMP* (stop based on max iterations)
    - *test_MSE_CONVERGENCE_IST* (stop based on MSE)
    - *test_MAX_INTERATIONS_IST* (stop based on max iterations)

    """

    def test_MSE_CONVERGENCE_AMP(self, success=True):
        magni.cs.reconstruction.amp.config.update(
             {'report_history': True, 'true_solution': self.alpha})

        history = self._amp_history_run(self.y, self.A, self.alpha,
                                        success=success)

        self.assertEqual(history['stop_criterion'], 'MSECONVERGENCE')
        self.assertEqual(history['stop_reason'], 'MSECONVERGENCE')
        self.assertEqual(history['stop_iteration'], 30)
        self.assertEqual(len(history['MSE']), 32)
        self.assertEqual(len(history['threshold_parameters']), 32)
        self.assertEqual(len(history['alpha_bar']), 32)
        self.assertEqual(len(history['stop_criterion_value']), 32)

    def test_MAX_ITERATION_AMP(self, success=False):
        magni.cs.reconstruction.amp.config.update(
             {'report_history': True, 'iterations': 8})

        history = self._amp_history_run(self.y, self.A, self.alpha,
                                        success=success)

        self.assertEqual(history['stop_criterion'], 'MSECONVERGENCE')
        self.assertEqual(history['stop_reason'], 'MAX_ITERATIONS')
        self.assertEqual(history['stop_iteration'], 7)
        self.assertEqual(len(history['MSE']), 1)
        self.assertEqual(len(history['threshold_parameters']), 9)
        self.assertEqual(len(history['alpha_bar']), 9)
        self.assertEqual(len(history['stop_criterion_value']), 9)

    def test_MSE_CONVERGENCE_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'report_history': True,
             'true_solution': self.a})

        history = self._gamp_history_run(
            self.z, self.F, self.F_sq, self.a, success=success)

        self.assertEqual(history['stop_criterion'], 'MSECONVERGENCE')
        self.assertEqual(history['stop_reason'], 'MSECONVERGENCE')
        self.assertEqual(history['stop_iteration'], 10)
        self.assertEqual(len(history['MSE']), 12)
        self.assertEqual(len(history['input_channel_parameters']), 12)
        self.assertEqual(len(history['output_channel_parameters']), 12)
        self.assertEqual(len(history['alpha_bar']), 12)
        self.assertEqual(len(history['alpha_tilde']), 12)
        self.assertEqual(len(history['stop_criterion_value']), 12)

    def test_MAX_ITERATIONS_GAMP(self, success=False):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'report_history': True,
             'iterations': 8})

        history = self._gamp_history_run(
            self.z, self.F, self.F_sq, self.a, success=success)

        self.assertEqual(history['stop_criterion'], 'MSECONVERGENCE')
        self.assertEqual(history['stop_reason'], 'MAX_ITERATIONS')
        self.assertEqual(history['stop_iteration'], 7)
        self.assertEqual(len(history['MSE']), 1)
        self.assertEqual(len(history['output_channel_parameters']), 9)
        self.assertEqual(len(history['output_channel_parameters']), 9)
        self.assertEqual(len(history['alpha_bar']), 9)
        self.assertEqual(len(history['alpha_tilde']), 9)
        self.assertEqual(len(history['stop_criterion_value']), 9)

    def test_MSE_CONVERGENCE_IST(self, success=True):
        magni.cs.reconstruction.it.config.update(
            {'report_history': True,
             'stop_criterion': 'mse_convergence',
             'true_solution': self.alpha})

        history = self._ist_history_run(
            self.y, self.A, self.alpha, success=False)

        self.assertEqual(history['stop_criterion'], 'MSE_CONVERGENCE')
        self.assertEqual(history['stop_reason'], 'MSE_CONVERGENCE')
        self.assertEqual(history['stop_iteration'], 3)
        self.assertEqual(len(history['MSE']), 5)
        self.assertEqual(len(history['alpha']), 5)
        self.assertEqual(len(history['stop_criterion_value']), 5)

    def test_MAX_ITERATIONS_IST(self, success=False):
        magni.cs.reconstruction.it.config.update(
            {'report_history': True,
             'stop_criterion': 'mse_convergence',
             'iterations': 2})

        history = self._ist_history_run(
            self.y, self.A, self.alpha, success=False)

        self.assertEqual(history['stop_criterion'], 'MSE_CONVERGENCE')
        self.assertEqual(history['stop_reason'], 'MAX_ITERATIONS')
        self.assertEqual(history['stop_iteration'], 1)
        self.assertEqual(len(history['MSE']), 1)
        self.assertEqual(len(history['alpha']), 3)
        self.assertEqual(len(history['stop_criterion_value']), 3)


class FeatureStopCriterionTest(FeatureTest, unittest.TestCase):
    """
    Test of the stop criterion feature in reconstruction algorithms.

    The following tests are implemented:

    - *test_AMP_stop_criterion_error_handling
    - *test_residual_AMP* (stop based on residual)
    - *test_residual_measurements_ratio_AMP* (stop based on ratio of
       measurements to residual)
    - *test_normalised_MSE_convergence_AMP* (stop based on NMSE)
    - *test_GAMP_stop_criterion_error_handling
    - *test_residual_GAMP* (stop based on residual)
    - *test_residual_measurements_ratio_GAMP* (stop based on ratio of
       measurements to residual)
    - *test_normalised_MSE_convergence_GAMP* (stop based on NMSE)
    - *test_residual_IST* (stop based on residual)
    - *test_residual_measurements_ratio_IST* (stop based on ratio of
       measurements to residual)

    """

    def test_AMP_stop_criterion_error_handling(self):
        sc = magni.cs.reconstruction.amp.stop_criterion
        with self.assertRaises(TypeError):
            sc.ValidatedStopCriterion('fail')
        with self.assertRaises(TypeError):
            sc.ValidatedStopCriterion({})('fail')
        with self.assertRaises(TypeError):
            sc.NormalisedMSEConvergence('fail')
        with self.assertRaises(TypeError):
            sc.NormalisedMSEConvergence({'tolerance': 1e-3})('fail')

    def test_residual_AMP(self, success=True):
        sc = magni.cs.reconstruction.amp.stop_criterion
        magni.cs.reconstruction.amp.config.update(
            {'stop_criterion': sc.Residual})

        self._amp_run(self.y, self.A, self.alpha, success=success)

    def test_residual_measurements_ratio_AMP(self, success=True):
        sc = magni.cs.reconstruction.amp.stop_criterion
        magni.cs.reconstruction.amp.config.update(
            {'stop_criterion': sc.ResidualMeasurementsRatio})

        self._amp_run(self.y, self.A, self.alpha, success=success)

    def test_normalised_MSE_convergence_AMP(self, success=True):
        sc = magni.cs.reconstruction.amp.stop_criterion
        magni.cs.reconstruction.amp.config.update(
            {'stop_criterion': sc.NormalisedMSEConvergence})

        self._amp_run(self.y, self.A, self.alpha, success=success)

    def test_GAMP_stop_criterion_error_handling(self):
        sc = magni.cs.reconstruction.gamp.stop_criterion
        with self.assertRaises(TypeError):
            sc.ValidatedStopCriterion('fail')
        with self.assertRaises(TypeError):
            sc.ValidatedStopCriterion({})('fail')
        with self.assertRaises(TypeError):
            sc.NormalisedMSEConvergence('fail')
        with self.assertRaises(TypeError):
            sc.NormalisedMSEConvergence({'tolerance': 1e-3})('fail')

    def test_residual_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        sc = magni.cs.reconstruction.gamp.stop_criterion
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'stop_criterion': sc.Residual})

        self._gamp_run(self.z, self.F, self.F_sq, self.a, success=success)

    def test_residual_measurements_ratio_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        sc = magni.cs.reconstruction.gamp.stop_criterion
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'stop_criterion': sc.ResidualMeasurementsRatio})

        self._gamp_run(self.z, self.F, self.F_sq, self.a, success=success)

    def test_normalised_MSE_convergence_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        sc = magni.cs.reconstruction.gamp.stop_criterion
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'stop_criterion': sc.NormalisedMSEConvergence})

        self._gamp_run(self.z, self.F, self.F_sq, self.a, success=success)

    def test_residual_IST(self, success=True):
        magni.cs.reconstruction.it.config.update(
            {'stop_criterion': 'residual'})

        self._ist_run(self.y, self.A, self.alpha, success=False)

    def test_residual_measurements_ratio_IST(self, success=True):
        magni.cs.reconstruction.it.config.update(
            {'stop_criterion': 'residual'})

        self._ist_run(self.y, self.A, self.alpha, success=False)


class FeatureWarmStartTest(FeatureTest, unittest.TestCase):
    """
    Test of the warm_start feature in reconstruction algorithms.

    The following tests are implemented:

    - *test_warm_start_IT* (Iterative thresholding)
    - *test_warm_start_AMP* (Approximate Message Passing)
    - *test_warm_start_GAMP* (Generalised Approximate Message Passing)

    """

    def test_warm_start_IT(self, success_iht=True, success_ist=True):
        it_config = {'warm_start': 0.1 * np.ones(self.alpha.shape)}
        magni.cs.reconstruction.it.config.update(it_config)
        self._iht_run(self.y, self.A, self.alpha, success=success_iht)
        self._ist_run(self.y, self.A, self.alpha, success=success_ist)
        self.assertIsNotNone(
            magni.cs.reconstruction.it.config['warm_start'])

    def test_warm_start_AMP(self, success=True):
        magni.cs.reconstruction.amp.config.update(
            {'warm_start': 0.1 * np.ones(self.a.shape)})

        self._amp_run(self.y, self.A, self.alpha, success=success)

    def test_warm_start_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'warm_start': (0.1 * np.ones(self.a.shape),
                            2 * np.ones(self.a.shape))})

        self._gamp_run(self.z, self.F, self.F_sq, self.a, success=success)


class PhaseSpaceExtremesTest(unittest.TestCase):
    """
    Tests of border case (extreme) phase space values.

    The following tests are implemented:

    - *test_basic_setup* (not extremum)
    - *test_invalid_A_and_y* (empty A and y)
    - *test_k_equals_zero*
    - *test_k_equals_m*
    - *test_m_equals_one*
    - *test_m_equals_n*
    - *test_m_and_n_equals_one*
    - *test_n_equals_one*

    """

    def setUp(self):
        self.n = 500
        self.m = 200
        self.k = 10
        self.seed = 6021

        magni.cs.reconstruction.it.config.update(iterations=200)
        magni.cs.reconstruction.gamp.config.update(iterations=200)

        # GAMP setup
        input_channel_params = {'tau': self.k/self.n, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 0.5,
                                 'noise_level_estimation': 'sample_variance'}
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params})

    def tearDown(self):
        magni.cs.reconstruction.it.config.reset()
        magni.cs.reconstruction.amp.config.reset()
        magni.cs.reconstruction.gamp.config.reset()

    def test_basic_setup(self):
        y, A, alpha = use_rademacher(self.n, self.m, self.k, seed=self.seed)
        A_asq = A**2

        self._iht_run(y, A, alpha)
        self._ist_run(y, A, alpha)
        self._gamp_run(y, A, A_asq, alpha)

    def test_invalid_A_and_y(self):
        A = np.array([])
        A_asq = np.array([])
        y = np.array([])
        alpha = np.array([])
        with self.assertRaises(ValueError):
            self._iht_run(y, A, alpha)

        with self.assertRaises(ValueError):
            self._ist_run(y, A, alpha)

        with self.assertRaises(ValueError):
            self._gamp_run(y, A, A_asq, alpha)

    def test_k_equals_zero(self):
        k = 0
        y, A, alpha = use_rademacher(self.n, self.m, k, seed=self.seed)
        A_asq = A**2

        self._iht_run(y, A, alpha)
        self._ist_run(y, A, alpha)
        self._gamp_run(y, A, A_asq, alpha)

    def test_k_equals_m(self):
        k = self.m
        y, A, alpha = use_rademacher(self.n, self.m, k, seed=self.seed)
        A_asq = A**2

        self._iht_run(y, A, alpha, success=False)
        self._ist_run(y, A, alpha, success=False)
        self._gamp_run(y, A, A_asq, alpha, success=False)

    def test_m_equals_one(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = 1
            y, A, alpha = use_rademacher(self.n, m, self.k, seed=self.seed)
            A_asq = A**2

            self._iht_run(y, A, alpha, success=False)
            self._ist_run(y, A, alpha, success=False)
            self._gamp_run(y, A, A_asq, alpha, success=False)

    def test_m_equals_n(self):
        m = self.n
        y, A, alpha = use_rademacher(self.n, m, self.k, seed=self.seed)
        A_asq = A**2

        self._iht_run(y, A, alpha)
        self._ist_run(y, A, alpha)
        self._gamp_run(y, A, A_asq, alpha)

    def test_m_and_n_equals_one(self):
        n = 1
        m = 1
        k = 1
        y, A, alpha = use_rademacher(n, m, k, seed=self.seed)
        A_asq = A**2

        self._iht_run(y, A, alpha, success=False)
        self._ist_run(y, A, alpha, success=False)
        self._gamp_run(y, A, A_asq, alpha, success=False)

    def test_n_equals_one(self):
        n = 1
        k = 1
        y, A, alpha = use_rademacher(n, self.m, k, seed=self.seed)
        A_asq = A**2

        self._iht_run(y, A, alpha)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._ist_run(y, A, alpha, success=False)
        self._gamp_run(y, A, A_asq, alpha)

    def _gamp_run(self, z, F, F_sq, a, success=True):
        a_hat = magni.cs.reconstruction.gamp.run(z, F, F_sq)

        if success:
            self.assertTrue(np.allclose(a_hat, a, atol=1e-2))
        else:
            self.assertFalse(np.allclose(a_hat, a, atol=1e-2))

    def _iht_run(self, y, A, alpha, success=True):
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'], 'hard')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)
        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))

    def _ist_run(self, y, A, alpha, success=True):
        ist_config = {'threshold_operator': 'soft'}
        magni.cs.reconstruction.it.config.update(ist_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'], 'soft')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)

        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))


class PhaseSpaceTest(object):
    """
    Phase space test base class.

    The following tests are implemented:

    - *test_default_IT* (default configuration)
    - *test_fixed_IT* (fixed threshold)
    - *test_adaptive_fixed_IT* (adaptive step-size, fixed threshold)
    - *test_weighted_fixed_IT* (weighted, fixed threshold)
    - *test_residual_soft_threshold_AMP* (soft threshold, residual level)
    - *test_median_soft_threshold_AMP* (soft threshold, median level)
    - *test_iidsGB_AWGN_GAMP* (s-GB)
    - *test_iidsGB_AWGN_EM_GAMP* (s-GB with EM learning)
    - *test_iidBG_AWGN_GAMP* (MMSE GAMP)
    - *test_iidBG_AWGN_EM_GAMP* (MMSE GAMP with EM learning)
    - *test_iidBG_AWGN_GAMP_rangan_sum_approx* (rangan sum approx GAMP)
    - *test_iidBG_AWGN_GAMP_krzakala_sum_approx* (krzakala sum approx GAMP)

    """

    def setUp(self, n=None, delta=None, rho=None, seed=6021):
        m = int(delta * n)

        self.k = int(rho * m)
        self.tau = delta * rho

        self.y, self.A, self.alpha = use_rademacher(n, m, self.k, seed=seed)
        self.oracle_support = self.alpha != 0

        self.z, self.F, self.a = use_gaussian(n, m, self.k, seed=seed)
        self.F_sq = self.F**2

        magni.cs.reconstruction.it.config.update(iterations=200)
        magni.cs.reconstruction.gamp.config.update(iterations=200)

    def tearDown(self):
        magni.cs.reconstruction.it.config.reset()
        magni.cs.reconstruction.amp.config.reset()
        magni.cs.reconstruction.gamp.config.reset()

    def test_residual_soft_threshold_AMP(self, success=True):
        magni.cs.reconstruction.amp.config['threshold_parameters'] = {
            'threshold_level_update_method': 'residual'}

        self._amp_run(self.y, self.A, self.alpha, success=success)

    def test_median_soft_threshold_AMP(self, success=True):
        magni.cs.reconstruction.amp.config['threshold_parameters'] = {
            'threshold_level_update_method': 'median'}

        self._amp_run(self.y, self.A, self.alpha, success=success)

    def test_iidsGB_AWGN_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        IIDsGB = magni.cs.reconstruction.gamp.input_channel.IIDsGB
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'input_channel': IIDsGB})

        self._gamp_run(self.z, self.F, self.F_sq, self.a, success=success)

    def test_iidsGB_AWGN_EM_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': True,
                                'em_damping': 0.5}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'em'}
        IIDsGB = magni.cs.reconstruction.gamp.input_channel.IIDsGB
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'input_channel': IIDsGB})

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._gamp_run(self.z, self.F, self.F_sq, self.a, success=success)

    def test_iidBG_AWGN_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params})

        self._gamp_run(self.z, self.F, self.F_sq, self.a, success=success)

    def test_iidBG_AWGN_EM_GAMP(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': True}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'em'}
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params})

        self._gamp_run(self.z, self.F, self.F_sq, self.a, success=success)

    def test_iidBG_AWGN_GAMP_rangan_sum_approx(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params})

        self.assertEqual(
            magni.cs.reconstruction.gamp.config['sum_approximation_constant'],
            {'rangan': 1.0})
        self._gamp_run(self.z, self.F, None, self.a, success=success)

    def test_iidBG_AWGN_GAMP_krzakala_sum_approx(self, success=True):
        input_channel_params = {'tau': self.tau, 'theta_bar': 0,
                                'theta_tilde': 1, 'use_em': False}
        output_channel_params = {'sigma_sq': 1,
                                 'noise_level_estimation': 'sample_variance'}
        magni.cs.reconstruction.gamp.config.update(
            {'input_channel_parameters': input_channel_params,
             'output_channel_parameters': output_channel_params,
             'sum_approximation_constant': {'krzakala': 1.0 / self.F.shape[0]}}
        )

        self.assertEqual(
            magni.cs.reconstruction.gamp.config['sum_approximation_constant'],
            {'krzakala': 1.0 / self.F.shape[0]})
        self._gamp_run(self.z, self.F, None, self.a, success=success)

    def test_default_IT(self, success_iht=True, success_ist=True):
        self._iht_run(self.y, self.A, self.alpha, success=success_iht)
        self._ist_run(self.y, self.A, self.alpha, success=success_ist)

    def test_fixed_IT(self, success_iht=True, success_ist=True):
        it_config = {'threshold': 'fixed',
                     'threshold_fixed': self.k}
        magni.cs.reconstruction.it.config.update(it_config)
        self._iht_run(self.y, self.A, self.alpha, success=success_iht)
        self._ist_run(self.y, self.A, self.alpha, success=success_ist)
        self.assertEqual(magni.cs.reconstruction.it.config['threshold'],
                         'fixed')
        self.assertEqual(magni.cs.reconstruction.it.config['threshold_fixed'],
                         self.k)

    def test_adaptive_fixed_IT(self, success_iht=True, success_ist=True):
        it_config = {'threshold': 'fixed',
                     'threshold_fixed': self.k,
                     'kappa': 'adaptive'}
        magni.cs.reconstruction.it.config.update(it_config)
        self._iht_run(self.y, self.A, self.alpha, success=success_iht)
        self._ist_run(self.y, self.A, self.alpha, success=success_ist)
        self.assertEqual(magni.cs.reconstruction.it.config['threshold'],
                         'fixed')
        self.assertEqual(magni.cs.reconstruction.it.config['threshold_fixed'],
                         self.k)
        self.assertEqual(magni.cs.reconstruction.it.config['kappa'],
                         'adaptive')

    def test_weighted_fixed_IT(self, success_iht=True, success_ist=True):
        threshold_weights = np.linspace(
            1, 0.5, self.alpha.shape[0]).reshape(-1, 1)
        it_config = {'threshold': 'fixed',
                     'threshold_fixed': self.k,
                     'threshold_weights': threshold_weights}
        magni.cs.reconstruction.it.config.update(it_config)
        self._wiht_run(self.y, self.A, self.alpha, success=success_iht)
        self._wist_run(self.y, self.A, self.alpha, success=success_ist)
        self.assertEqual(magni.cs.reconstruction.it.config['threshold'],
                         'fixed')
        self.assertEqual(magni.cs.reconstruction.it.config['threshold_fixed'],
                         self.k)
        self.assertTrue(np.allclose(
            magni.cs.reconstruction.it.config['threshold_weights'],
            threshold_weights))

    def _amp_run(self, y, A, a, success=True):
        threshold_params = {
            'theta': magni.cs.reconstruction.amp.util.theta_mm(
                float(A.shape[0]) / A.shape[1]), 'tau_hat_sq': 1.0}
        magni.cs.reconstruction.amp.config['threshold_parameters'].update(
            threshold_params)
        a_hat = magni.cs.reconstruction.amp.run(y, A)

        if success:
            self.assertTrue(np.allclose(a_hat, a, atol=1e-2))
        else:
            self.assertFalse(np.allclose(a_hat, a, atol=1e-2))

    def _gamp_run(self, z, F, F_sq, a, success=True):
        a_hat = magni.cs.reconstruction.gamp.run(z, F, F_sq)
        if success:
            self.assertTrue(np.allclose(a_hat, a, atol=1e-2))
        else:
            self.assertFalse(np.allclose(a_hat, a, atol=1e-2))

    def _iht_run(self, y, A, alpha, success=True):
        iht_config = {'threshold_operator': 'hard'}
        magni.cs.reconstruction.it.config.update(iht_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'], 'hard')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)
        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))

    def _ist_run(self, y, A, alpha, success=True):
        ist_config = {'threshold_operator': 'soft'}
        magni.cs.reconstruction.it.config.update(ist_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'], 'soft')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)

        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))

    def _wiht_run(self, y, A, alpha, success=True):
        iht_config = {'threshold_operator': 'weighted_hard'}
        magni.cs.reconstruction.it.config.update(iht_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'],
            'weighted_hard')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)
        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))

    def _wist_run(self, y, A, alpha, success=True):
        ist_config = {'threshold_operator': 'weighted_soft'}
        magni.cs.reconstruction.it.config.update(ist_config)
        self.assertEqual(
            magni.cs.reconstruction.it.config['threshold_operator'],
            'weighted_soft')

        alpha_hat = magni.cs.reconstruction.it.run(y, A)

        if success:
            self.assertTrue(np.allclose(alpha_hat, alpha, atol=1e-2))
        else:
            self.assertFalse(np.allclose(alpha_hat, alpha, atol=1e-2))


class PhaseSpaceTest1(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.08, 0.05)

    """

    def setUp(self):
        n = 500
        delta = 0.08
        rho = 0.05
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_residual_soft_threshold_AMP(self):
        PhaseSpaceTest.test_residual_soft_threshold_AMP(self, success=False)

    def test_median_soft_threshold_AMP(self, success=True):
        PhaseSpaceTest.test_median_soft_threshold_AMP(self, success=False)

    def test_iidsGB_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_GAMP(self, success=False)

    def test_iidsGB_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP(self, success=False)

    def test_iidBG_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP_rangan_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_rangan_sum_approx(
            self, success=False)

    def test_iidBG_AWGN_GAMP_krzakala_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_krzakala_sum_approx(
            self, success=False)


class PhaseSpaceTest2(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.24, 0.01)

    """

    def setUp(self):
        n = 500
        delta = 0.24
        rho = 0.01
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_residual_soft_threshold_AMP(self):
        PhaseSpaceTest.test_residual_soft_threshold_AMP(self, success=False)

    def test_median_soft_threshold_AMP(self, success=True):
        PhaseSpaceTest.test_median_soft_threshold_AMP(self, success=False)

    def test_iidsGB_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_GAMP(self, success=False)

    def test_iidsGB_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP(self, success=False)

    def test_iidBG_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP_rangan_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_rangan_sum_approx(
            self, success=False)

    def test_iidBG_AWGN_GAMP_krzakala_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_krzakala_sum_approx(
            self, success=False)


class PhaseSpaceTest3(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.38, 0.12)

    """

    def setUp(self):
        n = 500
        delta = 0.38
        rho = 0.12
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_fixed_IT(self):
        PhaseSpaceTest.test_fixed_IT(self, success_ist=False)

    def test_adaptive_fixed_IT(self):
        PhaseSpaceTest.test_adaptive_fixed_IT(self, success_ist=False)


class PhaseSpaceTest4(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.62, 0.38)

    """

    def setUp(self):
        n = 500
        delta = 0.62
        rho = 0.38
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_residual_soft_threshold_AMP(self):
        PhaseSpaceTest.test_residual_soft_threshold_AMP(self, success=False)

    def test_median_soft_threshold_AMP(self, success=True):
        PhaseSpaceTest.test_median_soft_threshold_AMP(self, success=False)

    def test_default_IT(self):
        PhaseSpaceTest.test_default_IT(self, success_iht=False,
                                       success_ist=False)

    def test_fixed_IT(self):
        PhaseSpaceTest.test_fixed_IT(self, success_iht=False,
                                     success_ist=False)

    def test_adaptive_fixed_IT(self):
        PhaseSpaceTest.test_adaptive_fixed_IT(self, success_iht=False,
                                              success_ist=False)

    def test_weighted_fixed_IT(self):
        PhaseSpaceTest.test_weighted_fixed_IT(self, success_iht=False,
                                              success_ist=False)


class PhaseSpaceTest5(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.78, 0.22)

    """

    def setUp(self):
        n = 500
        delta = 0.78
        rho = 0.22
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_fixed_IT(self):
        PhaseSpaceTest.test_fixed_IT(self, success_ist=False)

    def test_adaptive_fixed_IT(self):
        PhaseSpaceTest.test_adaptive_fixed_IT(self, success_ist=False)

    def test_weighted_fixed_IT(self):
        PhaseSpaceTest.test_weighted_fixed_IT(self, success_ist=False)


class PhaseSpaceTest6(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.84, 0.08)

    """

    def setUp(self):
        n = 500
        delta = 0.84
        rho = 0.08
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_fixed_IT(self):
        PhaseSpaceTest.test_fixed_IT(self, success_ist=False)

    def test_adaptive_fixed_IT(self):
        PhaseSpaceTest.test_adaptive_fixed_IT(self, success_ist=False)

    def test_weighted_fixed_IT(self):
        PhaseSpaceTest.test_weighted_fixed_IT(self, success_ist=False)


class PhaseSpaceTest7(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.96, 0.91)

    """

    def setUp(self):
        n = 500
        delta = 0.96
        rho = 0.91
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_residual_soft_threshold_AMP(self):
        PhaseSpaceTest.test_residual_soft_threshold_AMP(self, success=False)

    def test_median_soft_threshold_AMP(self, success=True):
        PhaseSpaceTest.test_median_soft_threshold_AMP(self, success=False)

    def test_iidsGB_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_GAMP(self, success=False)

    def test_iidsGB_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP(self, success=False)

    def test_iidBG_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP_rangan_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_rangan_sum_approx(
            self, success=False)

    def test_iidBG_AWGN_GAMP_krzakala_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_krzakala_sum_approx(
            self, success=False)

    def test_default_IT(self):
        PhaseSpaceTest.test_default_IT(self, success_iht=False,
                                       success_ist=False)

    def test_fixed_IT(self):
        PhaseSpaceTest.test_fixed_IT(self, success_iht=False,
                                     success_ist=False)

    def test_adaptive_fixed_IT(self):
        PhaseSpaceTest.test_adaptive_fixed_IT(self, success_iht=False,
                                              success_ist=False)

    def test_weighted_fixed_IT(self):
        PhaseSpaceTest.test_weighted_fixed_IT(self, success_iht=False,
                                              success_ist=False)


class PhaseSpaceTestA(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.06, 0.92)

    """

    def setUp(self):
        n = 500
        delta = 0.06
        rho = 0.92
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_residual_soft_threshold_AMP(self):
        PhaseSpaceTest.test_residual_soft_threshold_AMP(self, success=False)

    def test_median_soft_threshold_AMP(self, success=True):
        PhaseSpaceTest.test_median_soft_threshold_AMP(self, success=False)

    def test_iidsGB_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_GAMP(self, success=False)

    def test_iidsGB_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP(self, success=False)

    def test_iidBG_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP_rangan_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_rangan_sum_approx(
            self, success=False)

    def test_iidBG_AWGN_GAMP_krzakala_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_krzakala_sum_approx(
            self, success=False)

    def test_default_IT(self):
        PhaseSpaceTest.test_default_IT(self, success_iht=False,
                                       success_ist=False)

    def test_fixed_IT(self):
        PhaseSpaceTest.test_fixed_IT(self, success_iht=False,
                                     success_ist=False)

    def test_adaptive_fixed_IT(self):
        PhaseSpaceTest.test_adaptive_fixed_IT(self, success_iht=False,
                                              success_ist=False)

    def test_weighted_fixed_IT(self):
        PhaseSpaceTest.test_weighted_fixed_IT(self, success_iht=False,
                                              success_ist=False)


class PhaseSpaceTestB(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.19, 0.84)

    """

    def setUp(self):
        n = 500
        delta = 0.19
        rho = 0.84
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_residual_soft_threshold_AMP(self):
        PhaseSpaceTest.test_residual_soft_threshold_AMP(self, success=False)

    def test_median_soft_threshold_AMP(self, success=True):
        PhaseSpaceTest.test_median_soft_threshold_AMP(self, success=False)

    def test_iidsGB_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_GAMP(self, success=False)

    def test_iidsGB_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP(self, success=False)

    def test_iidBG_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP_rangan_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_rangan_sum_approx(
            self, success=False)

    def test_iidBG_AWGN_GAMP_krzakala_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_krzakala_sum_approx(
            self, success=False)

    def test_default_IT(self):
        PhaseSpaceTest.test_default_IT(self, success_iht=False,
                                       success_ist=False)

    def test_fixed_IT(self):
        PhaseSpaceTest.test_fixed_IT(self, success_iht=False,
                                     success_ist=False)

    def test_adaptive_fixed_IT(self):
        PhaseSpaceTest.test_adaptive_fixed_IT(self, success_iht=False,
                                              success_ist=False)

    def test_weighted_fixed_IT(self):
        PhaseSpaceTest.test_weighted_fixed_IT(self, success_iht=False,
                                              success_ist=False)


class PhaseSpaceTestC(PhaseSpaceTest, unittest.TestCase):
    """
    Test of reconstruction capabilities at Phase Space point:
    (delta, rho) = (0.29, 0.94)

    """

    def setUp(self):
        n = 500
        delta = 0.29
        rho = 0.94
        PhaseSpaceTest.setUp(self, n=n, delta=delta, rho=rho)

    def test_residual_soft_threshold_AMP(self):
        PhaseSpaceTest.test_residual_soft_threshold_AMP(self, success=False)

    def test_median_soft_threshold_AMP(self, success=True):
        PhaseSpaceTest.test_median_soft_threshold_AMP(self, success=False)

    def test_iidsGB_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_GAMP(self, success=False)

    def test_iidsGB_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidsGB_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP(self, success=False)

    def test_iidBG_AWGN_EM_GAMP(self):
        PhaseSpaceTest.test_iidBG_AWGN_EM_GAMP(self, success=False)

    def test_iidBG_AWGN_GAMP_rangan_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_rangan_sum_approx(
            self, success=False)

    def test_iidBG_AWGN_GAMP_krzakala_sum_approx(self):
        PhaseSpaceTest.test_iidBG_AWGN_GAMP_krzakala_sum_approx(
            self, success=False)

    def test_default_IT(self):
        PhaseSpaceTest.test_default_IT(self, success_iht=False,
                                       success_ist=False)

    def test_fixed_IT(self):
        PhaseSpaceTest.test_fixed_IT(self, success_iht=False,
                                     success_ist=False)

    def test_adaptive_fixed_IT(self):
        PhaseSpaceTest.test_adaptive_fixed_IT(self, success_iht=False,
                                              success_ist=False)

    def test_weighted_fixed_IT(self):
        PhaseSpaceTest.test_weighted_fixed_IT(self, success_iht=False,
                                              success_ist=False)


class TestUSERademacher(unittest.TestCase):
    """
    Test of the use_rademacher test fixture function.

    """

    def test_seed_6021(self):
        n = 10
        m = 6
        k = 3
        seed = 6021

        alpha_true = np.array([
            [-1], [1], [1], [0], [0], [0], [0], [0], [0], [0]])
        A_true = np.array([
            [0.3970924,  0.39094998, -0.51535881, -0.29376165,  0.80329912,
             0.2343297,  0.20381475, -0.4006275,  0.97687495, -0.02913711],
            [-0.21781685, -0.46838027, -0.39565219, -0.29879357, -0.1528902,
             -0.09484526, -0.24859693,  0.42678941, -0.17170236, -0.09260817],
            [-0.08024309, -0.24175707, -0.1299679, -0.15608146, -0.51588714,
             -0.48385891,  0.15647558, -0.54407042,  0.16007046, -0.39455782],
            [-0.11461544, -0.09242993,  0.10134369,  0.03684144,  0.24202215,
             0.22913925, -0.16115897,  0.07449874,  0.24777711, -0.20584097],
            [-0.49012155,  0.30646838,  0.27297925, -0.03009987,  0.21501576,
             -0.16483217,  0.49937075,  0.04385046,  0.26298357,  0.33893551],
            [-0.06364924,  0.68731702, -0.21930248, -0.20445363,  0.38122107,
             -0.05793133,  0.12713844, -1.14521796, -0.62776378, -0.1934683]])
        y_true = A_true.dot(alpha_true)

        y, A, alpha = use_rademacher(n, m, k, seed)

        self.assertTrue(np.alltrue(alpha_true == alpha))
        self.assertTrue(np.allclose(A_true, A))
        self.assertTrue(np.allclose(y_true, y))


class TestUSEGaussian(unittest.TestCase):
    """
    Test of the use_gaussian test fixture function.

    """

    def test_seed_6021(self):
        n = 10
        m = 6
        k = 3
        seed = 6021

        alpha_true = np.array([
            2.5616611, -0.30927792, -0.56096039, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0]).reshape(-1, 1)
        A_true = np.array([
            [0.3970924,  0.39094998, -0.51535881, -0.29376165,  0.80329912,
             0.2343297,  0.20381475, -0.4006275,  0.97687495, -0.02913711],
            [-0.21781685, -0.46838027, -0.39565219, -0.29879357, -0.1528902,
             -0.09484526, -0.24859693,  0.42678941, -0.17170236, -0.09260817],
            [-0.08024309, -0.24175707, -0.1299679, -0.15608146, -0.51588714,
             -0.48385891,  0.15647558, -0.54407042,  0.16007046, -0.39455782],
            [-0.11461544, -0.09242993,  0.10134369,  0.03684144,  0.24202215,
             0.22913925, -0.16115897,  0.07449874,  0.24777711, -0.20584097],
            [-0.49012155,  0.30646838,  0.27297925, -0.03009987,  0.21501576,
             -0.16483217,  0.49937075,  0.04385046,  0.26298357,  0.33893551],
            [-0.06364924,  0.68731702, -0.21930248, -0.20445363,  0.38122107,
             -0.05793133,  0.12713844, -1.14521796, -0.62776378, -0.1934683]])
        y_true = A_true.dot(alpha_true)

        y, A, alpha = use_gaussian(n, m, k, seed)

        self.assertTrue(np.allclose(alpha_true, alpha))
        self.assertTrue(np.allclose(A_true, A))
        self.assertTrue(np.allclose(y_true, y))


def use_gaussian(n, m, k, seed):
    """
    Prepare an instance of the USE/Gaussian problem suite

    Prepares:

    * :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` from Uniform Spherical
    Ensemble (USE).
    * :math:`\mathbf{alpha} \in \mathbb{R}^{n}` with :math:`k` non-zero entries
    drawn from the standard normal distribution.
    * :math:`\mathbf{y} = \mathbf{A}\mathbf{\alpha}`

    Parameters
    ----------
    n : int
        The problem size.
    m : int
        The number of measurements.
    k : int
        The number of non-zero coefficients.
    seed : int
        The seed used in the random number generator.

    Returns
    -------
    (y, A, alpha) : tuple
        The measurements, measurement matrix, and coefficients.

    """

    @_decorate_validation
    def validate_input():
        _numeric('n', 'integer', range_='[1;inf)')
        _numeric('m', 'integer', range_='[1;inf)')
        _numeric('k', 'integer', range_='[0;inf)')
        _numeric('seed', 'integer', range_='[0;inf)')

    @_decorate_validation
    def validate_output():
        _numeric('y', ('integer', 'floating', 'complex'), shape=(m, 1))
        _numeric('A', ('integer', 'floating', 'complex'), shape=(m, n))
        _numeric('alpha', ('integer', 'floating', 'complex'), shape=(n, 1))

    validate_input()

    np.random.seed(seed=seed)

    A = 1/np.sqrt(m) * np.random.randn(m, n)
    alpha = np.zeros((n, 1))
    alpha[:k, 0] = np.random.randn(k)
    y = A.dot(alpha)

    validate_output()

    return y, A, alpha


def use_rademacher(n, m, k, seed):
    """
    Prepare an instance of the USE/Rademacher problem suite

    Prepares:

    * :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` from Uniform Spherical
    Ensemble (USE).
    * :math:`\mathbf{alpha} \in \mathbb{R}^{n}` with :math:`k` non-zero entries
    drawn from the Rademacher distribution {1, -1}.
    * :math:`\mathbf{y} = \mathbf{A}\mathbf{\alpha}`

    Parameters
    ----------
    n : int
        The problem size.
    m : int
        The number of measurements.
    k : int
        The number of non-zero coefficients.
    seed : int
        The seed used in the random number generator.

    Returns
    -------
    (y, A, alpha) : tuple
        The measurements, measurement matrix, and coefficients.

    """

    @_decorate_validation
    def validate_input():
        _numeric('n', 'integer', range_='[1;inf)')
        _numeric('m', 'integer', range_='[1;inf)')
        _numeric('k', 'integer', range_='[0;inf)')
        _numeric('seed', 'integer', range_='[0;inf)')

    @_decorate_validation
    def validate_output():
        _numeric('y', ('integer', 'floating', 'complex'), shape=(m, 1))
        _numeric('A', ('integer', 'floating', 'complex'), shape=(m, n))
        _numeric('alpha', ('integer', 'floating', 'complex'), shape=(n, 1))

    validate_input()

    np.random.seed(seed=seed)

    A = 1/np.sqrt(m) * np.random.randn(m, n)
    alpha = np.zeros((n, 1))
    alpha[:k, 0] = np.random.randint(0, 2, k) * 2 - 1
    y = A.dot(alpha)

    validate_output()

    return y, A, alpha
