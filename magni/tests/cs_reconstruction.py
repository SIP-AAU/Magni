"""
..
    Copyright (c) 2015, Magni developers.
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

See the docstring of the FastOpsTests, PhaseSpaceTest, and
ITTestPhaseSpaceExtremes classes.

Routine listings
----------------
FastOpsTests(unittest.TestCase):
    Tests of FastOp input, i.e. function based measurements and FFT dictionary.
ITTestPhaseSpaceExtremes(unittest.TestCase):
    Tests of border case (extreme) phase space values for IT.
PhaseSpaceTest(object)
    Phase space test base class
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
import unittest

import numpy as np

import magni
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


class FastOpsTests(unittest.TestCase):
    """
    Tests of FastOp input, i.e. function based measurements and FFT dictionary.

    The following tests are implemented:

    - *test_IT_with_DCT*
    - *test_IT_with_DFT*

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
        self.alpha_real[:k, 0] = np.random.randn(k)
        self.alpha_complex = np.zeros((n, 1), dtype=np.complex128)
        self.alpha_complex[:k, 0] = (np.random.randn(k) +
                                     1j * np.random.randn(k))

        magni.cs.reconstruction.it.config.update(
            {'iterations': 200, 'threshold': 'fixed', 'threshold_fixed': k})

    def tearDown(self):
        magni.cs.reconstruction.it.config.reset()

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


class ITTestPhaseSpaceExtremes(unittest.TestCase):
    """
    Tests of border case (extreme) phase space values for IT.

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

    def tearDown(self):
        magni.cs.reconstruction.it.config.reset()

    def test_basic_setup(self):
        y, A, alpha = use_rademacher(self.n, self.m, self.k, seed=self.seed)

        self._iht_run(y, A, alpha)
        self._ist_run(y, A, alpha)

    def test_invalid_A_and_y(self):
        A = np.array([])
        y = np.array([])
        with self.assertRaises(ValueError):
            alpha_hat = magni.cs.reconstruction.it.run(y, A)

    def test_k_equals_zero(self):
        k = 0
        y, A, alpha = use_rademacher(self.n, self.m, k, seed=self.seed)

        self._iht_run(y, A, alpha)
        self._ist_run(y, A, alpha)

    def test_k_equals_m(self):
        k = self.m
        y, A, alpha = use_rademacher(self.n, self.m, k, seed=self.seed)

        self._iht_run(y, A, alpha, success=False)
        self._ist_run(y, A, alpha, success=False)

    def test_m_equals_one(self):
        m = 1
        y, A, alpha = use_rademacher(self.n, m, self.k, seed=self.seed)

        self._iht_run(y, A, alpha, success=False)
        self._ist_run(y, A, alpha, success=False)

    def test_m_equals_n(self):
        m = self.n
        y, A, alpha = use_rademacher(self.n, m, self.k, seed=self.seed)

        self._iht_run(y, A, alpha)
        self._ist_run(y, A, alpha)

    def test_m_and_n_equals_one(self):
        n = 1
        m = 1
        k = 1
        y, A, alpha = use_rademacher(n, m, k, seed=self.seed)

        self._iht_run(y, A, alpha, success=False)
        self._ist_run(y, A, alpha, success=False)

    def test_n_equals_one(self):
        n = 1
        k = 1
        y, A, alpha = use_rademacher(n, self.m, k, seed=self.seed)

        self._iht_run(y, A, alpha)
        self._ist_run(y, A, alpha, success=False)

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
    - *test_warm_start_IT* (warm start)

    """

    def setUp(self, n=None, delta=None, rho=None, seed=6021):
        m = int(delta * n)

        self.k = int(rho * m)
        self.rho = rho
        self.y, self.A, self.alpha = use_rademacher(n, m, self.k, seed=seed)
        self.oracle_support = self.alpha != 0

        magni.cs.reconstruction.it.config.update(iterations=200)

    def tearDown(self):
        magni.cs.reconstruction.it.config.reset()

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

    def test_warm_start_IT(self, success_iht=True, success_ist=True):
        it_config = {'warm_start': 0.1 * np.ones(self.alpha.shape)}
        magni.cs.reconstruction.it.config.update(it_config)
        self._iht_run(self.y, self.A, self.alpha, success=success_iht)
        self._ist_run(self.y, self.A, self.alpha, success=success_ist)
        self.assertIsNotNone(
            magni.cs.reconstruction.it.config['warm_start'])

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

    def test_warm_start_IT(self):
        PhaseSpaceTest.test_warm_start_IT(self, success_iht=False,
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

    def test_warm_start_IT(self):
        PhaseSpaceTest.test_warm_start_IT(self, success_iht=False,
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

    def test_warm_start_IT(self):
        PhaseSpaceTest.test_warm_start_IT(self, success_iht=False,
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

    def test_warm_start_IT(self):
        PhaseSpaceTest.test_warm_start_IT(self, success_iht=False,
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

    def test_warm_start_IT(self):
        PhaseSpaceTest.test_warm_start_IT(self, success_iht=False,
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
