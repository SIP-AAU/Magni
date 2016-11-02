"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests for `magni.cs.phase_transition`.

Routine listings
----------------
TestAnalysis(unittest.TestCase)
    Tests of _analysis.run
TestAnalysisEstimatePT(unittest.TestCase)
    Tests of the _analysis._estimatePT function.
TestDataGenerateMatrix(unittest.TestCase)
    Tests of the _data.generate_matrix function.
TestDataGenerateNoise(unittest.TestCase)
    Tests of the _data.generate_noise function.
TestDataGenerateVector(unittest.TestCase)
    Tests of the _data.generate_vector function.
TestDetermine(unittest.TestCase)
    Tests of _util.determine function.
class TestIO(unittest.TestCase)
    Tests of the io module.
TestSimulationSimulate(unittest.TestCase)
    Tests of the _simulation._simulate function.

"""

from __future__ import division
from contextlib import contextmanager
import os
import shutil
import sys
import unittest
import warnings

import numpy as np
import tables as tb
try:
    import sklearn
except ImportError:
    sklearn = False

import magni


class TestAnalysis(unittest.TestCase):
    """
    Tests of _analysis.run

    **Testing Strategy**
    Test logistic regression phase transition fit against reference.

    """

    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.file_ = 'pt_test_data.hdf5'
        shutil.copy(os.path.join(self.path, self.file_), self.file_)
        self.label = 'test_pt_data'
        magni.cs.phase_transition.config.update(
            {'delta': np.linspace(0.025, 1.00, 40).tolist(),
             'monte_carlo': 10,
             'problem_size': 32**2,
             'rho': np.linspace(0.01, 1.00, 100).tolist(),
             'seed': 6021})

        self.assertTrue(os.path.exists(self.file_))
        with tb.File(self.file_, mode='r') as h5_file:
            self.reference_pt = h5_file.root.ref_phase_transition.read()
            self.reference_pt_80db = (
                h5_file.root.ref_phase_transition_highsens.read())

        self.synthetic_file = 'cs_phase_transition_analysis_test.hdf5'
        self.assertFalse(os.path.exists(self.synthetic_file))
        with tb.File(self.synthetic_file, mode='a') as h5_file:
            h5_file.create_group('/', 'all_or_none_pt')
            h5_file.create_array(
                '/all_or_none_pt', 'dist',
                np.concatenate(
                    (np.zeros((1, 100, 10)), np.ones((1, 100, 10))), axis=0))

    def tearDown(self):
        magni.cs.phase_transition.config.reset()
        if os.path.exists(self.file_):
            os.remove(self.file_)
        self.assertFalse(os.path.exists(self.file_))

        if os.path.exists(self.synthetic_file):
            os.remove(self.synthetic_file)
        self.assertFalse(os.path.exists(self.synthetic_file))

    def test_built_in_solver_reference_default(self):
        self.assertEqual(
            magni.cs.phase_transition.config['logit_solver'], 'built-in')
        self.assertEqual(magni.cs.phase_transition.config['SNR'], 40)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            magni.cs.phase_transition._analysis.run(self.file_, self.label)
        with tb.File(self.file_, mode='r') as h5_file:
            estimated_pt = h5_file.get_node(
                '/'.join(['', self.label, 'phase_transition'])).read()
            estimated_percentiles = h5_file.get_node(
                '/'.join(['', self.label, 'phase_transition_percentiles'])
            ).read()
        self.assertTrue(np.allclose(estimated_pt, self.reference_pt))
        for p in range(estimated_percentiles.shape[0]):
            if p < 2:
                # 10%, 25% percentiles - should be above 50%
                self.assertTrue(
                    np.all(estimated_percentiles[p, :] >= self.reference_pt))
            else:
                # 75%, 90% percentiles - should be below 50%
                self.assertTrue(
                    np.all(estimated_percentiles[p, :] <= self.reference_pt))

    def test_built_in_solver_reference_80dB(self):
        magni.cs.phase_transition.config['SNR'] = 80
        self.assertEqual(
            magni.cs.phase_transition.config['logit_solver'], 'built-in')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            magni.cs.phase_transition._analysis.run(self.file_, self.label)
        with tb.File(self.file_, mode='r') as h5_file:
            estimated_pt = h5_file.get_node(
                '/'.join(['', self.label, 'phase_transition'])).read()
            estimated_percentiles = h5_file.get_node(
                '/'.join(['', self.label, 'phase_transition_percentiles'])
            ).read()
        self.assertTrue(
            np.allclose(estimated_pt, self.reference_pt_80db, atol=1e-1))
        for p in range(estimated_percentiles.shape[0]):
            if p < 2:
                # 10%, 25% percentiles - should be above 50%
                self.assertTrue(
                    np.all(
                        estimated_percentiles[p, :] >= self.reference_pt_80db))
            else:
                # 75%, 90% percentiles - should be below 50%
                self.assertTrue(
                    np.all(
                        estimated_percentiles[p, :] <= self.reference_pt_80db))
        self.assertEqual(magni.cs.phase_transition.config['SNR'], 80)

    @unittest.skipIf(not sklearn, 'Scikit-learn is not available')
    def test_sklearn_solver_reference_default(self):
        magni.cs.phase_transition.config['logit_solver'] = 'sklearn'
        self.assertEqual(magni.cs.phase_transition.config['SNR'], 40)
        magni.cs.phase_transition._analysis.run(self.file_, self.label)
        with tb.File(self.file_, mode='r') as h5_file:
            estimated_pt = h5_file.get_node(
                '/'.join(['', self.label, 'phase_transition'])).read()
            estimated_percentiles = h5_file.get_node(
                '/'.join(['', self.label, 'phase_transition_percentiles'])
            ).read()
        self.assertTrue(
            np.allclose(estimated_pt, self.reference_pt, atol=1e-1))
        for p in range(estimated_percentiles.shape[0]):
            if p < 2:
                # 10%, 25% percentiles - should be above 50%
                self.assertTrue(np.all(estimated_percentiles[p, :-1] >=
                                       self.reference_pt[:-1]))
            else:
                # 75%, 90% percentiles - should be below 50%
                self.assertTrue(
                    np.all(estimated_percentiles[p, :] <= self.reference_pt))
        self.assertEqual(
            magni.cs.phase_transition.config['logit_solver'], 'sklearn')

    @unittest.skipIf(not sklearn, 'Scikit-learn is not available')
    def test_sklearn_solver_reference_80dB(self):
        magni.cs.phase_transition.config['SNR'] = 80
        magni.cs.phase_transition.config['logit_solver'] = 'sklearn'
        magni.cs.phase_transition._analysis.run(self.file_, self.label)
        with tb.File(self.file_, mode='r') as h5_file:
            estimated_pt = h5_file.get_node(
                '/'.join(['', self.label, 'phase_transition'])).read()
            estimated_percentiles = h5_file.get_node(
                '/'.join(['', self.label, 'phase_transition_percentiles'])
            ).read()
        self.assertTrue(
            np.allclose(estimated_pt, self.reference_pt_80db))
        for p in range(estimated_percentiles.shape[0]):
            if p < 2:
                # 10%, 25% percentiles - should be above 50%
                self.assertTrue(np.all(estimated_percentiles[p, :-1] >=
                                       self.reference_pt_80db[:-1]))
            else:
                # 75%, 90% percentiles - should be below 50%
                self.assertTrue(
                    np.all(
                        estimated_percentiles[p, :] <= self.reference_pt_80db))
        self.assertEqual(
            magni.cs.phase_transition.config['logit_solver'], 'sklearn')
        self.assertEqual(magni.cs.phase_transition.config['SNR'], 80)

    def test_built_in_solver_all_or_none(self):
        magni.cs.phase_transition.config['delta'] = [0.2, 0.3]
        self.assertEqual(
            magni.cs.phase_transition.config['logit_solver'], 'built-in')
        magni.cs.phase_transition._analysis.run(
            self.synthetic_file, 'all_or_none_pt')
        with tb.File(self.synthetic_file, mode='r') as h5_file:
            estimated_pt = h5_file.get_node(
                '/'.join(['', 'all_or_none_pt', 'phase_transition'])).read()
        self.assertTrue(np.allclose(estimated_pt, np.array([1, 0])))

    @unittest.skipIf(not sklearn, 'Scikit-learn is not available')
    def test_sklearn_solver_all_or_none(self):
        magni.cs.phase_transition.config['delta'] = [0.2, 0.3]
        magni.cs.phase_transition.config['logit_solver'] = 'sklearn'
        self.assertEqual(
            magni.cs.phase_transition.config['logit_solver'], 'sklearn')
        magni.cs.phase_transition._analysis.run(
            self.synthetic_file, 'all_or_none_pt')
        with tb.File(self.synthetic_file, mode='r') as h5_file:
            estimated_pt = h5_file.get_node(
                '/'.join(['', 'all_or_none_pt', 'phase_transition'])
            ).read()
        self.assertTrue(np.allclose(estimated_pt, np.array([1, 0])))


class TestAnalysisEstimatePT(unittest.TestCase):
    """
    Tests of the _analysis._estimatePT function.

    **Testing Strategy**
    Test af standard logistic fit and its "reverse".

    """

    def test_std_logistic(self):
        rho = np.linspace(0, 1, 11)
        success = np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])

        pt_val = magni.cs.phase_transition._analysis._estimate_PT(
            rho, success, [0.5])
        self.assertEqual(np.round(pt_val[0], 2), 0.45)

    def test_non_convergence(self):
        rho = np.linspace(0, 1, 11)
        success = np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])[::-1]

        with self.assertRaises(RuntimeWarning):
            pt_val = magni.cs.phase_transition._analysis._estimate_PT(
                rho, success, [0.5])


class TestBackup(unittest.TestCase):
    """
    Test of the _backup functionality.

    **Testing Strategy**
    Create a backup and test get and set methods.

    """

    def setUp(self):
        self.path = 'cs_phase_transition_backup_tests.hdf5'
        self.assertFalse(os.path.exists(self.path))
        magni.cs.phase_transition._backup.create(self.path)
        self.assertTrue(os.path.exists(self.path))

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_get_backup(self):
        done = magni.cs.phase_transition._backup.get(self.path)
        self.assertFalse(np.any(done))

    def test_set_backup(self):
        magni.cs.phase_transition._backup.set(
            self.path, (0, 0), 1.0, 2.0, 3.0, 4.0)
        done = magni.cs.phase_transition._backup.get(self.path)
        self.assertTrue(np.any(done))

        with tb.File(self.path, mode='r') as h5:
            self.assertEqual(h5.root.time[0, 0], 1.0)
            self.assertEqual(h5.root.dist[0, 0], 2.0)
            self.assertEqual(h5.root.mse[0, 0], 3.0)
            self.assertEqual(h5.root.norm[0, 0], 4.0)


class TestDataGenerateMatrix(unittest.TestCase):
    """
    Tests of the _data.generate_matrix function.

    **Testing Strategy**
    Matrices generated by the function are compared to known "true" matrices.

    """

    def setUp(self):
        self.l = 4
        self.m = 3
        self.n = 9
        self.seed = 6021
        np.random.seed(self.seed)

    def tearDown(self):
        magni.cs.phase_transition.config.reset()

    def test_default(self):
        A_true = np.array([[0.91579356, 0.8554431, -0.73516709, -0.5351897,
                            0.92666919, 0.73139786,  0.36218319, -0.59300362,
                            0.90664675],
                           [-0.0671974, -0.47660808, -0.66815151, -0.72081901,
                            -0.34468206, -0.47720612, -0.16854207, -0.36796994,
                            0.39610724],
                           [-0.39598822, -0.20263723, -0.11446798, -0.44044516,
                            -0.14992827, -0.4871668, -0.91674255, -0.71620166,
                            0.14522645]])

        A = magni.cs.phase_transition._data.generate_matrix(self.m, self.n)

        self.assertTrue(np.allclose(A, A_true))

    def test_uniform_line_2d_dct(self):
        Phi_true = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        n_sqrt = np.sqrt(self.n)
        self.assertTrue(n_sqrt.is_integer())
        n_sqrt = int(n_sqrt)
        Psi_true = magni.imaging.dictionaries.get_DCT((n_sqrt, n_sqrt)).A

        A_true = Phi_true.dot(Psi_true)

        magni.cs.phase_transition.config['system_matrix'] = 'RandomDCT2D'
        A = magni.cs.phase_transition._data.generate_matrix(self.m, self.n).A

        self.assertEqual(magni.cs.phase_transition.config['system_matrix'],
                         'RandomDCT2D')
        self.assertTrue(np.allclose(A, A_true))

        with self.assertRaises(ValueError):
            magni.cs.phase_transition._data.generate_matrix(3, 7)

    def test_custom_system_matrix(self):
        magni.cs.phase_transition.config['system_matrix'] = 'custom'
        with self.assertRaises(TypeError):
            # One must configure a system matrix factory
            magni.cs.phase_transition._data.generate_matrix(self.m, self.n)

        def generate_custom_ndarray(m, n):
            return np.random.randint(10, size=(m, n))

        magni.cs.phase_transition.config[
            'custom_system_matrix_factory'] = generate_custom_ndarray
        A_true_ndarray = np. array([[3, 7, 6, 5, 4, 2, 2, 3, 6],
                                    [3, 9, 8, 7, 0, 1, 4, 0, 5],
                                    [2, 7, 6, 0, 9, 2, 4, 5, 5]])
        A = magni.cs.phase_transition._data.generate_matrix(self.m, self.n)
        self.assertTrue(np.allclose(A, A_true_ndarray))

        def generate_wrong_ndarray(m, n):
            return np.random.randint(10, size=(n, m))

        magni.cs.phase_transition.config[
            'custom_system_matrix_factory'] = generate_wrong_ndarray
        with self.assertRaises(ValueError):
            magni.cs.phase_transition._data.generate_matrix(self.m, self.n)

        self.assertEqual(
            magni.cs.phase_transition.config['system_matrix'], 'custom')


class TestDataGenerateNoise(unittest.TestCase):
    """
    Tests of the _data.generate_noise function.

    **Testing Strategy**
    Various noise vectors are generated from a specified SNR. An estimated SNR
    is compared to the specified SNR.

    """

    def setUp(self):
        np.random.seed(6021)
        self.n = 1241415
        self.m = 148411
        self.k = 12151
        self.SNRs = [float(SNR) if SNR % 2 == 0 else SNR
                     for SNR in np.random.choice(
                             np.arange(1, 60), size=10, replace=False)]
        self.coefficients = ['rademacher', 'gaussian', 'laplace', 'bernoulli']

    def tearDown(self):
        magni.cs.phase_transition.config.reset()

    def test_AWGN(self):
        magni.cs.phase_transition.config['noise'] = 'AWGN'
        for coeffs in self.coefficients:
            magni.cs.phase_transition.config['coefficients'] = coeffs
            signal_vector = magni.cs.phase_transition._data.generate_vector(
                self.n, self.k)
            estimated_signal_power = (
                1.0/self.n * np.linalg.norm(signal_vector)**2)

            for SNR in self.SNRs:
                magni.cs.phase_transition.config['SNR'] = SNR
                noise_vector = magni.cs.phase_transition._data.generate_noise(
                    self.m, self.n, self.k)
                estimated_noise_power = (
                    1.0/self.m * np.linalg.norm(noise_vector)**2)
                estimated_SNR = 10 * np.log10(estimated_signal_power /
                                              estimated_noise_power)
                self.assertTrue(np.allclose(estimated_SNR, SNR, atol=5e-1))
                self.assertEqual(magni.cs.phase_transition.config['SNR'], SNR)

            self.assertEqual(
                magni.cs.phase_transition.config['coefficients'], coeffs)

        self.assertEqual(magni.cs.phase_transition.config['noise'], 'AWGN')

    def test_AWLN(self):
        magni.cs.phase_transition.config['noise'] = 'AWLN'
        for coeffs in self.coefficients:
            magni.cs.phase_transition.config['coefficients'] = coeffs
            signal_vector = magni.cs.phase_transition._data.generate_vector(
                self.n, self.k)
            estimated_signal_power = (
                1.0/self.n * np.linalg.norm(signal_vector)**2)

            for SNR in self.SNRs:
                magni.cs.phase_transition.config['SNR'] = SNR
                noise_vector = magni.cs.phase_transition._data.generate_noise(
                    self.m, self.n, self.k)
                estimated_noise_power = (
                    1.0/self.m * np.linalg.norm(noise_vector)**2)
                estimated_SNR = 10 * np.log10(estimated_signal_power /
                                              estimated_noise_power)

                self.assertTrue(np.allclose(estimated_SNR, SNR, atol=1e-1))
                self.assertEqual(magni.cs.phase_transition.config['SNR'], SNR)

            self.assertEqual(
                magni.cs.phase_transition.config['coefficients'], coeffs)

        self.assertEqual(magni.cs.phase_transition.config['noise'], 'AWLN')

    def test_custom_noise(self):
        magni.cs.phase_transition.config['noise'] = 'custom'
        with self.assertRaises(TypeError):
            # One must configure a noise factory
            magni.cs.phase_transition._data.generate_noise(
                self.m, self.n, self.k)

        def generate_custom_noise(m, n, k, noise_power):
            return np.random.normal(scale=np.sqrt(noise_power), size=(m, 1))

        magni.cs.phase_transition.config[
            'custom_noise_factory'] = generate_custom_noise
        for coeffs in self.coefficients:
            magni.cs.phase_transition.config['coefficients'] = coeffs
            signal_vector = magni.cs.phase_transition._data.generate_vector(
                self.n, self.k)
            estimated_signal_power = (
                1.0/self.n * np.linalg.norm(signal_vector)**2)

            for SNR in self.SNRs:
                magni.cs.phase_transition.config['SNR'] = SNR
                noise_vector = magni.cs.phase_transition._data.generate_noise(
                    self.m, self.n, self.k)
                estimated_noise_power = (
                    1.0/self.m * np.linalg.norm(noise_vector)**2)
                estimated_SNR = 10 * np.log10(estimated_signal_power /
                                              estimated_noise_power)

                print(estimated_SNR, SNR)

                self.assertTrue(np.allclose(estimated_SNR, SNR, atol=1e-1))
                self.assertEqual(magni.cs.phase_transition.config['SNR'], SNR)

            self.assertEqual(
                magni.cs.phase_transition.config['coefficients'], coeffs)

        def generate_wrong_noise_shape(m, n, k, noise_power):
            return np.random.normal(scale=np.sqrt(noise_power), size=(1, n))

        magni.cs.phase_transition.config[
            'custom_noise_factory'] = generate_wrong_noise_shape
        with self.assertRaises(ValueError):
            magni.cs.phase_transition._data.generate_noise(
                self.m, self.n, self.k)

        self.assertEqual(magni.cs.phase_transition.config['noise'], 'custom')

    def test_None_noise(self):
        self.assertEqual(magni.cs.phase_transition.config['noise'], None)
        with self.assertRaises(RuntimeError):
            magni.cs.phase_transition._data.generate_noise(
                self.m, self.n, self.k)


class TestDataGenerateVector(unittest.TestCase):
    """
    Tests of the _data.generate_vector function.

    **Testing Strategy**
    Vectors generated by the function are compared to known "true" vectors.

    """

    def setUp(self):
        self.n = 17
        self.k = 8
        self.seed = 6021
        np.random.seed(self.seed)

    def tearDown(self):
        magni.cs.phase_transition.config.reset()

    def test_default(self):
        x_true = np.array([[1.], [1.], [1.], [-1.], [-1.], [1.], [-1.], [-1.],
                           [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]
                           ])

        x = magni.cs.phase_transition._data.generate_vector(self.n, self.k)

        self.assertTrue(np.allclose(x_true, x))

    def test_rademacher(self):
        magni.cs.phase_transition.config.update(coefficients='rademacher')
        x_true = np.array([[1.], [1.], [1.], [-1.], [-1.], [1.], [-1.], [-1.],
                           [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]
                           ])

        x = magni.cs.phase_transition._data.generate_vector(self.n, self.k)

        self.assertTrue(np.allclose(x_true, x))
        self.assertEqual(
            magni.cs.phase_transition.config['coefficients'], 'rademacher')

    def test_gaussian(self):
        magni.cs.phase_transition.config.update(coefficients='gaussian')
        x_true = np.array([[0.97267375], [0.95762797], [-1.26236611],
                           [-0.71956614], [1.96767297], [0.5739882],
                           [0.49924214], [-0.98133296],
                           [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]
                           ])

        x = magni.cs.phase_transition._data.generate_vector(self.n, self.k)

        self.assertTrue(np.allclose(x_true, x))
        self.assertEqual(
            magni.cs.phase_transition.config['coefficients'], 'gaussian')

    def test_laplace(self):
        magni.cs.phase_transition.config.update(coefficients='laplace')
        x_true = np.array([[0.86206783], [-1.74301165], [0.58042009],
                           [0.59285876], [-0.34547475], [-0.71840635],
                           [1.52441889], [1.24356796],
                           [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]
                           ])

        x = magni.cs.phase_transition._data.generate_vector(self.n, self.k)

        self.assertTrue(np.allclose(x_true, x))
        self.assertEqual(
            magni.cs.phase_transition.config['coefficients'], 'laplace')

    def test_bernoulli(self):
        magni.cs.phase_transition.config.update(coefficients='bernoulli')
        x_true = np.array([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
                           [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]
                           ])

        x = magni.cs.phase_transition._data.generate_vector(self.n, self.k)

        self.assertTrue(np.allclose(x_true, x))
        self.assertEqual(
            magni.cs.phase_transition.config['coefficients'], 'bernoulli')

    def test_linear_support(self):
        support_distribution = np.reshape((np.arange(self.n) + 1) /
                                          (self.n * (self.n + 1) / 2),
                                          (self.n, 1))
        magni.cs.phase_transition.config.update(
            support_distribution=support_distribution)
        x_true = np.array([[0.], [0.], [0.], [0.], [-1.], [0.], [0.], [0.],
                           [-1.], [1.], [0.], [-1.], [1.], [0.], [-1.], [1.],
                           [-1.]])

        x = magni.cs.phase_transition._data.generate_vector(self.n, self.k)

        self.assertTrue(np.allclose(x_true, x))
        self.assertEqual(
            magni.cs.phase_transition.config['coefficients'],
            'rademacher')
        self.assertIs(
            magni.cs.phase_transition.config['support_distribution'],
            support_distribution)

    def test_exponential_support(self):
        support_distribution = np.exp(-np.arange(self.n)).reshape(self.n, 1)
        support_distribution /= np.sum(support_distribution)
        magni.cs.phase_transition.config.update(
            support_distribution=support_distribution)
        x_true = np.array([[-1.], [1.], [-1.], [-1.], [1.], [1.], [1.], [-1.],
                           [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]
                           ])

        x = magni.cs.phase_transition._data.generate_vector(self.n, self.k)
        self.assertTrue(np.allclose(x_true, x))
        self.assertEqual(
            magni.cs.phase_transition.config['coefficients'],
            'rademacher')
        self.assertIs(
            magni.cs.phase_transition.config['support_distribution'],
            support_distribution)

    def test_uniform_support(self):
        support_distribution = np.ones((self.n, 1)) / self.n
        magni.cs.phase_transition.config.update(
            support_distribution=support_distribution)
        x_true = np.array([[0.], [-1.], [0.], [0.], [-1.], [0.], [-1.], [0.],
                           [-1.], [0.], [0.], [0.], [1.], [1.], [-1.], [1.],
                           [0.]])

        x = magni.cs.phase_transition._data.generate_vector(self.n, self.k)
        self.assertTrue(np.allclose(x_true, x))
        self.assertEqual(
            magni.cs.phase_transition.config['coefficients'],
            'rademacher')
        self.assertIs(
            magni.cs.phase_transition.config['support_distribution'],
            support_distribution)


class TestDetermine(unittest.TestCase):
    """
    Tests of _util.determine function.

    This function is the main entry function
    `magni.cs.phase_transition.determine`.

    **Testing Strategy**
    A very small phase transition run is compared to a reference to test that
    phase transitions are stored correctly.

    The kwarg pass through and pre_simulation_hook functionality of the phase
    transition subpackage is tested by comparing the reported setup to a
    reference.

    Error handling is tested separately.

    """

    def setUp(self):
        magni.utils.multiprocessing.config['workers'] = 1
        self.determine_test_path = 'cs_phase_transition_determine_test.hdf5'

        magni.cs.reconstruction.gamp.config.update({
            'input_channel_parameters': {
                'tau': 0.5, 'theta_bar': 0, 'theta_tilde': 1, 'use_em': False},
            'output_channel_parameters': {
                'sigma_sq': 1, 'noise_level_estimation': 'sample_variance'},
            'iterations': 10,
            'report_A_asq_setup': True
        })

        magni.cs.phase_transition.config.update({
            'seed': 6021,
            'problem_size': 32**2,
            'delta': [0.3],
            'rho': [0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            'coefficients': 'laplace',
            'system_matrix': 'RandomDCT2D',
            'monte_carlo': 1
        })

        self.dummy_path = 'cs_phase_transition_error_handling.hdf5'
        self.assertFalse(os.path.exists(self.dummy_path))

    def tearDown(self):
        magni.utils.multiprocessing.config.reset()
        magni.cs.reconstruction.gamp.config.reset()
        magni.cs.phase_transition.config.reset()
        if os.path.exists(self.dummy_path):
            os.remove(self.dummy_path)
        self.assertFalse(os.path.exists(self.dummy_path))

    def test_sum_approximation_gamp(self):
        magni.utils.multiprocessing.config['workers'] = 0
        file_ = 'ma_gamp'
        with _capture_output(file_):
            magni.cs.phase_transition.determine(
                magni.cs.reconstruction.gamp.run,
                'cs_phase_transition_determine_test.hdf5',
                label='ma_gamp_test',
                pre_simulation_hook=_set_tau)

        with open(file_ + '.stdout', mode='r') as stdout:
            for line in stdout:
                if line.startswith('GAMP'):
                    self.assertEqual(
                        line.strip()[:79],
                        'GAMP is using the A_asq: ' +
                        '<magni.utils.matrices.SumApproximationMatrix' +
                        ' object at')
                else:
                    self.assertEqual(
                        line.strip(),
                        'The sum approximation method is: rangan')

    def test_full_gamp(self):
        magni.utils.multiprocessing.config['workers'] = 0
        file_ = 'full_gamp'
        with _capture_output(file_):
            magni.cs.phase_transition.determine(
                magni.cs.reconstruction.gamp.run,
                'cs_phase_transition_determine_test.hdf5',
                label='full_gamp_test',
                pre_simulation_hook=_construct_A_asq_and_set_tau)

        with open(file_ + '.stdout', mode='r') as stdout:
            for line in stdout:
                if line.startswith('GAMP'):
                    self.assertEqual(
                        line.strip()[:73],
                        'GAMP is using the A_asq: ' +
                        '<magni.utils.matrices.MatrixCollection object at')
                else:
                    self.assertEqual(
                        line.strip(),
                        'The sum approximation method is: None')

    def test_z_correct_dists_ma_gamp(self):
        with tb.File('cs_phase_transition_determine_test.hdf5') as h5_file:
            ma_gamp_dist = h5_file.get_node('/ma_gamp_test/dist').read()

        if sys.version_info[0] == 2:
            ma_gamp_dist_ref = np.array(
                [[[0.00052035], [0.00670562], [0.04877081],
                  [0.11715555], [0.17149385], [0.41228690]]])
        else:
            # random.sample is not stable across Python 2 --> 3 border
            ma_gamp_dist_ref = np.array(
                [[[2.72592311e-04], [2.48630874e-03], [6.26038548e-02],
                  [1.65474043e-01], [3.03143840e-01], [3.92334845e-01]]])

        self.assertTrue(np.allclose(ma_gamp_dist, ma_gamp_dist_ref))

    def test_z_correct_dists_full_gamp(self):
        with tb.File('cs_phase_transition_determine_test.hdf5') as h5_file:
            full_gamp_dist = h5_file.get_node('/full_gamp_test/dist').read()

        if sys.version_info[0] == 2:
            full_gamp_dist_ref = np.array(
                [[[3.72378090e-04], [4.89454444e-03], [4.51150168e-02],
                  [1.11151619e-01], [1.70156455e-01], [4.09506352e-01]]])
        else:
            # random.sample is not stable across Python 2 --> 3 border
            full_gamp_dist_ref = np.array(
                [[[1.62446873e-04], [2.36392118e-03], [6.09754113e-02],
                  [1.61375617e-01], [2.92454219e-01], [3.91102236e-01]]])

        self.assertTrue(np.allclose(full_gamp_dist, full_gamp_dist_ref))

    def test_invalid_label_character(self):
        with self.assertRaises(RuntimeError):
            magni.cs.phase_transition.determine(
                _dummy_func, self.dummy_path, label='(')

    def test_invalid_label_path(self):
        with self.assertRaises(RuntimeError):
            magni.cs.phase_transition.determine(
                _dummy_func, self.dummy_path, label='/some/path/')

    def test_noncallable_pre_simulation_hook(self):
        with self.assertRaises(RuntimeError):
            magni.cs.phase_transition.determine(
                _dummy_func, self.dummy_path, pre_simulation_hook='')

    def test_overwrite_label(self):
        magni.utils.multiprocessing.config['workers'] = 0
        label = 'test'
        with tb.File(self.dummy_path, mode='w') as h5:
            h5.create_array('/', label, np.array([1]))

        with self.assertRaises(IOError):
            magni.cs.phase_transition.determine(
                _dummy_func, self.dummy_path, label=label)

        with self.assertRaises(RuntimeError):
            # Should overwrite label but fail on executing _dummy_func
            magni.cs.phase_transition.determine(
                _dummy_func, self.dummy_path, label=label, overwrite=True)


class TestIO(unittest.TestCase):
    """
    Tests of the io module.

    **Testing Strategy**
    Input/Output values are compared to a reference.

    """

    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.file_ = 'pt_test_data.hdf5'
        shutil.copy(os.path.join(self.path, self.file_), self.file_)
        self.assertTrue(os.path.exists(self.file_))

        self.label = 'test_io_data'
        with tb.File(self.file_, mode='r') as h5_file:
            self.reference_rho = h5_file.get_node(
                '/' + self.label + '/phase_transition').read()
        self.default_delta = np.linspace(0, 1, len(self.reference_rho) + 1)[1:]

    def tearDown(self):
        if os.path.exists(self.file_):
            os.remove(self.file_)
        self.assertFalse(os.path.exists(self.file_))

    def test_default(self):
        delta, rho = magni.cs.phase_transition.io.load_phase_transition(
            self.file_, label=self.label)
        self.assertTrue(np.allclose(delta, self.default_delta))
        self.assertTrue(np.allclose(rho, self.reference_rho))

    def test_custom_delta(self):
        custom_delta = np.ones(self.reference_rho.shape)
        delta, rho = magni.cs.phase_transition.io.load_phase_transition(
            self.file_, label=self.label, delta=custom_delta)
        self.assertTrue(np.allclose(delta, custom_delta))
        self.assertTrue(np.allclose(rho, self.reference_rho))

    def test_invalid_delta(self):
        with self.assertRaises(ValueError):
            # Invalid delta.shape
            magni.cs.phase_transition.io.load_phase_transition(
                self.file_, label=self.label, delta=np.zeros(1))

        with self.assertRaises(ValueError):
            # Invalid range of delta
            magni.cs.phase_transition.io.load_phase_transition(
                self.file_, label=self.label,
                delta=3 * np.ones(self.reference_rho.shape))


class TestSimulationSimulate(unittest.TestCase):
    """
    Tests of the _simulation._simulate function.

    **Testing Strategy**
    Phase transition distance simulations are compared to known "true"
    distances.

    """

    _backup = magni.cs.phase_transition._simulation._backup

    def setUp(self):
        self.seed = 6021
        self.delta = [0.2]
        self.rho = [0.1]
        self.monte_carlo = 1
        self.coefficients = 'rademacher'
        self.n = magni.cs.phase_transition.config['problem_size']

        class BackupStub():
            def set(self, path, ij_tup, stat_time, stat_dist, stat_mse,
                    stat_norm):
                self.path = path
                self.ij_tup = ij_tup
                self.stat_time = stat_time
                self.stat_dist = stat_dist
                self.stat_mse = stat_mse
                self.stat_norm = stat_norm

        self.backup_stub = BackupStub()

        magni.cs.phase_transition._simulation._backup = self.backup_stub

    def tearDown(self):
        magni.cs.reconstruction.it.config.reset()
        magni.cs.phase_transition.config.reset()
        magni.cs.phase_transition._simulation._backup = self.__class__._backup

    def test_default(self):
        magni.cs.phase_transition.config.update(
            {'seed': self.seed, 'delta': self.delta, 'rho': self.rho,
             'monte_carlo': self.monte_carlo,
             'coefficients': self.coefficients})

        path = 'path'
        ij_tup = (0, 0)

        self.assertFalse(self.backup_stub.__dict__)

        magni.cs.phase_transition._simulation._simulate(
            magni.cs.reconstruction.it.run, ij_tup, [self.seed], path)

        self.assertEqual(self.backup_stub.path, path)
        self.assertEqual(self.backup_stub.ij_tup, ij_tup)
        self.assertTrue(np.allclose(self.backup_stub.stat_mse,
                                    np.array([[1.08899121e-07]])))
        self.assertTrue(np.allclose(self.backup_stub.stat_norm,
                                    np.array([4])))
        self.assertTrue(np.allclose(
            self.backup_stub.stat_dist,
            self.backup_stub.stat_mse * self.n / self.backup_stub.stat_norm**2)
        )

    def test_pre_sim_hook(self):
        magni.cs.phase_transition.config.update(
            {'seed': self.seed, 'delta': self.delta, 'rho': self.rho,
             'monte_carlo': self.monte_carlo,
             'coefficients': self.coefficients})

        def pre_sim_hook(var):
            magni.cs.reconstruction.it.config.update(
                {'threshold': 'fixed',
                 'threshold_fixed': int(self.rho[0] * self.delta[0] * self.n),
                 'kappa': 'fixed', 'kappa_fixed': 0.45, 'tolerance': 1e-4})

        path = 'path'
        ij_tup = (0, 0)

        self.assertFalse(self.backup_stub.__dict__)

        magni.cs.phase_transition._simulation._simulate(
            magni.cs.reconstruction.it.run, ij_tup, [self.seed], path,
            pre_sim_hook)

        self.assertEqual(self.backup_stub.path, path)
        self.assertEqual(self.backup_stub.ij_tup, ij_tup)
        self.assertTrue(np.allclose(self.backup_stub.stat_mse,
                                    np.array([[2.47979301e-10]])))
        self.assertTrue(np.allclose(self.backup_stub.stat_norm,
                                    np.array([4])))
        self.assertTrue(np.allclose(
            self.backup_stub.stat_dist,
            self.backup_stub.stat_mse * self.n / self.backup_stub.stat_norm**2)
        )

    def test_true_linear_support(self):
        support_distribution = np.reshape((np.arange(self.n) + 1) /
                                          (self.n * (self.n + 1) / 2),
                                          (self.n, 1))
        weights = np.linspace(0.5, 1.0, self.n).reshape(-1, 1)
        magni.cs.phase_transition.config.update(
            {'seed': self.seed, 'delta': self.delta, 'rho': self.rho,
             'monte_carlo': self.monte_carlo,
             'coefficients': self.coefficients,
             'support_distribution': support_distribution})
        magni.cs.reconstruction.it.config.update(
            {'threshold': 'fixed',
             'threshold_fixed': int(self.rho[0] * self.delta[0] * self.n),
             'threshold_operator': 'weighted_hard',
             'threshold_weights': weights,
             'kappa': 'fixed', 'kappa_fixed': 0.45, 'tolerance': 1e-4})

        path = 'path'
        ij_tup = (0, 0)

        self.assertFalse(self.backup_stub.__dict__)

        magni.cs.phase_transition._simulation._simulate(
            magni.cs.reconstruction.it.run, ij_tup, [self.seed], path)

        self.assertEqual(self.backup_stub.path, path)
        self.assertEqual(self.backup_stub.ij_tup, ij_tup)
        self.assertTrue(np.allclose(self.backup_stub.stat_mse,
                                    np.array([[3.76042577e-10]])))
        self.assertTrue(np.allclose(self.backup_stub.stat_norm,
                                    np.array([4])))
        self.assertTrue(np.allclose(
            self.backup_stub.stat_dist,
            self.backup_stub.stat_mse * self.n / self.backup_stub.stat_norm**2)
        )

    def test_false_linear_support(self):
        support_distribution = np.reshape((np.arange(self.n) + 1) /
                                          (self.n * (self.n + 1) / 2),
                                          (self.n, 1))
        weights = np.linspace(1.0, 0.5, self.n).reshape(-1, 1)
        magni.cs.phase_transition.config.update(
            {'seed': self.seed, 'delta': self.delta, 'rho': self.rho,
             'monte_carlo': self.monte_carlo,
             'coefficients': self.coefficients,
             'support_distribution': support_distribution})
        magni.cs.reconstruction.it.config.update(
            {'threshold': 'fixed',
             'threshold_fixed': int(self.rho[0] * self.delta[0] * self.n),
             'threshold_operator': 'weighted_hard',
             'threshold_weights': weights,
             'kappa': 'fixed', 'kappa_fixed': 0.45, 'tolerance': 1e-4})

        path = 'path'
        ij_tup = (0, 0)

        self.assertFalse(self.backup_stub.__dict__)

        magni.cs.phase_transition._simulation._simulate(
            magni.cs.reconstruction.it.run, ij_tup, [self.seed], path)

        self.assertEqual(self.backup_stub.path, path)
        self.assertEqual(self.backup_stub.ij_tup, ij_tup)
        self.assertTrue(np.allclose(self.backup_stub.stat_mse,
                                    np.array([[0.00784895]])))
        self.assertTrue(np.allclose(self.backup_stub.stat_norm,
                                    np.array([4])))
        self.assertTrue(np.allclose(
            self.backup_stub.stat_dist,
            self.backup_stub.stat_mse * self.n / self.backup_stub.stat_norm**2)
        )

    def test_noisy_measurements(self):
        magni.cs.phase_transition.config.update(
            {'seed': self.seed, 'delta': self.delta, 'rho': self.rho,
             'monte_carlo': self.monte_carlo,
             'coefficients': self.coefficients,
             'noise': 'AWGN'})

        path = 'path'
        ij_tup = (0, 0)

        self.assertFalse(self.backup_stub.__dict__)

        magni.cs.phase_transition._simulation._simulate(
            magni.cs.reconstruction.it.run, ij_tup, [self.seed], path)

        self.assertEqual(magni.cs.phase_transition.config['noise'], 'AWGN')
        self.assertEqual(self.backup_stub.path, path)
        self.assertEqual(self.backup_stub.ij_tup, ij_tup)
        self.assertTrue(np.allclose(self.backup_stub.stat_mse,
                                    np.array([3.23953658e-07])))
        self.assertTrue(np.allclose(self.backup_stub.stat_norm,
                                    np.array([4])))
        self.assertTrue(np.allclose(
            self.backup_stub.stat_dist,
            self.backup_stub.stat_mse * self.n / self.backup_stub.stat_norm**2)
        )


# Pre-simulation hooks for GAMP
def _construct_A_asq_and_set_tau(var):
    """
    Build the entrywise absolute squared system matrix for AMP.

    Also call `set_tau`, to set the signal density.

    """

    Phi = var['A']._matrices[0]

    dct_mtx = magni.imaging.dictionaries.get_DCT_transform_matrix(
        int(np.sqrt(Phi.shape[1])))
    dct_mtx_sq = dct_mtx**2
    Psi_sq = magni.utils.matrices.Separable2DTransform(
        dct_mtx_sq.T, dct_mtx_sq.T)
    A_asq = magni.utils.matrices.MatrixCollection((Phi, Psi_sq))

    magni.cs.phase_transition.config['algorithm_kwargs'] = {'A_asq': A_asq}
    _set_tau(var)


def _set_tau(var):
    """
    Set GB prior "density" for AMP.

    """

    magni.cs.reconstruction.gamp.config[
        'input_channel_parameters']['tau'] = var['k'] / var['n']


def _dummy_func():
    pass


# Utilities
@contextmanager
def _capture_output(file_):
    """
    Contextmanager for capturing stdout and stderr.

    Inspired by http://stackoverflow.com/a/17981937.

    """

    cur_out, cur_err = sys.stdout, sys.stderr

    try:
        # Apparently, StringIO does not work with multiprocessing in a unittest
        cap_out = open(file_ + '.stdout', mode='a')  # Only capture last write
        cap_err = open(file_ + '.stderr', mode='a')  # Only capture last write
        sys.stdout, sys.stderr = cap_out, cap_err
        yield
    finally:
        sys.stdout, sys.stderr = cur_out, cur_err
        cap_out.close()
        cap_err.close()
