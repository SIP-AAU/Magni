"""
..
    Copyright (c) 2016-2017, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests for `magni.utils.matrices`.

"""

from __future__ import division
import unittest

import numpy as np

import magni


class TestMatrix(unittest.TestCase):
    """
    Test of Matrix.

    Implemented tests:

    * test_asarray : transforming to an explicit ndarray
    * test_invalid_matrix : invalid complex conj or T matrix
    * test_matrix_state : return internal matrix state

    """

    def test_asarray(self):
        func = lambda vec: -vec
        matrix = magni.utils.matrices.Matrix(func, func, (), (3, 3))
        matrix_as_ndarray = np.asarray(matrix)
        self.assertTrue(np.allclose(matrix.A, matrix_as_ndarray))

    def test_invalid_matrix(self):
        vector = np.ones((3, 1))
        func = lambda vec: -vec
        complex_matrix = magni.utils.matrices.Matrix(
            func, func, (), (3, 3), is_complex=True)

        self.assertTrue(np.allclose(complex_matrix.dot(vector), -vector))
        self.assertTrue(
            np.allclose(complex_matrix.conj().T.dot(vector), -vector))
        self.assertTrue(np.allclose(complex_matrix.A, -np.eye(3)))

        with self.assertRaises(ValueError):
            complex_matrix.T.dot(vector)
        with self.assertRaises(ValueError):
            complex_matrix.conj().dot(vector)
        with self.assertRaises(ValueError):
            complex_matrix.T.A
        with self.assertRaises(ValueError):
            complex_matrix.conj().A

    def test_matrix_state(self):
        func = lambda vec: -vec
        conj_trans = lambda vec: vec
        matrix = magni.utils.matrices.Matrix(func, conj_trans, (), (3, 3))
        state = matrix.matrix_state
        vec = np.arange(3)

        self.assertTrue(np.allclose(state['func'](vec), func(vec)))
        self.assertTrue(np.allclose(state['conj_trans'](vec), conj_trans(vec)))
        self.assertEqual(len(state['args']), 0)
        self.assertFalse(state['is_complex'])
        self.assertTrue(state['is_valid'])


class TestMatrixCollection(unittest.TestCase):
    """
    Test of MatrixCollection.

    Implemented tests:

    * test_asarray : transforming to an explicit ndarray
    * test_incompatible_matrices : matrices with incompatible shapes
    * test_matrix_state : return internal matrix state

    """

    def test_asarray(self):
        func1 = lambda vec: -vec
        matrix1 = magni.utils.matrices.Matrix(func1, func1, (), (3, 3))
        func2 = lambda vec: vec[::-1]
        matrix2 = magni.utils.matrices.Matrix(func2, func2, (), (3, 3))
        matrix_collection = magni.utils.matrices.MatrixCollection(
            (matrix1, matrix2))
        matrix_collection_as_ndarray = np.asarray(matrix_collection)
        self.assertTrue(
            np.allclose(matrix_collection.A, matrix_collection_as_ndarray))

    def test_incompatible_matrices(self):
        func = lambda vec: -vec
        matrix1 = magni.utils.matrices.Matrix(func, func, (), (3, 3))
        matrix2 = magni.utils.matrices.Matrix(func, func, (), (4, 4))

        with self.assertRaises(ValueError):
            matrix_collection = magni.utils.matrices.MatrixCollection(
                (matrix1, matrix2))

    def test_matrix_state(self):
        func1 = lambda vec: -vec
        func2 = lambda vec: vec
        matrix1 = magni.utils.matrices.Matrix(func1, func1, (), (3, 3))
        matrix2 = magni.utils.matrices.Matrix(func2, func2, (), (3, 3))
        matrix_collection = magni.utils.matrices.MatrixCollection(
            [matrix1, matrix2])
        state = matrix_collection.matrix_state
        vec = np.arange(3).reshape(3, 1)

        self.assertTrue(np.allclose(state['matrices'][0].dot(vec), func1(vec)))
        self.assertTrue(np.allclose(state['matrices'][1].dot(vec), func2(vec)))

        mc_matrices = matrix_collection._matrices
        self.assertTrue(state['matrices'] is not mc_matrices)
        self.assertTrue(state['matrices'][0] is not mc_matrices[0])
        self.assertTrue(state['matrices'][1] is not mc_matrices[1])


class TestSeparable2DTransform(unittest.TestCase):
    """
    Test of Separable2DTransform.

    Implemented tests:

    * test_asarray : transforming to an explicit ndarray
    * test_DCT : comparison with FFT based DCT
    * test_DFT : comparison with FFT based DFT
    * test_matrix_state : return internal matrix state

    """

    def setUp(self):
        np.random.seed(6021)
        self.N = 2**8
        self.vec = magni.imaging.mat2vec(np.random.randn(self.N, self.N))
        self.mtx = np.random.randn(10, 10)

    def test_asarray(self):
        sep_matrix = magni.utils.matrices.Separable2DTransform(
            self.mtx, self.mtx)
        sep_matrix_as_ndarray = np.asarray(sep_matrix)
        self.assertTrue(np.allclose(sep_matrix.A, sep_matrix_as_ndarray))

    def test_DCT(self):
        dct_mtx = magni.imaging.dictionaries.get_DCT_transform_matrix(self.N)
        dct_matrix_sep = magni.utils.matrices.Separable2DTransform(
            dct_mtx.T, dct_mtx.T)
        dct_matrix_fft = magni.imaging.dictionaries.get_DCT((self.N, self.N))

        sep_dct = dct_matrix_sep.T.dot(self.vec)
        fft_dct = dct_matrix_fft.T.dot(self.vec)

        sep_idct = dct_matrix_sep.dot(self.vec)
        fft_idct = dct_matrix_fft.dot(self.vec)

        self.assertTrue(np.allclose(sep_dct, fft_dct))
        self.assertTrue(np.allclose(sep_idct, fft_idct))

    def test_DFT(self):
        dft_mtx = magni.imaging.dictionaries.get_DFT_transform_matrix(self.N)
        dft_matrix_sep = magni.utils.matrices.Separable2DTransform(
            dft_mtx.conj().T, dft_mtx.conj().T)
        dft_matrix_fft = magni.imaging.dictionaries.get_DFT((self.N, self.N))

        sep_dft = dft_matrix_sep.conj().T.dot(self.vec)
        fft_dft = dft_matrix_fft.conj().T.dot(self.vec)

        sep_idft = dft_matrix_sep.dot(self.vec)
        fft_idft = dft_matrix_fft.dot(self.vec)

        self.assertTrue(np.allclose(sep_dft, fft_dft))
        self.assertTrue(np.allclose(sep_idft, fft_idft))

    def test_matrix_state(self):
        sep_matrix = magni.utils.matrices.Separable2DTransform(
            self.mtx, self.mtx)
        state = sep_matrix.matrix_state
        vec = np.arange(100).reshape(100, 1)

        self.assertTrue(np.allclose(
            np.kron(state['mtx_l'], state['mtx_r']).dot(vec),
            sep_matrix.dot(vec)))

        self.assertTrue(state['mtx_l'] is not self.mtx)
        self.assertTrue(state['mtx_r'] is not self.mtx)


class TestSRM(unittest.TestCase):
    """
    Test of Structurally Random Matrix (SRM)

    Implemented tests:

    * test_only_F : no sub-sampling and pre-randomization
    * test_DF : sub-sampling
    * test_DFR_l : sub-sampling and local pre-randomization
    * test_DFR_g : sub-sampling and global pre-randomizaton
    * test_full_SRM : sub-sampling and local and global pre-randomizaton
    * test_custom_validation : custom input validation
    * test_F_norm_property : the special property for norm computations
    * test_matrix_state : return internal matrix state

    """

    def setUp(self):
        np.random.seed(6021)

        n_sqrt = 3
        n = n_sqrt**2
        m = n_sqrt

        points = np.random.choice(n, size=m, replace=False)
        points.sort()
        coords = np.vstack([points // n_sqrt, points % n_sqrt]).T

        self.D = magni.imaging.measurements.construct_measurement_matrix(
            coords, n_sqrt, n_sqrt)
        self.F = magni.imaging.dictionaries.get_DCT((n_sqrt, n_sqrt))
        self.signs = np.random.choice([-1, 1], size=n)
        self.permutation = np.random.permutation(n)

        self.vec = np.random.randn(9, 1)

    def test_only_F(self):
        A = magni.utils.matrices.SRM(self.F)
        self.assertTrue(np.allclose(A.dot(self.vec), self.F.dot(self.vec)))
        self.assertDictEqual(
            A._includes, {'sub_sampling': False,
                          'local_pre_randomization': False,
                          'global_pre_randomization': False})

    def test_DF(self):
        A = magni.utils.matrices.SRM(self.F, D=self.D)
        self.assertTrue(
            np.allclose(A.dot(self.vec), self.D.dot(self.F.dot(self.vec))))
        self.assertDictEqual(
            A._includes, {'sub_sampling': True,
                          'local_pre_randomization': False,
                          'global_pre_randomization': False})

    def test_DFR_l(self):
        A = magni.utils.matrices.SRM(self.F, D=self.D, l_ran_arr=self.signs)
        F_sign = self.F.A.dot(np.diag(self.signs))

        self.assertTrue(
            np.allclose(A.dot(self.vec), self.D.dot(F_sign.dot(self.vec))))
        self.assertDictEqual(
            A._includes, {'sub_sampling': True,
                          'local_pre_randomization': True,
                          'global_pre_randomization': False})

    def test_DFR_g(self):
        A = magni.utils.matrices.SRM(
            self.F, D=self.D, g_ran_arr=self.permutation)
        F_permutation = np.zeros(self.F.shape)
        for k, permut in enumerate(self.permutation):
            F_permutation[:, k] = self.F.A[:, permut]

        self.assertTrue(
            np.allclose(A.dot(self.vec),
                        self.D.dot(F_permutation.dot(self.vec))))
        self.assertDictEqual(
            A._includes, {'sub_sampling': True,
                          'local_pre_randomization': False,
                          'global_pre_randomization': True})

    def test_full_SRM(self):
        A = magni.utils.matrices.SRM(
            self.F, D=self.D, l_ran_arr=self.signs, g_ran_arr=self.permutation)
        F_permutation = np.zeros(self.F.shape)
        for k, permut in enumerate(self.permutation):
            F_permutation[:, k] = self.F.A[:, permut]
        FR = F_permutation.dot(np.diag(self.signs))

        self.assertTrue(
            np.allclose(A.dot(self.vec),
                        self.D.dot(FR.dot(self.vec))))
        self.assertDictEqual(
            A._includes, {'sub_sampling': True,
                          'local_pre_randomization': True,
                          'global_pre_randomization': True})

        T_vec = np.arange(self.D.shape[0]).reshape(-1, 1).astype(float)
        self.assertTrue(
            np.allclose(A.T.dot(T_vec),
                        FR.T.dot(self.D.T.dot(T_vec))))

    def test_custom_validation(self):
        with self.assertRaises(ValueError):
            magni.utils.matrices.SRM(
                self.F, D=self.D,
                l_ran_arr=np.zeros(self.F.shape[0]).astype(int))

        with self.assertRaises(ValueError):
            magni.utils.matrices.SRM(
                self.F, D=self.D,
                g_ran_arr=np.ones(self.F.shape[0]).astype(int))

    def test_F_norm_property(self):
        A = magni.utils.matrices.SRM(self.F)
        self.assertTrue(A.matrix_state['F_norm'] is None)

        A.matrix_state = {'F_norm': self.F.A}
        self.assertTrue(np.allclose(
            A.matrix_state['F_norm'].dot(self.vec), A.dot(self.vec)))

        A.matrix_state = {'F_norm': None}
        self.assertTrue(A.matrix_state['F_norm'] is None)

        with self.assertRaises(KeyError):
            A.matrix_state = {'fail': 'fail'}

        with self.assertRaises(TypeError):
            A.matrix_state = {'F_norm': 'fail'}

    def test_matrix_state(self):
        A = magni.utils.matrices.SRM(
            self.F, D=self.D, l_ran_arr=self.signs, g_ran_arr=self.permutation)
        state = A.matrix_state

        self.assertTrue(np.allclose(
            magni.utils.matrices.MatrixCollection(
                state['matrices']).dot(self.vec),
            A.dot(self.vec)))
        self.assertTrue(np.allclose(state['l_ran_arr'], self.signs))
        self.assertTrue(np.allclose(state['g_ran_arr'], self.permutation))
        self.assertTrue(state['F_norm'] is None)
        self.assertEqual(state['includes'], {'sub_sampling': True,
                                             'local_pre_randomization': True,
                                             'global_pre_randomization': True})

        self.assertTrue(state['matrices'] is not A._matrices)
        self.assertTrue(state['l_ran_arr'] is not self.signs)
        self.assertTrue(state['g_ran_arr'] is not self.permutation)
        self.assertTrue(state['includes'] is not A._includes)


class TestNorm(unittest.TestCase):
    """
    Test of norm computation.

    Implemented tests:

    * test_Matrix : norm of Matrix
    * test_MatrixCollection : norm of MatrixCollection
    * test_Separable2DTransform_frobenius : Frobenius norm of 2D sep transform
    * test_SRM_frobenius : Frobenius norm of SRM

    """

    def setUp(self):
        self.N = 10
        np.random.seed(6021)

    def test_Matrix(self):
        func = lambda vec: -vec
        matrix = magni.utils.matrices.Matrix(func, func, (), (3, 3))
        norm = magni.utils.matrices.norm(matrix)
        self.assertEqual(norm, np.sqrt(3))

    def test_MatrixCollection(self):
        func1 = lambda vec: -vec
        matrix1 = magni.utils.matrices.Matrix(func1, func1, (), (3, 3))
        func2 = lambda vec: vec[::-1]
        matrix2 = magni.utils.matrices.Matrix(func2, func2, (), (3, 3))
        matrix_collection = magni.utils.matrices.MatrixCollection(
            (matrix1, matrix2))
        norm = magni.utils.matrices.norm(matrix_collection)
        self.assertEqual(norm, np.sqrt(3))

    def test_Separable2DTransform_frobenius(self):
        mtx = np.random.randn(self.N, self.N)
        sep_matrix = magni.utils.matrices.Separable2DTransform(mtx, mtx)
        ref_norm = np.linalg.norm(sep_matrix.A)

        # Offload to numpy
        np_norm = np.linalg.norm(sep_matrix)
        self.assertEqual(np_norm, ref_norm)

        # Default none == fro norm
        none_norm = magni.utils.matrices.norm(sep_matrix)
        self.assertAlmostEqual(none_norm, ref_norm)

        # Direct fro norm
        fro_norm = magni.utils.matrices.norm(sep_matrix, ord='fro')
        self.assertAlmostEqual(fro_norm, ref_norm)

    def test_SRM_frobenius(self):
        mtx = np.random.randn(self.N, self.N)
        F = magni.utils.matrices.Separable2DTransform(mtx, mtx)
        dct_mtx = magni.imaging.dictionaries.get_DCT_transform_matrix(self.N)
        F_dct = magni.utils.matrices.Separable2DTransform(
            dct_mtx.T, dct_mtx.T)
        signs = np.random.choice([-1, 1], size=F.shape[1])
        permutation = np.random.permutation(F.shape[1])
        D = np.eye(F.shape[0])[self.N*3:]
        SRM_1 = magni.utils.matrices.SRM(F, D=D, l_ran_arr=signs)
        SRM_1.matrix_state = {'F_norm': F}
        SRM_2 = magni.utils.matrices.SRM(F, g_ran_arr=permutation)
        SRM_3 = magni.utils.matrices.SRM(F_dct, D=D, g_ran_arr=permutation)
        SRM_3.matrix_state = {'F_norm': F_dct}
        SRM_4 = magni.utils.matrices.SRM(F_dct, l_ran_arr=signs)
        SRM_4.matrix_state = {'F_norm': F_dct}
        ref_norm_SRM_1 = np.linalg.norm(SRM_1.A)
        ref_norm_SRM_2 = np.linalg.norm(SRM_2.A)
        ref_norm_SRM_3 = np.linalg.norm(SRM_3.A)
        ref_norm_SRM_4 = np.linalg.norm(SRM_4.A)
        self.assertAlmostEqual(ref_norm_SRM_3, np.sqrt(D.shape[0]))
        self.assertAlmostEqual(ref_norm_SRM_4, np.sqrt(F_dct.shape[1]))

        # Offload to numpy
        np_norm_SRM_1 = np.linalg.norm(SRM_1)
        self.assertEqual(np_norm_SRM_1, ref_norm_SRM_1)
        np_norm_SRM_2 = np.linalg.norm(SRM_2)
        self.assertEqual(np_norm_SRM_2, ref_norm_SRM_2)
        np_norm_SRM_3 = np.linalg.norm(SRM_3)
        self.assertEqual(np_norm_SRM_3, ref_norm_SRM_3)
        np_norm_SRM_4 = np.linalg.norm(SRM_4)
        self.assertEqual(np_norm_SRM_4, ref_norm_SRM_4)

        # Default none == fro norm
        none_norm_SRM_1 = magni.utils.matrices.norm(SRM_1)
        self.assertNotAlmostEqual(none_norm_SRM_1, ref_norm_SRM_1)
        self.assertTrue(
            ref_norm_SRM_1 * 0.95 <= none_norm_SRM_1 <= ref_norm_SRM_1 * 1.05)
        none_norm_SRM_2 = magni.utils.matrices.norm(SRM_2)
        self.assertAlmostEqual(none_norm_SRM_2, ref_norm_SRM_2)
        none_norm_SRM_3 = magni.utils.matrices.norm(SRM_3)
        self.assertAlmostEqual(none_norm_SRM_3, ref_norm_SRM_3)
        none_norm_SRM_4 = magni.utils.matrices.norm(SRM_4)
        self.assertAlmostEqual(none_norm_SRM_4, ref_norm_SRM_4)

        # Direct fro norm
        fro_norm_SRM_1 = magni.utils.matrices.norm(SRM_1, ord='fro')
        self.assertEqual(fro_norm_SRM_1, none_norm_SRM_1)
        fro_norm_SRM_2 = magni.utils.matrices.norm(SRM_2, ord='fro')
        self.assertEqual(fro_norm_SRM_2, none_norm_SRM_2)
        fro_norm_SRM_3 = magni.utils.matrices.norm(SRM_3, ord='fro')
        self.assertEqual(fro_norm_SRM_3, none_norm_SRM_3)
        fro_norm_SRM_4 = magni.utils.matrices.norm(SRM_4, ord='fro')
        self.assertEqual(fro_norm_SRM_4, none_norm_SRM_4)
