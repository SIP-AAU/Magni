"""
..
    Copyright (c) 2016, Magni developers.
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


class TestNorm(unittest.TestCase):
    """
    Test of norm computation.

    Implemented tests:

    * test_Matrix : norm of Matrix
    * test_MatrixCollection : norm of MatrixCollection
    * test_Separable2DTransform_frobenius : Frobenius norm of 2D sep transform

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
