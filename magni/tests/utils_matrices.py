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

    """

    def test_asarray(self):
        func = lambda vec: -vec
        matrix = magni.utils.matrices.Matrix(func, func, (), (3, 3))
        print(matrix.A)
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


class TestMatrixCollection(unittest.TestCase):
    """
    Test of MatrixCollection.

    Implemented tests:

    * test_asarray : transforming to an explicit ndarray
    * test_incompatible_matrices : matrices with incompatible shapes

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
