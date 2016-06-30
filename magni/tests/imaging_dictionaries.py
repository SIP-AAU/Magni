"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests for `magni.imaging.dictionaries`.

**Testing Strategy**

This test set-up primarily serves to verify the overcomplete transform
capabilities implemented in the above module. The following operations are
tested:

* Whether a vector transformed both back and forth ends up identical to the
  original vector
* A known vector transformed to the secondary domain is identical to the known
  correct result

These cases are tested with 2D transforms.

"""

from __future__ import division
import unittest

import numpy as np
from pkg_resources import parse_version as _parse_version
from scipy import __version__ as _scipy_version

from magni.imaging.dictionaries import get_DCT
from magni.imaging.dictionaries import get_DFT
from magni.imaging import mat2vec
from magni.imaging import vec2mat


_scipy_pre_016 = _parse_version(_scipy_version) < _parse_version('0.16.0')


class TransformsMixin(object):
    """
    Test of overcomplete transforms on a square matrix.

    """

    @unittest.skipIf(_scipy_pre_016, "Not supported for SciPy <= 0.16.0")
    def test_wrong_size_dct(self):
        # Check that the function does not allow specifying an
        # _under_-complete transform
        with self.assertRaises(ValueError):
            matrix = get_DCT(self.array_shape, (self.array_shape[0] - 1,
                                                self.array_shape[1] - 1))

    def test_wrong_size_dft(self):
        # Check that the function does not allow specifying an
        # _under_-complete transform
        with self.assertRaises(ValueError):
            matrix = get_DFT(self.array_shape, (self.array_shape[0] - 1,
                                                self.array_shape[1] - 1))

    @unittest.skipIf(_scipy_pre_016, "Not supported for SciPy <= 0.16.0")
    def test_roundtrip_dct(self):
        # Test DCT
        dct_mtx = get_DCT(self.array_shape, self.array_shape_oc)
        d2_array_dct = dct_mtx.T.dot(mat2vec(self.d2_array))
        d2_array_roundtrip = vec2mat(dct_mtx.dot(d2_array_dct),
                                     self.array_shape)

        # Does the DCT array have the expected shape?
        self.assertSequenceEqual(d2_array_dct.shape,
                                 (self.array_shape_oc[0] *
                                  self.array_shape_oc[1], 1))
        # Is the result identical to the original?
        self.assertTrue(np.allclose(self.d2_array,
                                    d2_array_roundtrip))

    def test_roundtrip_dft(self):
        # Test DFT
        dft_mtx = get_DFT(self.array_shape,
                          self.array_shape_oc)
        d2_array_dft = dft_mtx.conj().T.dot(mat2vec(self.d2_array))
        d2_array_roundtrip = vec2mat(dft_mtx.dot(d2_array_dft),
                                     self.array_shape)

        # Does the DFT array have the expected shape?
        self.assertSequenceEqual(d2_array_dft.shape,
                                 (self.array_shape_oc[0] *
                                  self.array_shape_oc[1], 1))
        # Is the result identical to the original?
        self.assertTrue(np.allclose(self.d2_array,
                                    d2_array_roundtrip))

    @unittest.skipIf(_scipy_pre_016, "Not supported for SciPy <= 0.16.0")
    def test_forward_dct(self):
        # Manually build the DCT-II transform matrices
        # according to the documentation for scipy.fftpack.dct

        idx_freq_0 = np.arange(self.array_shape_oc[0]).reshape((-1, 1))
        idx_freq_1 = np.arange(self.array_shape_oc[1]).reshape((-1, 1))
        idx_space_0 = np.arange(self.array_shape_oc[0]).reshape((1, -1))
        idx_space_1 = np.arange(self.array_shape_oc[1]).reshape((1, -1))

        coeff_mtx_0 = np.pi * idx_freq_0.dot(
            (2 * idx_space_0 + 1)/(2*self.array_shape_oc[0]))
        dct_mtx_0 = 2 * np.cos(coeff_mtx_0)
        # Due to norm='ortho'
        dct_mtx_0[0, :] = (np.sqrt(1 / (4 * self.array_shape_oc[0])) *
                           dct_mtx_0[0, :])
        dct_mtx_0[1:, :] = (np.sqrt(1 / (2 * self.array_shape_oc[0])) *
                            dct_mtx_0[1:, :])

        coeff_mtx_1 = np.pi * idx_freq_1.dot(
            (2 * idx_space_1 + 1) / (2 * self.array_shape_oc[1]))
        dct_mtx_1 = 2 * np.cos(coeff_mtx_1)
        # Due to norm='ortho'
        dct_mtx_1[0, :] = (np.sqrt(1 / (4 * self.array_shape_oc[1])) *
                           dct_mtx_1[0, :])
        dct_mtx_1[1:, :] = (np.sqrt(1 / (2 * self.array_shape_oc[1])) *
                            dct_mtx_1[1:, :])

        # Compute the reference DCT transform
        reference = dct_mtx_0.dot(np.pad(self.d2_array,
                                         ((0, self.array_shape_oc[0] -
                                           self.array_shape[0]),
                                          (0, 0)), 'constant')).T
        reference = dct_mtx_1.dot(np.pad(reference,
                                         ((0, self.array_shape_oc[1] -
                                           self.array_shape[1]),
                                          (0, 0)),  'constant')).T

        # Compute the DCT transform by the tested function
        dct_mtx = get_DCT(self.array_shape, self.array_shape_oc)
        d2_array_dct = vec2mat(dct_mtx.T.dot(mat2vec(self.d2_array)),
                               self.array_shape_oc)

        # Is the result identical to the reference?
        self.assertTrue(np.allclose(reference, d2_array_dct))

    def test_forward_dft(self):
        # Manually build the DFT transform matrices
        # according to https://en.wikipedia.org/wiki/Discrete_Fourier_transform

        idx_freq_0 = np.arange(self.array_shape_oc[0]).reshape((-1, 1))
        idx_freq_1 = np.arange(self.array_shape_oc[1]).reshape((-1, 1))
        idx_space_0 = np.arange(self.array_shape_oc[0]).reshape((1, -1))
        idx_space_1 = np.arange(self.array_shape_oc[1]).reshape((1, -1))

        coeff_mtx_0 = -2 * np.pi * 1j * idx_freq_0.dot(idx_space_0 /
                                                       self.array_shape_oc[0])
        dft_mtx_0 = np.sqrt(1 / self.array_shape_oc[0]) * np.exp(coeff_mtx_0)

        coeff_mtx_1 = -2 * np.pi * 1j * idx_freq_1.dot(idx_space_1 /
                                                       self.array_shape_oc[1])
        dft_mtx_1 = np.sqrt(1 / self.array_shape_oc[1]) * np.exp(coeff_mtx_1)

        # Compute the reference DFT transform
        reference = dft_mtx_0.dot(np.pad(self.d2_array,
                                         ((0, self.array_shape_oc[0] -
                                           self.array_shape[0]),
                                          (0, 0)), 'constant')).T
        reference = dft_mtx_1.dot(np.pad(reference,
                                         ((0, self.array_shape_oc[1] -
                                           self.array_shape[1]),
                                          (0, 0)),  'constant')).T

        # Compute the DFT transform by the tested function
        dft_mtx = get_DFT(self.array_shape, self.array_shape_oc)
        d2_array_dft = vec2mat(dft_mtx.conj().T.dot(mat2vec(self.d2_array)),
                               self.array_shape_oc)

        # Is the result identical to the reference?
        self.assertTrue(np.allclose(reference, d2_array_dft))


class TestTransformSquare(unittest.TestCase, TransformsMixin):
    """
    Test of overcomplete transforms on a square matrix.

    """

    def setUp(self):
        n = 13

        self.d2_array = np.arange(n**2).reshape((n, n))
        self.array_shape = self.d2_array.shape
        self.array_shape_oc = (2*n, 2*n)


class TestTransformNonSquare(unittest.TestCase, TransformsMixin):
    """
    Test of overcomplete transforms on a non-square transform.

    """

    def setUp(self):
        m, n = 17, 23

        self.d2_array = np.arange(m*n).reshape((m, n))
        self.array_shape = self.d2_array.shape
        self.array_shape_oc = (2*m, 2*n)
