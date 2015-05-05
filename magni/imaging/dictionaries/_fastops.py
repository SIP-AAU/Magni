"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality related to linear transformations.

Routine listings
----------------
dct2(x, m, n)
    2D discrete cosine transform.
idct2(x, m, n)
    2D inverse discrete cosine tranform.
dft2(x, m, n)
    2D discrete Fourier transform.
idft2(x, m, n)
    2D inverse discrete Fourier transform.

"""

from __future__ import division

import numpy as np
import scipy.fftpack

from magni.imaging._util import mat2vec as _mat2vec
from magni.imaging._util import vec2mat as _vec2mat
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


def dct2(x, mn_tuple):
    """
    Apply the 2D Discrete Cosine Transform (DCT) to `x`.

    `x` is assumed to be the column vector resulting from stacking the columns
    of the associated matrix which the transform is to be taken on.

    Parameters
    ----------
    x : ndarray
        The m*n x 1 vector representing the associated column stacked matrix.
    m : int
        Number of rows in the associated matrix.
    n : int
        Number of columns in the associated matrix.

    Returns
    -------
    ndarray
        A m*n x 1 vector of coefficients scaled such that x = idct2(dct2(x)).

    See Also
    --------
    scipy.fftpack.dct : 1D DCT

    """

    m, n = mn_tuple

    @_decorate_validation
    def validate_input():
        _validate_transform(x, m, n)

    validate_input()

    # 2D DCT using the seperability property of the 1D DCT.
    # http://stackoverflow.com/questions/14325795/scipys-fftpack-dct-and-idct
    # Including the reshape operation, the full transform is:
    # dct(dct(x.reshape(n, m).T).T).T.T.reshape(m * n, 1)

    result = _vec2mat(x, (m, n))
    result = scipy.fftpack.dct(result, norm='ortho').T
    result = scipy.fftpack.dct(result, norm='ortho').T
    return _mat2vec(result)


def idct2(x, mn_tuple):
    """
    Apply the 2D Inverse Discrete Cosine Transform (iDCT) to `x`.

    `x` is assumed to be the column vector resulting from stacking the columns
    of the associated matrix which the transform is to be taken on.

    Parameters
    ----------
    x : ndarray
        The m*n x 1 vector representing the associated column stacked matrix.
    m : int
        Number of rows in the associated matrix.
    n : int
        Number of columns in the associated matrix.

    Returns
    -------
    ndarray
        A m*n x 1 vector of coefficients scaled such that x = dct2(idct2(x)).

    See Also
    --------
    scipy.fftpack.idct : 1D inverse DCT

    """

    m, n = mn_tuple

    @_decorate_validation
    def validate_input():
        _validate_transform(x, m, n)

    validate_input()

    result = _vec2mat(x, (m, n))
    result = scipy.fftpack.idct(result, norm='ortho').T
    result = scipy.fftpack.idct(result, norm='ortho').T
    return _mat2vec(result)


def dft2(x, mn_tuple):
    """
    Apply the 2D Discrete Fourier Transform (DFT) to `x`.

    `x` is assumed to be the column vector resulting from stacking the columns
    of the associated matrix which the transform is to be taken on.

    Parameters
    ----------
    x : ndarray
        The m*n x 1 vector representing the associated column stacked matrix.
    m : int
        Number of rows in the associated matrix.
    n : int
        Number of columns in the associated matrix.

    Returns
    -------
    ndarray
        A m*n x 1 vector of coefficients scaled such that x = dft2(idft2(x)).

    See Also
    --------
    numpy.fft.fft2 : The underlying 2D FFT used to compute the 2D DFT.

    Notes
    -----
    This is a normalised DFT, i.e. a normalisation constant of
    :math:`\sqrt{m * n}` is used.

    """

    m, n = mn_tuple

    @_decorate_validation
    def validate_input():
        _validate_transform(x, m, n)

    validate_input()

    return _mat2vec(1 / np.sqrt(n * m) * np.fft.fft2(_vec2mat(x, (m, n))))


def idft2(x, mn_tuple):
    """
    Apply the 2D Inverse Discrete Fourier Transform (iDFT) to `x`.

    `x` is assumed to be the column vector resulting from stacking the columns
    of the associated matrix which the transform is to be taken on.

    Parameters
    ----------
    x : ndarray
        The m*n x 1 vector representing the associated column stacked matrix.
    m : int
        Number of rows in the associated matrix.
    n : int
        Number of columns in the associated matrix.

    Returns
    -------
    ndarray
        A m*n x 1 vector of coefficients scaled such that x = idft2(dft2(x)).

    See Also
    --------
    numpy.fft.ifft2 : The underlying 2D iFFT used to compute the 2D iDFT.

    Notes
    -----
    This is a normalised iDFT, i.e. a normalisation constant of
    :math:`\sqrt{m * n}` is used.

    """

    m, n = mn_tuple

    @_decorate_validation
    def validate_input():
        _validate_transform(x, m, n)

    validate_input()

    output = _mat2vec(np.sqrt(n * m) * np.fft.ifft2(_vec2mat(x, (m, n))))

    if np.allclose(output.imag, 0):
        output = output.real

    return output


def _validate_transform(x, m, n):
    """
    Validatate a 2D transform.

    """

    _numeric('m', 'integer', range_='[1;inf)')
    _numeric('n', 'integer', range_='[1;inf)')
    _numeric('x', ('integer', 'floating', 'complex'), shape=(m * n, 1))
