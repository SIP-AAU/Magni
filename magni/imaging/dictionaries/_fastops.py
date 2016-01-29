"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality related to linear transformations.

Routine listings
----------------
dct2(x, mn_tuple, overcomplete_mn_tuple=None)
    2D discrete cosine transform.
idct2(x, mn_tuple, overcomplete_mn_tuple=None)
    2D inverse discrete cosine tranform.
dft2(x, mn_tuple, overcomplete_mn_tuple=None)
    2D discrete Fourier transform.
idft2(x, mn_tuple, overcomplete_mn_tuple=None)
    2D inverse discrete Fourier transform.

"""

from __future__ import division

import numpy as np
import scipy.fftpack

from magni.imaging._util import mat2vec as _mat2vec
from magni.imaging._util import vec2mat as _vec2mat
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


def dct2(x, mn_tuple, overcomplete_mn_tuple=None):
    """
    Apply the 2D Discrete Cosine Transform (DCT).

    `x` is assumed to be the column vector resulting from stacking the columns
    of the associated matrix which the transform is to be taken on.

    Parameters
    ----------
    x : ndarray
        The m*n x 1 vector representing the associated column stacked matrix.
    mn_tuple : tuple of int
        `(m, n)` - `m` number of rows in the matrix represented by `x`; `n`
        number of columns in the matrix.
    overcomplete_mn_tuple : tuple of int, optional
        `(mo, no)` - `mo` number of rows in the matrix represented by the
        function's output; `no` number of columns in the matrix.

    Returns
    -------
    alpha : ndarray
        When `overcomplete_mn_tuple` is omitted: an m*n x 1 vector of
        coefficients scaled such that x = idct2(dct2(x)). When
        `overcomplete_mn_tuple` is supplied: an mo*no x 1 vector of
        coefficients scaled such that x = idct2(dct2(x)).

    See Also
    --------
    scipy.fftpack.dct : 1D DCT

    Notes
    -----
    The overcomplete transform feature invoked by supplying
    `overcomplete_mn_tuple` is only available for SciPy >= 0.16.1.

    The 2D DCT uses the seperability property of the 1D DCT.
    http://stackoverflow.com/questions/14325795/scipys-fftpack-dct-and-idct
    Including the reshape operation, the full transform is:
    dct(dct(x.reshape(n, m).T).T).T.T.reshape(m * n, 1)

    """

    @_decorate_validation
    def validate_input():
        _validate_transform_fwd(x, mn_tuple, overcomplete_mn_tuple)

    validate_input()

    m, n = mn_tuple
    result = _vec2mat(x, (m, n))

    if overcomplete_mn_tuple is None:
        result = scipy.fftpack.dct(result, norm='ortho').T
        result = scipy.fftpack.dct(result, norm='ortho').T
    else:
        mo, no = overcomplete_mn_tuple
        result = scipy.fftpack.dct(result, norm='ortho', n=no).T
        result = scipy.fftpack.dct(result, norm='ortho', n=mo).T

    return _mat2vec(result)


def idct2(x, mn_tuple, overcomplete_mn_tuple=None):
    """
    Apply the 2D Inverse Discrete Cosine Transform (iDCT).

    `x` is assumed to be the column vector resulting from stacking the columns
    of the associated matrix which the transform is to be taken on.

    Parameters
    ----------
    x : ndarray
        The vector representing the associated column-stacked matrix. When
        `overcomplete_mn_tuple` is omitted: the shape must be m*n x 1. When
        `overcomplete_mn_tuple` is supplied: the shape must be mo*no x 1.
    mn_tuple : tuple of int
        `(m, n)` - `m` number of rows in the associated matrix, `n` number of
        columns in the associated matrix. When `overcomplete_mn_tuple` is
        supplied, this is the shape of the function's output.
    overcomplete_mn_tuple : tuple of int, optional
        `(mo, no)` - `mo` number of rows in the associated matrix, `no` number
        of columns in the associated matrix. When supplied, this is the shape
        of the function's input.

    Returns
    -------
    ndarray
        An m*n x 1 vector of coefficients scaled such that x = dct2(idct2(x)).

    See Also
    --------
    scipy.fftpack.idct : 1D inverse DCT

    Notes
    -----
    The overcomplete transform feature invoked by supplying
    `overcomplete_mn_tuple` is only available for SciPy >= 0.16.1.

    """

    @_decorate_validation
    def validate_input():
        _validate_transform_bwd(x, mn_tuple, overcomplete_mn_tuple)

    validate_input()

    m, n = mn_tuple

    if overcomplete_mn_tuple is None:
        result = _vec2mat(x, (m, n))
        result = scipy.fftpack.idct(result, norm='ortho').T
        result = scipy.fftpack.idct(result, norm='ortho').T
    else:
        mo, no = overcomplete_mn_tuple
        result = _vec2mat(x, (mo, no))
        result = scipy.fftpack.idct(result, norm='ortho').T
        result = result[:n, :]
        result = scipy.fftpack.idct(result, norm='ortho').T
        result = result[:m, :]

    return _mat2vec(result)


def dft2(x, mn_tuple, overcomplete_mn_tuple=None):
    """
    Apply the 2D Discrete Fourier Transform (DFT).

    `x` is assumed to be the column vector resulting from stacking the columns
    of the associated matrix which the transform is to be taken on.

    Parameters
    ----------
    x : ndarray
        The m*n x 1 vector representing the associated column stacked matrix.
    mn_tuple : tuple of int
        `(m, n)` - `m` number of rows in the matrix represented by `x`; `n`
        number of columns in the matrix.
    overcomplete_mn_tuple : tuple of int, optional
        `(mo, no)` - `mo` number of rows in the matrix represented by the
        function's output; `no` number of columns in the matrix.

    Returns
    -------
    ndarray
        When `overcomplete_mn_tuple` is omitted: an m*n x 1 vector of
        coefficients scaled such that x = idft2(dft2(x)). When
        `overcomplete_mn_tuple` is supplied: an mo*no x 1 vector of
        coefficients scaled such that x = idft2(dft2(x)).

    See Also
    --------
    numpy.fft.fft2 : The underlying 2D FFT used to compute the 2D DFT.

    Notes
    -----
    This is a normalised DFT, i.e. a normalisation constant of
    :math:`\sqrt{m * n}` is used.

    """

    @_decorate_validation
    def validate_input():
        _validate_transform_fwd(x, mn_tuple, overcomplete_mn_tuple)

    validate_input()

    m, n = mn_tuple
    result = _vec2mat(x, (m, n))

    if overcomplete_mn_tuple is None:
        result = 1 / np.sqrt(n * m) * np.fft.fft2(result)
    else:
        mo, no = overcomplete_mn_tuple
        result = 1 / np.sqrt(no * mo) * np.fft.fft2(result, s=(mo, no))

    return _mat2vec(result)


def idft2(x, mn_tuple, overcomplete_mn_tuple=None):
    """
    Apply the 2D Inverse Discrete Fourier Transform (iDFT).

    `x` is assumed to be the column vector resulting from stacking the columns
    of the associated matrix which the transform is to be taken on.

    Parameters
    ----------
    x : ndarray
        The m*n x 1 vector representing the associated column stacked matrix.
    mn_tuple : tuple of int
        `(m, n)` - `m` number of rows in the associated matrix, `n` number of
        columns in the associated matrix. When `overcomplete_mn_tuple`
        is supplied, this is the shape of the function's output.
    overcomplete_mn_tuple : tuple of int, optional
        `(mo, no)` - `mo` number of rows in the associated matrix, `no` number
        of columns in the associated matrix. When supplied, this is the shape
        of the function's input.

    Returns
    -------
    ndarray
        An m*n x 1 vector of coefficients scaled such that x = dft2(idft2(x)).

    See Also
    --------
    numpy.fft.ifft2 : The underlying 2D iFFT used to compute the 2D iDFT.

    Notes
    -----
    This is a normalised iDFT, i.e. a normalisation constant of
    :math:`\sqrt{m * n}` is used.

    """

    @_decorate_validation
    def validate_input():
        _validate_transform_bwd(x, mn_tuple, overcomplete_mn_tuple)

    validate_input()

    m, n = mn_tuple

    if overcomplete_mn_tuple is None:
        result = _vec2mat(x, (m, n))
        result = np.sqrt(n * m) * np.fft.ifft2(result)
    else:
        mo, no = overcomplete_mn_tuple
        result = _vec2mat(x, (mo, no))
        result = np.sqrt(no * mo) * np.fft.ifft2(result)
        result = result[:m, :n]

    if np.allclose(result.imag, 0):
        result = result.real

    return _mat2vec(result)


def _validate_transform_fwd(x, mn_tuple, overcomplete_mn_tuple=None):
    """
    Validate a 2D transform.

    Parameters
    ----------
    x : ndarray
        The m*n x 1 vector representing the associated column stacked matrix.
    mn_tuple : tuple of int
        `(m, n)` - `m` number of rows in the matrix represented by `x`; `n`
        number of columns in the matrix.
    overcomplete_mn_tuple : tuple of int, optional
        `(mo, no)` - `mo` number of rows in the matrix represented by the
        function's output; `no` number of columns in the matrix.

    """

    _levels('mn_tuple', (
        _generic(None, 'explicit collection', len_=2),
        _numeric(None, 'integer', range_='[1;inf)')))
    m, n = mn_tuple
    _numeric('x', ('integer', 'floating', 'complex'), shape=(m * n, 1))

    if overcomplete_mn_tuple is not None:
        _generic('overcomplete_mn_tuple', 'explicit collection', len_=2)
        _numeric(('overcomplete_mn_tuple', 0), 'integer',
                 range_='[{};inf)'.format(m))
        _numeric(('overcomplete_mn_tuple', 1), 'integer',
                 range_='[{};inf)'.format(n))


def _validate_transform_bwd(x, mn_tuple, overcomplete_mn_tuple=None):
    """
    Validate a 2D transform.

    Parameters
    ----------
    x : ndarray
        The m*n x 1 vector representing the associated column stacked matrix.
    mn_tuple : tuple of int
        `(m, n)` - `m` number of rows in the associated matrix, `n` number of
        columns in the associated matrix.
    overcomplete_mn_tuple : tuple of int, optional
        `(mo, no)` - `mo` number of rows in the associated matrix, `no` number
        of columns in the associated matrix.

    """

    _levels('mn_tuple', (
        _generic(None, 'explicit collection', len_=2),
        _numeric(None, 'integer', range_='[1;inf)')))
    m, n = mn_tuple

    if overcomplete_mn_tuple is None:
        shape = (m * n, 1)
    else:
        _generic('overcomplete_mn_tuple', 'explicit collection', len_=2)
        _numeric(('overcomplete_mn_tuple', 0), 'integer',
                 range_='[{};inf)'.format(m))
        _numeric(('overcomplete_mn_tuple', 1), 'integer',
                 range_='[{};inf)'.format(n))
        shape = (overcomplete_mn_tuple[0] * overcomplete_mn_tuple[1], 1)

    _numeric('x', ('integer', 'floating', 'complex'), shape=shape)
