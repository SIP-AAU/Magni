"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing 1D matrices for building 2D separable transforms.

Routine listings
----------------
get_DCT_transform_matrix(N)
    Return the normalised N-by-N discrete cosine transform (DCT) matrix.
get_DFT_transform_matrix(N)
    Return the normalised N-by-N discrete fourier transform (DFT) matrix.

"""

from __future__ import division

import numpy as np
import scipy.linalg

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


def get_DCT_transform_matrix(N):
    """
    Return the normalised N-by-N discrete cosine transform (DCT) matrix.

    Applying the returned transform matrix to a vector x: D.dot(x) yields the
    DCT of x. Applying the returned transform matrix to a matrix A: D.dot(A)
    applies the DCT to the columns of A. Taking D.dot(A.dot(D.T)) applies the
    DCT to both columns and rows, i.e. a full 2D separable DCT transform. The
    inverse transform (the 1D IDCT) is D.T.

    Parameters
    ----------
    N : int
        The size of the DCT transform matrix to return.

    Returns
    -------
    D : ndarray
        The DCT transform matrix.

    Notes
    -----
    The returned DCT matrix normalised such that is consitutes a orthonormal
    transform as given by equations (2.119) and (2.120) in [1]_.

    References
    ----------
    .. [1] A.N. Akansu, R.A. Haddad, and P.R. Haddad, *Multiresolution Signal
       Decomposition: Transforms, Subbands, and Wavelets*, Academic Press,
       2000.

    Examples
    --------
    For example, get a 5-by-5 DCT matrix

    >>> import numpy as np
    >>> from magni.imaging.dictionaries import get_DCT_transform_matrix
    >>> D = get_DCT_transform_matrix(5)
    >>> np.round(np.abs(D), 4)
    array([[ 0.4472,  0.4472,  0.4472,  0.4472,  0.4472],
           [ 0.6015,  0.3717,  0.    ,  0.3717,  0.6015],
           [ 0.5117,  0.1954,  0.6325,  0.1954,  0.5117],
           [ 0.3717,  0.6015,  0.    ,  0.6015,  0.3717],
           [ 0.1954,  0.5117,  0.6325,  0.5117,  0.1954]])

    and apply the 2D DCT transform to a dummy image

    >>> np.random.seed(6021)
    >>> img = np.random.randn(5, 5)
    >>> img_dct = D.dot(img.dot(D.T))
    >>> np.round(img_dct, 4)
    array([[-0.5247, -0.0225,  0.9098,  0.369 , -0.477 ],
           [ 1.7309, -0.4142,  1.9455, -0.6726, -1.3676],
           [ 0.6987,  0.5355,  0.7213, -0.8498, -0.1023],
           [ 0.0078, -0.0545,  0.3649, -1.4694,  1.732 ],
           [-1.5864,  0.156 ,  0.8932, -0.8091,  0.5056]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('N', 'integer', range_='[1;inf)')

    validate_input()

    nn, rr = np.meshgrid(*map(np.arange, (N, N)))

    D = np.cos((2 * nn + 1) * rr * np.pi / (2 * N))
    D[0, :] /= np.sqrt(N)
    D[1:, :] /= np.sqrt(N/2)

    return D


def get_DFT_transform_matrix(N):
    """
    Return the normalised N-by-N discrete fourier transform (DFT) matrix.

    Applying the returned transform matrix to a vector x: D.dot(x) yields the
    DFT of x. Applying the returned transform matrix to a matrix A: D.dot(A)
    applies the DFT to the columns of A. Taking D.dot(A.dot(D.T)) applies the
    DFT to both columns and rows, i.e. a full 2D separable DFT transform. The
    inverse transform (the 1D IDFT) is D.T.

    Parameters
    ----------
    N : int
        The size of the DFT transform matrix to return.

    Returns
    -------
    D : ndarray
        The DFT transform matrix.

    See Also
    --------
    scipy.linalg.dft : The function used to generate the DFT transform matrix.

    Notes
    -----
    The returned DFT matrix normalised such that is consitutes a orthonormal
    transform as given by equations (2.105) and (2.109) in [2]_.

    References
    ----------
    .. [2] A.N. Akansu, R.A. Haddad, and P.R. Haddad, *Multiresolution Signal
       Decomposition: Transforms, Subbands, and Wavelets*, Academic Press,
       2000.

    Examples
    --------
    For example, get a 5-by-5 DFT matrix

    >>> import numpy as np, scipy.fftpack
    >>> from magni.imaging.dictionaries import get_DFT_transform_matrix
    >>> D = get_DFT_transform_matrix(5)
    >>> np.round(D, 2)
    array([[ 0.45+0.j  ,  0.45+0.j  ,  0.45+0.j  ,  0.45+0.j  ,  0.45+0.j  ],
           [ 0.45+0.j  ,  0.14-0.43j, -0.36-0.26j, -0.36+0.26j,  0.14+0.43j],
           [ 0.45+0.j  , -0.36-0.26j,  0.14+0.43j,  0.14-0.43j, -0.36+0.26j],
           [ 0.45+0.j  , -0.36+0.26j,  0.14-0.43j,  0.14+0.43j, -0.36-0.26j],
           [ 0.45+0.j  ,  0.14+0.43j, -0.36+0.26j, -0.36-0.26j,  0.14-0.43j]])

    and apply the 2D DFT transform to a dummy image

    >>> np.random.seed(6021)
    >>> img = np.random.randn(5, 5)
    >>> img_dft = D.dot(img.dot(D.T))
    >>> np.round(img_dft, 2)
    array([[-0.52+0.j  ,  0.44+0.48j,  0.11-0.39j,  0.11+0.39j,  0.44-0.48j],
           [ 1.04-0.59j,  1.32+0.13j, -1.20-0.39j,  0.35+0.66j,  0.19-0.36j],
           [ 0.18-1.24j,  0.75+0.44j, -0.75+0.72j, -0.52-0.8j ,  0.77+0.13j],
           [ 0.18+1.24j,  0.77-0.13j, -0.52+0.8j , -0.75-0.72j,  0.75-0.44j],
           [ 1.04+0.59j,  0.19+0.36j,  0.35-0.66j, -1.20+0.39j,  1.32-0.13j]])

    which may be shifted to have the zero-frequency component at the center of
    the spectrum

    >>> np.round(scipy.fftpack.fftshift(img_dft), 2)
    array([[-0.75-0.72j,  0.75-0.44j,  0.18+1.24j,  0.77-0.13j, -0.52+0.8j ],
           [-1.20+0.39j,  1.32-0.13j,  1.04+0.59j,  0.19+0.36j,  0.35-0.66j],
           [ 0.11+0.39j,  0.44-0.48j, -0.52+0.j  ,  0.44+0.48j,  0.11-0.39j],
           [ 0.35+0.66j,  0.19-0.36j,  1.04-0.59j,  1.32+0.13j, -1.20-0.39j],
           [-0.52-0.8j ,  0.77+0.13j,  0.18-1.24j,  0.75+0.44j, -0.75+0.72j]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('N', 'integer', range_='[1;inf)')

    validate_input()

    D = scipy.linalg.dft(N, scale='sqrtn')

    return D
