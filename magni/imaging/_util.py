"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the public functions of the magni.imaging subpackage.

"""

from __future__ import division

import numpy as np

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


def double_mirror(img, fftstyle=False):
    """
    Mirror image in both the vertical and horisontal axes.

    The image is mirrored around its upper left corner first in the horizontal
    axis and then in the vertical axis such that an image of four times the
    size of the original is returned. If `fftstyle` is True, the image is
    constructed such it would represent a fftshifted version of the mirrored
    `img` such that entry (0, 0) is the DC component.

    Parameters
    ----------
    img : ndarray
        The image to mirror.
    fftstyle : bool
        The flag that indicates if the fftstyle mirrored image is returned.

    Returns
    -------
    mirrored_img : ndarray
        The mirrored image.

    Examples
    --------
    For example, mirror a very simple 2-by-3 pixel image.

    >>> import numpy as np
    >>> from magni.imaging._util import double_mirror
    >>> img = np.arange(6).reshape(2, 3)
    >>> img
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> double_mirror(img)
    array([[5, 4, 3, 3, 4, 5],
           [2, 1, 0, 0, 1, 2],
           [2, 1, 0, 0, 1, 2],
           [5, 4, 3, 3, 4, 5]])
    >>> double_mirror(img, fftstyle=True)
    array([[0, 0, 0, 0, 0, 0],
           [0, 5, 4, 3, 4, 5],
           [0, 2, 1, 0, 1, 2],
           [0, 5, 4, 3, 4, 5]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', ('boolean', 'integer', 'floating', 'complex'),
                 shape=(-1, -1))
        _numeric('fftstyle', 'boolean')

    validate_input()

    if fftstyle:
        mirrored_img = np.zeros((img.shape[0] * 2, img.shape[1] * 2),
                                dtype=img.dtype)
        tmp = np.pad(img, ((img.shape[0] - 1, 0), (img.shape[1] - 1, 0)),
                     mode='reflect')
        mirrored_img[1:, 1:] = tmp
    else:
        mirrored_img = np.pad(img, ((img.shape[0], 0), (img.shape[1], 0)),
                              mode='symmetric')

    return mirrored_img


def get_inscribed_masks(img, as_vec=False):
    """
    Return a set of inscribed masks covering the image.

    Two masks are returned. One is the disc with radius equal to that of the
    inscribed circle for `img`. The other is the inscribed square of the first
    mask. If `as_vec` is True, the `img` must be a vector representation of the
    (matrix) image. In this case, the masks are also returned in vector
    representation.

    Parameters
    ----------
    img : ndarray
        The square image of even height/width which the masks should cover.
    as_vec : bool
        The indicator of whether or not to treat `img` as a vector instead of
        an image (the default is False, which implies that `img` is treated as
        a matrix.

    Returns
    -------
    cicle_mask : ndarray
        The inscribed cicle mask.
    square_mask : ndarray
        The inscribed square mask.

    Examples
    --------
    For example, get the inscribed masks of an 8-by-8 image:

    >>> import numpy as np
    >>> from magni.imaging._util import get_inscribed_masks, mat2vec
    >>> img = np.arange(64).reshape(8, 8)
    >>> circle_mask, square_mask = get_inscribed_masks(img)
    >>> np_printoptions = np.get_printoptions()
    >>> np.set_printoptions(formatter={'bool': lambda x: str(int(x))})
    >>> circle_mask
    array([[0, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [0, 1, 1, 1, 1, 1, 1, 0]], dtype=bool)

    >>> square_mask
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    Or get the same masks based on a vector:

    >>> img_vec = mat2vec(img)
    >>> c_vec_mask, s_vec_mask = get_inscribed_masks(img_vec, as_vec=True)
    >>> c_vec_mask
    array([[0],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [0],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [0],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [0]], dtype=bool)
    >>> np.set_printoptions(**np_printoptions)

    """

    @_decorate_validation
    def validate_input():
        _numeric('as_vec', 'boolean')

        if not as_vec:
            _numeric('img', ('boolean', 'integer', 'floating', 'complex'),
                     shape=(-1, -1))
            if img.shape[0] != img.shape[1]:
                raise ValueError('The input image must be square')

            if np.mod(img.shape[0], 2) != 0:
                raise ValueError('The image height/width must be an even' +
                                 ' number of pixels')
        else:
            _numeric('img', ('boolean', 'integer', 'floating', 'complex'),
                     shape=(-1, 1))
            if not np.allclose(np.mod(np.sqrt(img.size), 1), 0):
                raise ValueError('The input image must be square')

            if np.mod(np.sqrt(img.shape[0]), 2) != 0:
                raise ValueError('The image height/width must be an even' +
                                 ' number of pixels')

    validate_input()

    if as_vec:
        r = np.sqrt(img.shape[0]) / 2
    else:
        r = img.shape[0] / 2

    x, y = np.meshgrid(*map(np.arange, (r, r)))

    circle_mask = double_mirror(x**2 + y**2 <= r**2)
    square_mask = double_mirror(np.maximum(x, y) <= r / np.sqrt(2))

    if as_vec:
        return mat2vec(circle_mask), mat2vec(square_mask)
    else:
        return circle_mask, square_mask


def mat2vec(x):
    """
    Reshape `x` from matrix to vector by stacking columns.

    Parameters
    ----------
    x : ndarray
        Matrix that should be reshaped to vector.

    Returns
    -------
    ndarray
        Column vector formed by stacking the columns of the matrix `x`.

    See Also
    --------
    vec2mat : The inverse operation

    Notes
    -----
    The returned column vector is C contiguous.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging._util import mat2vec
    >>> x = np.arange(4).reshape(2, 2)
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> mat2vec(x)
    array([[0],
           [2],
           [1],
           [3]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('x', ('boolean', 'integer', 'floating', 'complex'),
                 shape=(-1, -1))

    validate_input()

    return x.T.reshape(-1, 1)


def vec2mat(x, mn_tuple):
    """
    Reshape `x` from column vector to matrix.

    Parameters
    ----------
    x : ndarray
        Matrix that should be reshaped to vector.
    mn_tuple : tuple
        A tuple (m, n) containing the parameters m, n as listed below.
    m : int
        Number of rows in the resulting matrix.
    n : int
        Number of columns in the resulting matrix.

    Returns
    -------
    ndarray
       Matrix formed by taking `n` columns of lenght `m` from the column vector
       `x`.

    See Also
    --------
    mat2vec : The inverse operation

    Notes
    -----
    The returned matrix is C contiguous.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging._util import vec2mat
    >>> x = np.arange(4).reshape(4, 1)
    >>> x
    array([[0],
           [1],
           [2],
           [3]])
    >>> vec2mat(x, (2, 2))
    array([[0, 2],
           [1, 3]])

    """

    m, n = mn_tuple

    @_decorate_validation
    def validate_input():
        _numeric('m', 'integer', range_='[1;inf)')
        _numeric('n', 'integer', range_='[1;inf)')
        _numeric('x', ('boolean', 'integer', 'floating', 'complex'),
                 shape=(m * n, 1))

    validate_input()

    return x.reshape(n, m).T
