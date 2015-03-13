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
