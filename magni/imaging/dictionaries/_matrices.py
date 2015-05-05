"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing fast linear operations wrapped in matrix emulators.

Routine listings
----------------
get_DCT(shape)
    Get the DCT fast operation dictionary for the given image shape.
get_DFT(shape)
    Get the DFT fast operation dictionary for the given image shape.

See Also
--------
magni.imaging.dictionaries._fastops : Fast linear operations.
magni.utils.matrices : Matrix emulators.

"""


from __future__ import division

from magni.imaging.dictionaries import _fastops
from magni.utils.matrices import Matrix as _Matrix
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


def get_DCT(shape):
    """
    Get the DCT fast operation dictionary for the given image shape.

    Parameters
    ----------
    shape : list or tuple
        The shape of the image which the dictionary is the DCT dictionary.

    Returns
    -------
    matrix : magni.utils.matrices.Matrix
        The specified DCT dictionary.

    See Also
    --------
    magni.utils.matrices.Matrix : The matrix emulator class.

    Examples
    --------
    Create a dummy image:

    >>> import numpy as np, magni
    >>> img = np.random.randn(64, 64)
    >>> vec = magni.imaging.mat2vec(img)

    Perform DCT in the ordinary way:

    >>> dct_normal = magni.imaging.dictionaries._fastops.dct2(vec, img.shape)

    Perform DCT using the present function:

    >>> from magni.imaging.dictionaries import get_DCT
    >>> matrix = get_DCT(img.shape)
    >>> dct_matrix = matrix.T.dot(vec)

    Check that the two ways produce the same result:

    >>> np.allclose(dct_matrix, dct_normal)
    True

    """

    @_decorate_validation
    def validate_input():
        _levels('shape', (_generic(None, 'explicit collection', len_=2),
                          _numeric(None, 'integer', range_='[1;inf)')))

    validate_input()

    entries = shape[0] * shape[1]
    return _Matrix(_fastops.idct2, _fastops.dct2, (shape,), (entries, entries))


def get_DFT(shape):
    """
    Get the DFT fast operation dictionary for the given image shape.

    Parameters
    ----------
    shape : list or tuple
        The shape of the image which the dictionary is the DFT dictionary.

    Returns
    -------
    matrix : magni.utils.matrices.Matrix
        The specified DFT dictionary.

    See Also
    --------
    magni.utils.matrices.Matrix : The matrix emulator class.

    Examples
    --------
    Create a dummy image:

    >>> import numpy as np, magni
    >>> img = np.random.randn(64, 64)
    >>> vec = magni.imaging.mat2vec(img)

    Perform DFT in the ordinary way:

    >>> dft_normal = magni.imaging.dictionaries._fastops.dft2(vec, img.shape)

    Perform DFT using the present function:

    >>> from magni.imaging.dictionaries import get_DFT
    >>> matrix = get_DFT(img.shape)
    >>> dft_matrix = matrix.T.dot(vec)

    Check that the two ways produce the same result:

    >>> np.allclose(dft_matrix, dft_normal)
    True

    """

    @_decorate_validation
    def validate_input():
        _levels('shape', (_generic(None, 'explicit collection', len_=2),
                          _numeric(None, 'integer', range_='[1;inf)')))

    validate_input()

    entries = shape[0] * shape[1]
    return _Matrix(_fastops.idft2, _fastops.dft2, (shape,), (entries, entries))
