"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing fast linear operations wrapped in matrix emulators.

Routine listings
----------------
get_DCT(shape, overcomplete_shape=None)
    Get the DCT fast operation dictionary for the given image shape.
get_DFT(shape, overcomplete_shape=None)
    Get the DFT fast operation dictionary for the given image shape.

See Also
--------
magni.imaging.dictionaries._fastops : Fast linear operations.
magni.utils.matrices : Matrix emulators.

"""


from __future__ import division

from pkg_resources import parse_version as _parse_version
from scipy import __version__ as _scipy_version

from magni.imaging.dictionaries import _fastops
from magni.utils.matrices import Matrix as _Matrix
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


def get_DCT(shape, overcomplete_shape=None):
    """
    Get the DCT fast operation dictionary for the given image shape.

    Parameters
    ----------
    shape : list or tuple
        The shape of the image for which the dictionary is the DCT dictionary.
    overcomplete_shape : list or tuple, optional
        The shape of the (overcomplete) frequency domain for the DCT
        dictionary. The entries must be greater than or equal to the
        corresponding entries in `shape`.

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

    Compute the overcomplete transform (and back again) and check that the
    resulting image is identical to the original. Notice how this example first
    ensures that the necessary version of SciPy is available:

    >>> from pkg_resources import parse_version
    >>> from scipy import __version__ as _scipy_version
    >>> if parse_version(_scipy_version) >= parse_version('0.16.0'):
    ...     matrix = get_DCT(img.shape, img.shape)
    ...     dct_matrix = matrix.T.dot(vec)
    ...     vec_roundtrip = matrix.dot(dct_matrix)
    ...     np.allclose(vec, vec_roundtrip)
    ... else:
    ...     True
    True

    """

    @_decorate_validation
    def validate_input():
        _levels('shape', (
            _generic(None, 'explicit collection', len_=2),
            _numeric(None, 'integer', range_='[1;inf)')))

        if overcomplete_shape is not None:
            _generic('overcomplete_shape', 'explicit collection', len_=2),
            _numeric(('overcomplete_shape', 0),
                     'integer', range_='[{};inf)'.format(shape[0]))
            _numeric(('overcomplete_shape', 1),
                     'integer', range_='[{};inf)'.format(shape[1]))

    validate_input()

    entries = shape[0] * shape[1]

    if overcomplete_shape is None:
        args = (shape,)
        shape = (entries, entries)
    else:
        if _parse_version(_scipy_version) < _parse_version('0.16.0'):
            raise NotImplementedError(
                'Over-complete DCT requires SciPy >= 0.16.0')

        args = (shape, overcomplete_shape)
        shape = (entries, overcomplete_shape[0] * overcomplete_shape[1])

    return _Matrix(_fastops.idct2, _fastops.dct2, args, shape)


def get_DFT(shape, overcomplete_shape=None):
    """
    Get the DFT fast operation dictionary for the given image shape.

    Parameters
    ----------
    shape : list or tuple
        The shape of the image for which the dictionary is the DFT dictionary.
    overcomplete_shape : list or tuple, optional
        The shape of the (overcomplete) frequency domain for the DFT
        dictionary. The entries must be greater than or equal to the
        corresponding entries in `shape`.

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
    >>> dft_matrix = matrix.conj().T.dot(vec)

    Check that the two ways produce the same result:

    >>> np.allclose(dft_matrix, dft_normal)
    True

    Compute the overcomplete transform (and back again):

    >>> matrix = get_DFT(img.shape, img.shape)
    >>> dft_matrix = matrix.conj().T.dot(vec)
    >>> vec_roundtrip = matrix.dot(dft_matrix)

    Check that the twice transformed image is identical to the
    original:

    >>> np.allclose(vec, vec_roundtrip)
    True

    """

    @_decorate_validation
    def validate_input():
        _levels('shape', (
            _generic(None, 'explicit collection', len_=2),
            _numeric(None, 'integer', range_='[1;inf)')))

        if overcomplete_shape is not None:
            _generic('overcomplete_shape', 'explicit collection', len_=2),
            _numeric(('overcomplete_shape', 0),
                     'integer', range_='[{};inf)'.format(shape[0]))
            _numeric(('overcomplete_shape', 1),
                     'integer', range_='[{};inf)'.format(shape[1]))

    validate_input()

    entries = shape[0] * shape[1]

    if overcomplete_shape is None:
        args = (shape,)
        shape = (entries, entries)
    else:
        args = (shape, overcomplete_shape)
        shape = (entries, overcomplete_shape[0] * overcomplete_shape[1])

    return _Matrix(_fastops.idft2, _fastops.dft2, args, shape, is_complex=True)
