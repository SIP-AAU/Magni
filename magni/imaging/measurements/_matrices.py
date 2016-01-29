"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
construct_measurement_matrix(coords, h, w)
    Function for constructing a measurement matrix.

"""

from __future__ import division

import numpy as np

from magni.imaging.measurements._util import unique_pixels as _unique_pixels
from magni.utils.matrices import Matrix as _Matrix
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['construct_measurement_matrix']


def construct_measurement_matrix(coords, h, w):
    """
    Construct a measurement matrix extracting the specified measurements.

    Parameters
    ----------
    coords : ndarray
        The `k` floating point coordinates arranged into a 2D array where each
        row is a coordinate pair (x, y), such that `coords` has size `k` x 2.
    h : int
        The height of the image measured in pixels.
    w : int
        The width of the image measured in pixels.

    Returns
    -------
    Phi : magni.utils.matrices.Matrix
        The constructed measurement matrix.

    See Also
    --------
    magni.utils.matrices.Matrix : The matrix emulator class.

    Notes
    -----
    The function constructs two functions: one for extracting pixels at the
    coordinates specified and one for the transposed operation. These functions
    are then wrapped by a matrix emulator which is returned.

    Examples
    --------
    Create a dummy 5 by 5 pixel image and an example sampling pattern:

    >>> import numpy as np, magni
    >>> img = np.arange(25, dtype=np.float).reshape(5, 5)
    >>> vec = magni.imaging.mat2vec(img)
    >>> coords = magni.imaging.measurements.uniform_line_sample_image(
    ...              5, 5, 16., 17)

    Sample the image in the ordinary way:

    >>> unique = magni.imaging.measurements.unique_pixels(coords)
    >>> samples_normal = img[unique[:, 1], unique[:, 0]]
    >>> samples_normal = samples_normal.reshape((len(unique), 1))

    Sample the image using the present function:

    >>> from magni.imaging.measurements import construct_measurement_matrix
    >>> matrix = construct_measurement_matrix(coords, *img.shape)
    >>> samples_matrix = matrix.dot(vec)

    Check that the two ways produce the same result:

    >>> np.allclose(samples_matrix, samples_normal)
    True

    """

    @_decorate_validation
    def validate_input():
        _numeric('coords', ('integer', 'floating'), shape=(-1, 2))
        _numeric('h', 'integer', range_='[1;inf)')
        _numeric('w', 'integer', range_='[1;inf)')
        _numeric('coords[:, 0]', ('integer', 'floating'),
                 range_='[0;{}]'.format(w), shape=(-1,), var=coords[:, 0])
        _numeric('coords[:, 1]', ('integer', 'floating'),
                 range_='[0;{}]'.format(h), shape=(-1,), var=coords[:, 1])

    validate_input()

    coords = _unique_pixels(coords)
    mask = coords[:, 0] * w + coords[:, 1]

    def measure(vec):
        return vec[mask]

    def measure_T(vec):
        output = np.zeros((h * w, 1), dtype=vec.dtype)
        output[mask] = vec
        return output

    return _Matrix(measure, measure_T, [], (len(mask), h * w))
