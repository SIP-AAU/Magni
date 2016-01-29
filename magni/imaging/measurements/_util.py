"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
construct_pixel_mask(h, w, pixels)
    Construct a binary pixel mask.
unique_pixels(coords)
    Function for determining unique pixels from a set of coordinates.

"""

from __future__ import division

import numpy as np

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['construct_pixel_mask', 'unique_pixels']

# In principle most of the AFM-scanning related parameters should just be
# positive, however we have settled for:
min_l = 1e-9  # [m]
min_w = 1e-9  # [m]
min_speed = 1e-9  # [m/s]
min_sample_rate = 1.0  # [Hz]
min_time = 1.0  # [s]
min_scan_length = 1e-9  # [m]
min_num_points = 1  # []


def construct_pixel_mask(h, w, pixels):
    """
    Construct a binary pixel mask.

    An image (2D array) of shape `w` x `h` is created where all `pixels` are
    marked True.

    Parameters
    ----------
    h : int
        The height of the image in pixels.
    w : int
        The width of the image in pixels.
    pixels : ndarray
        The 2D array of pixels that make up the mask. Each row is a coordinate
        pair (x, y), such that `coords` has size len(`pixels`) x 2.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import construct_pixel_mask
    >>> h = 3
    >>> w = 3
    >>> pixels = np.array([[0, 0], [1, 1], [2, 1]])
    >>> construct_pixel_mask(h, w, pixels)
    array([[ True, False, False],
           [False,  True,  True],
           [False, False, False]], dtype=bool)

    """

    @_decorate_validation
    def validate_input():
        _numeric('h', 'integer', range_='[2;inf)')
        _numeric('w', 'integer', range_='[2;inf)')
        _numeric('pixels', 'integer', shape=(-1, 2))
        _numeric('pixels[:, 0]', 'integer', shape=(-1,),
                 range_='[0;{})'.format(w), var=pixels[:, 0])
        _numeric('pixels[:, 1]', 'integer', shape=(-1,),
                 range_='[0;{})'.format(h), var=pixels[:, 1])

    validate_input()

    mask = np.zeros((h, w), dtype=np.bool_)
    mask[pixels[:, 1], pixels[:, 0]] = True

    return mask


def sample_lines(coords, speed, sample_rate, time):
    """
    Determine the coordinates of the sampled points in a line scanning.

    Parameters
    ----------
    coords : ndarray
        The `k` floating point coordinates of the line segment endpoints
        arranged into a 2D array where each row is a coordinate pair (x, y),
        such that `coords` has size `k` x 2.
    speed : float
        The probe speed in units of meters/second.
    sample_rate : float
        The sample rate in units of Hertz.
    time : float
        The scan time in units of seconds.

    Returns
    -------
    coords : ndarray
        The coordinates of the samples arranged into a 2D array, such that each
        row is a coordinate pair (x, y).

    """

    dcoords = coords[1:] - coords[:-1]
    cumlen = np.zeros(coords.shape[0])
    cumlen[1:] = np.cumsum(np.sqrt(dcoords[:, 0]**2 + dcoords[:, 1]**2))
    speed = cumlen[-1] / time

    l = np.linspace(0, speed * time, int(sample_rate * time) + 1)
    x = np.interp(l, cumlen, coords[:, 0])
    y = np.interp(l, cumlen, coords[:, 1])

    return np.column_stack((x, y))


def unique_pixels(coords):
    """
    Identify unique pixels from a set of coordinates.

    The floating point `coords` are reduced to a unique set of integer pixels
    by flooring the floating point values.

    Parameters
    ----------
    coords : ndarray
        The `k` floating point coordinates arranged into a 2D array where each
        row is a coordinate pair (x, y), such that `coords` has size `k` x 2.

    Returns
    -------
    unique_pixels : ndarray
        The `l` <= `k` unique (integer) pixels, such that `unique_pixels` is a
        2D array and has size `l` x 2.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import unique_pixels
    >>> coords = np.array([[1.7, 1.0], [1.0, 1.2], [3.3, 4.3]])
    >>> np.int_(unique_pixels(coords))
    array([[1, 1],
           [3, 4]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('coords', ('integer', 'floating'), range_='[0;inf)',
                 shape=(-1, 2))

    validate_input()

    pixels = np.floor(coords).astype(np.int64, order='C')

    pixels_struct = pixels.view(pixels.dtype.descr * 2)
    unique, index = np.unique(pixels_struct, return_index=True)
    unique_pixels = pixels[np.sort(index)]

    return unique_pixels
