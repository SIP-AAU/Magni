"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
zigzag_sample_image(h, w, scan_length, num_points, angle=np.pi / 20)
    Function for zigzag sampling an image.
zigzag_sample_surface(l, w, speed, sample_rate, time, angle=np.pi / 20)
    Function for zigzag sampling a surface.

"""

from __future__ import division

import numpy as np

from magni.imaging.measurements import _util
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['zigzag_sample_image', 'zigzag_sample_surface']

_min_l = _util.min_l
_min_w = _util.min_w
_min_speed = _util.min_speed
_min_sample_rate = _util.min_sample_rate
_min_time = _util.min_time
_min_scan_length = _util.min_scan_length
_min_num_points = _util.min_num_points


def zigzag_sample_image(h, w, scan_length, num_points, angle=np.pi / 20):
    r"""
    Sample an image using a zigzag pattern.

    The coordinates (in units of pixels) resulting from sampling an image of
    size `h` times `w` using a zigzag pattern are determined. The `scan_length`
    determines the length of the path scanned whereas `num_points` indicates
    the number of samples taken on that path.

    Parameters
    ----------
    h : int
        The height of the area to scan in units of pixels.
    w : int
        The width of the area to scan in units of pixels.
    scan_length : float
        The length of the path to scan in units of pixels.
    num_points : int
        The number of samples to take on the scanned path.
    angle : float
        The angle measured in radians by which the lines deviate from being
        horizontal (the default is pi / 20).

    Returns
    -------
    coords : ndarray
        The coordinates of the samples arranged into a 2D array, such that each
        row is a coordinate pair (x, y).

    Notes
    -----
    The orientation of the coordinate system is such that the width `w` is
    measured along the x-axis whereas the height `h` is measured along the
    y-axis.

    The `angle` is measured clockwise relative to horizontal and is limited
    to the interval :math:`\left(0;\arctan\left(\frac{h}{w}\right)\right)`.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import zigzag_sample_image
    >>> h = 10
    >>> w = 10
    >>> scan_length = 50.0
    >>> num_points = 12
    >>> np.set_printoptions(suppress=True)
    >>> zigzag_sample_image(h, w, scan_length, num_points)
    array([[ 0.5       ,  0.5       ],
           [ 4.98949246,  1.21106575],
           [ 9.47898491,  1.9221315 ],
           [ 5.03152263,  2.63319725],
           [ 0.54203017,  3.344263  ],
           [ 4.94746229,  4.05532875],
           [ 9.43695474,  4.7663945 ],
           [ 5.0735528 ,  5.47746025],
           [ 0.58406034,  6.188526  ],
           [ 4.90543212,  6.89959175],
           [ 9.39492457,  7.6106575 ],
           [ 5.11558297,  8.32172325]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('h', 'integer', range_='[2;inf)')
        _numeric('w', 'integer', range_='[2;inf)')
        _numeric('scan_length', 'floating',
                 range_='[{};inf)'.format(_min_scan_length))
        _numeric('num_points', 'integer',
                 range_='[{};inf)'.format(_min_num_points))
        _numeric('angle', 'floating', range_='(0;{})'.format(np.arctan(h / w)))

    validate_input()

    coords = zigzag_sample_surface(float(h - 1), float(w - 1), scan_length,
                                   float(num_points - 1), 1., angle)
    coords = coords + 0.5
    return coords


def zigzag_sample_surface(l, w, speed, sample_rate, time, angle=np.pi / 20):
    r"""
    Sample a surface area using a zigzag pattern.

    The coordinates (in units of meters) resulting from sampling an area of
    size `l` times `w` using a zigzag pattern are determined. The scanned path
    is determined from the probe `speed` and the scan `time`.

    Parameters
    ----------
    l : float
        The length of the area to scan in units of meters.
    w : float
        The width of the area to scan in units of meters.
    speed : float
        The probe speed in units of meters/second.
    sample_rate : float
        The sample rate in units of Hertz.
    time : float
        The scan time in units of seconds.
    angle : float
        The angle measured in radians by which the lines deviate from being
        horizontal (the default is pi / 20).

    Returns
    -------
    coords : ndarray
        The coordinates of the samples arranged into a 2D array, such that each
        row is a coordinate pair (x, y).

    Notes
    -----
    The orientation of the coordinate system is such that the width `w` is
    measured along the x-axis whereas the length `l` is measured along the
    y-axis.

    The `angle` is measured clockwise relative to horizontal and is limited
    to the interval :math:`\left(0;\arctan\left(\frac{h}{w}\right)\right)`.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import zigzag_sample_surface
    >>> l = 1e-6
    >>> w = 1e-6
    >>> speed = 7e-7
    >>> sample_rate = 1.0
    >>> time = 12.0
    >>> np.set_printoptions(suppress=True)
    >>> zigzag_sample_surface(l, w, speed, sample_rate, time)
    array([[ 0.        ,  0.        ],
           [ 0.00000069,  0.00000011],
           [ 0.00000062,  0.00000022],
           [ 0.00000007,  0.00000033],
           [ 0.00000077,  0.00000044],
           [ 0.00000054,  0.00000055],
           [ 0.00000015,  0.00000066],
           [ 0.00000084,  0.00000077],
           [ 0.00000047,  0.00000088],
           [ 0.00000022,  0.00000099],
           [ 0.00000091,  0.0000009 ],
           [ 0.00000039,  0.0000008 ],
           [ 0.0000003 ,  0.00000069]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('l', 'floating', range_='[{};inf)'.format(_min_l))
        _numeric('w', 'floating', range_='[{};inf)'.format(_min_w))
        _numeric('speed', 'floating', range_='[{};inf)'.format(_min_speed))
        _numeric('sample_rate', 'floating',
                 range_='[{};inf)'.format(_min_sample_rate))
        _numeric('time', 'floating', range_='[{};inf)'.format(_min_time))
        _numeric('angle', 'floating', range_='(0;{})'.format(np.arctan(l / w)))

    validate_input()

    length = w / np.cos(angle)
    height = length * np.sin(angle)
    number = speed * time / length

    coords = np.zeros((int(np.ceil(number)) + 1, 2))
    coords[1::2, 0] = w
    coords[:, 1] = np.arange(np.ceil(number) + 1) * height
    coords[-1] = (coords[-2] +
                  np.remainder(number, 1) * (coords[-1] - coords[-2]))
    coords = coords.repeat(2, axis=0)

    for i in range(coords.shape[0]):
        if coords[i, 1] < 0:
            coords[i] = (coords[i - 1, 0] + (coords[i - 1, 1] - 0) /
                         height * (coords[i, 0] - coords[i - 1, 0]), 0)
            coords[i + 1:, 1] = -coords[i + 1:, 1]
        elif coords[i, 1] > l:
            coords[i] = (coords[i - 1, 0] + (l - coords[i - 1, 1]) /
                         height * (coords[i, 0] - coords[i - 1, 0]), l)
            coords[i + 1:, 1] = 2 * l - coords[i + 1:, 1]

    return _util.sample_lines(coords, speed, sample_rate, time)
