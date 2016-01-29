"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
square_spiral_sample_image(h, w, scan_length, num_points)
    Function for square spiral sampling an image.
square_spiral_sample_surface(l, w, speed, sample_rate, time)
    Function for square spiral sampling a surface.

"""

from __future__ import division

import numpy as np

from magni.imaging.measurements import _util
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['square_spiral_sample_image', 'square_spiral_sample_surface']

_min_l = _util.min_l
_min_w = _util.min_w
_min_speed = _util.min_speed
_min_sample_rate = _util.min_sample_rate
_min_time = _util.min_time
_min_scan_length = _util.min_scan_length
_min_num_points = _util.min_num_points


def square_spiral_sample_image(h, w, scan_length, num_points):
    """
    Sample an image using a square spiral pattern.

    The coordinates (in units of pixels) resulting from sampling an image of
    size `h` times `w` using a square spiral pattern are determined. The
    `scan_length` determines the length of the path scanned whereas
    `num_points` indicates the number of samples taken on that path.

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

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import square_spiral_sample_image
    >>> h = 10
    >>> w = 10
    >>> scan_length = 50.0
    >>> num_points = 12
    >>> np.set_printoptions(suppress=True)
    >>> square_spiral_sample_image(h, w, scan_length, num_points)
    array([[ 5.        ,  5.        ],
           [ 6.28571429,  5.97619048],
           [ 4.38095238,  3.71428571],
           [ 2.42857143,  5.92857143],
           [ 4.95238095,  7.57142857],
           [ 7.57142857,  6.02380952],
           [ 7.        ,  2.42857143],
           [ 2.83333333,  2.42857143],
           [ 1.14285714,  4.9047619 ],
           [ 1.35714286,  8.85714286],
           [ 5.52380952,  8.85714286],
           [ 8.85714286,  8.02380952]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('h', 'integer', range_='[2;inf)')
        _numeric('w', 'integer', range_='[2;inf)')
        _numeric('scan_length', 'floating',
                 range_='[{};inf)'.format(_min_scan_length))
        _numeric('num_points', 'integer',
                 range_='[{};inf)'.format(_min_num_points))

    validate_input()

    coords = square_spiral_sample_surface(float(h - 1), float(w - 1),
                                          scan_length, float(num_points), 1.0)
    coords = coords + 0.5

    return coords


def square_spiral_sample_surface(l, w, speed, sample_rate, time):
    """
    Sample a surface area using a square spiral pattern.

    The coordinates (in units of meters) resulting from sampling an area of
    size `l` times `w` using a square spiral pattern are determined. The
    scanned path is determined from the probe `speed` and the scan `time`.

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

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import square_spiral_sample_surface
    >>> l = 1e-6
    >>> w = 1e-6
    >>> speed = 7e-7
    >>> sample_rate = 1.0
    >>> time = 12.0
    >>> np.set_printoptions(suppress=True)
    >>> square_spiral_sample_surface(l, w, speed, sample_rate, time)
    array([[ 0.0000005,  0.0000005],
           [ 0.0000004,  0.0000004],
           [ 0.0000006,  0.0000007],
           [ 0.0000005,  0.0000003],
           [ 0.0000002,  0.0000007],
           [ 0.0000008,  0.0000007],
           [ 0.0000006,  0.0000002],
           [ 0.0000001,  0.0000004],
           [ 0.0000003,  0.0000009],
           [ 0.0000009,  0.0000008],
           [ 0.0000009,  0.0000001],
           [ 0.0000002,  0.0000001]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('l', 'floating', range_='[{};inf)'.format(_min_l))
        _numeric('w', 'floating', range_='[{};inf)'.format(_min_w))
        _numeric('speed', 'floating', range_='[{};inf)'.format(_min_speed))
        _numeric('sample_rate', 'floating',
                 range_='[{};inf)'.format(_min_sample_rate))
        _numeric('time', 'floating', range_='[{};inf)'.format(_min_time))

    validate_input()

    samples = int(sample_rate * time)
    scan_length = speed * time
    sample_dist = scan_length / samples
    min_size = np.min([l, w])
    delta_dist = np.abs(l - w) / 2

    # Determine number of rotations and length of line segments in spiral
    roots = np.polynomial.polynomial.polyroots(
        [-(scan_length + 2 * delta_dist),
         min_size + 2 * delta_dist - scan_length,
         min_size + 2 * delta_dist])  # [c, b, a] in ax^2 + bx + c = 0
    N = np.ceil(roots.max())  # scan_length ~= (1+1+2+2+..+N+N) * line_length
    line_len = min_size / (N + 1)

    coords = np.zeros((samples, 2))
    position = np.zeros(2)  # Start spiral at (0, 0); later shift its position
    directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # -x, +y, +x, -y
    direction_num = 0  # Start spiral by a horisontal line to the left

    # Handle possibly non-square sampling area
    delta = delta_dist * np.array([w > l, w < l])

    n = 1  # n \in (range(N) + 1)
    k = 0  # k = 0 <-> horisontal line segment; k = 1 <-> vertical line segment
    for sample in range(samples):
        # Mark coordinates
        coords[sample, :] = position

        # Update position
        position = position + directions[direction_num, :] * sample_dist

        # Handle corners
        if np.abs(position[k]) > np.ceil(n / 2) * line_len + delta[k]:
            # Fix overshoot from last iteration
            wrap_dist = np.abs(position[k]) - (np.ceil(n / 2) * line_len +
                                               delta[k])
            position = position - directions[direction_num, :] * wrap_dist

            # Find offset if several corners are involved
            offset = np.zeros(2)
            while wrap_dist > n * line_len + 2 * delta[k - 1]:
                wrap_dist = wrap_dist - n * line_len + 2 * delta[k - 1]
                direction_num = np.mod(direction_num + 1, 4)
                k = np.mod(direction_num, 2)
                if k == 0:
                    n = n + 1
                offset = offset + ((n * line_len + 2 * delta[k]) *
                                   directions[direction_num, :])
            else:
                direction_num = np.mod(direction_num + 1, 4)
                k = np.mod(direction_num, 2)
                if k == 0:
                    n = n + 1

            position = position + offset
            position = position + directions[direction_num, :] * wrap_dist

    # Shift to center of region
    coords = coords + np.array([w / 2, l / 2])

    return coords
