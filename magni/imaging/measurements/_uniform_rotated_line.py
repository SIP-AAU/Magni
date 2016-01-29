"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
uniform_rotated_line_sample_image(h, w, scan_length, num_points, angle=0.,
    follow_edge=True)
    Function for uniform rotated line sampling an image.
uniform_rotated_line_sample_surface(l, w, speed, sample_rate, time, angle=0.,
    follow_edge=True)
    Function for uniform rotated line sampling a surface.

"""

from __future__ import division

import numpy as np

from magni.imaging.measurements import _util
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['uniform_rotated_line_sample_image',
           'uniform_rotated_line_sample_surface']

_min_l = _util.min_l
_min_w = _util.min_w
_min_speed = _util.min_speed
_min_sample_rate = _util.min_sample_rate
_min_time = _util.min_time
_min_scan_length = _util.min_scan_length
_min_num_points = _util.min_num_points


def uniform_rotated_line_sample_image(h, w, scan_length, num_points, angle=0.,
                                      follow_edge=True):
    r"""
    Sample an image using a uniform rotated line pattern.

    The coordinates (in units of pixels) resulting from sampling an image of
    size `h` times `w` using a uniform rotated line pattern are determined. The
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
    angle : float
        The angle measured in radians by which the uniform lines are rotated
        (the default is 0.0 resulting in a pattern identical to that of
        uniform_line_sample_image).
    follow_edge: bool
        A flag indicating whether or not the pattern follows the edges of the
        rectangular area in-between lines (the default is True).

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

    The `angle` is limited to the interval :math:`[0;\pi)`. An `angle` of 0
    results in the same behaviour as that of uniform_line_sample_image. An
    increase in the `angle` rotates the overall direction counterclockwise,
    i.e., at :math:`\frac{\pi}{2}`, the uniform_line_sample_image sampling
    pattern is rotated 90 degrees counterclockwise.

    If the `follow_edge` flag is True, then the pattern follows the edges of
    the rectangular area when moving from one line to the next. If the flag is
    False, then the pattern follows a line perpendicular to the uniform lines
    when moving from one line to the next. In the latter case, some of the
    uniform lines are shortened to allow the described behaviour.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import \
    ... uniform_rotated_line_sample_image
    >>> h = 10
    >>> w = 10
    >>> scan_length = 50.0
    >>> num_points = 12
    >>> np.set_printoptions(suppress=True)
    >>> uniform_rotated_line_sample_image(h, w, scan_length, num_points)
    array([[ 0.5       ,  0.5       ],
           [ 4.59090909,  0.5       ],
           [ 8.68181818,  0.5       ],
           [ 9.22727273,  3.5       ],
           [ 5.13636364,  3.5       ],
           [ 1.04545455,  3.5       ],
           [ 1.04545455,  6.5       ],
           [ 5.13636364,  6.5       ],
           [ 9.22727273,  6.5       ],
           [ 8.68181818,  9.5       ],
           [ 4.59090909,  9.5       ],
           [ 0.5       ,  9.5       ]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('h', 'integer', range_='[2;inf)')
        _numeric('w', 'integer', range_='[2;inf)')
        _numeric('scan_length', 'floating',
                 range_='[{};inf)'.format(_min_scan_length))
        _numeric('num_points', 'integer',
                 range_='[{};inf)'.format(_min_num_points))
        _numeric('angle', 'floating', range_='[0;{})'.format(np.pi))
        _numeric('follow_edge', 'boolean')

    validate_input()

    coords = uniform_rotated_line_sample_surface(
        float(h - 1), float(w - 1), scan_length, float(num_points - 1), 1.,
        angle, follow_edge)
    coords = coords + 0.5

    return coords


def uniform_rotated_line_sample_surface(l, w, speed, sample_rate, time,
                                        angle=0., follow_edge=True):
    r"""
    Sample a surface area using a uniform rotated line pattern.

    The coordinates (in units of meters) resulting from sampling an area of
    size `l` times `w` using uniform rotated line pattern are determined. The
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
    angle : float
        The angle measured in radians by which the uniform lines are rotated
        (the default is 0.0 resulting in a pattern identical to that of
        uniform_line_sample_image).
    follow_edge: bool
        A flag indicating whether or not the pattern foolows the edges of the
        rectangular area in-between lines (the default is True).

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

    The `angle` is limited to the interval :math:`[0;\pi)`. An `angle` of 0
    results in the same behaviour as that of uniform_line_sample_surface. An
    increase in the `angle` rotates the overall direction counterclockwise,
    i.e., at :math:`\frac{\pi}{2}`, the uniform_line_sample_surface sampling
    pattern is rotated 90 degrees counterclockwise.

    If the `follow_edge` flag is True, then the pattern follows the edges of
    the rectangular area when moving from one line to the next. If the flag is
    False, then the pattern follows a line perpendicular to the uniform lines
    when moving from one line to the next. In the latter case, some of the
    uniform lines are shortened to allow the described behaviour.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import \
    ... uniform_rotated_line_sample_surface
    >>> l = 1e-6
    >>> w = 1e-6
    >>> speed = 7e-7
    >>> sample_rate = 1.0
    >>> time = 12.0
    >>> np.set_printoptions(suppress=True)
    >>> uniform_rotated_line_sample_surface(l, w, speed, sample_rate, time)
    array([[ 0.        ,  0.        ],
           [ 0.00000067,  0.        ],
           [ 0.00000083,  0.00000017],
           [ 0.00000017,  0.00000017],
           [ 0.00000033,  0.00000033],
           [ 0.000001  ,  0.00000033],
           [ 0.0000005 ,  0.0000005 ],
           [ 0.        ,  0.00000067],
           [ 0.00000067,  0.00000067],
           [ 0.00000083,  0.00000083],
           [ 0.00000017,  0.00000083],
           [ 0.00000033,  0.000001  ],
           [ 0.000001  ,  0.000001  ]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('l', 'floating', range_='[{};inf)'.format(_min_l))
        _numeric('w', 'floating', range_='[{};inf)'.format(_min_w))
        _numeric('speed', 'floating', range_='[{};inf)'.format(_min_speed))
        _numeric('sample_rate', 'floating',
                 range_='[{};inf)'.format(_min_sample_rate))
        _numeric('time', 'floating', range_='[{};inf)'.format(_min_time))
        _numeric('angle', 'floating', range_='[0;{})'.format(np.pi))
        _numeric('follow_edge', 'boolean')

    validate_input()

    if angle >= np.pi / 2:
        l, w = w, l
        angle = angle - np.pi / 2
        rotate = True
    else:
        rotate = False

    cos = np.cos(angle)
    sin = np.sin(angle)

    if angle > 0:
        coords_corner = np.float_([[cos * w - sin * l, cos * l + sin * w]])
        dir_l = -sin * coords_corner[0, 1] / cos
        dir_w = cos * coords_corner[0, 1] / sin

    def transform(coords):
        coords[:, 1] = -coords[:, 1]
        coords = coords.T
        coords = np.float_([[cos, -sin], [sin, cos]]).dot(coords)
        coords = coords.T
        coords[:, 1] = -coords[:, 1]
        return coords

    n = 3
    coords_prev = np.zeros((3, 2))

    while True:
        ratios = np.linspace(0, 1, n).reshape((n, 1))

        if angle > 0:
            Y = ratios * coords_corner[0, 1]
            X_lower = np.maximum(dir_l * ratios,
                                 coords_corner[0, 0] - dir_w * (1 - ratios))
            X_upper = np.minimum(dir_w * ratios,
                                 coords_corner[0, 0] - dir_l * (1 - ratios))
        else:
            Y = ratios * l
            X_lower = 0 + ratios * 0
            X_upper = w + ratios * 0

        coords_lower = np.column_stack((X_lower, Y))
        coords_upper = np.column_stack((X_upper, Y))

        if follow_edge:
            coords_lower = transform(coords_lower)
            coords_upper = transform(coords_upper)
            coords = np.zeros((3 * n, 2))

            # alternate between points
            coords[1::6] = coords_lower[0::2]
            coords[2::6] = coords_upper[0::2]
            coords[4::6] = coords_upper[1::2]
            coords[5::6] = coords_lower[1::2]

            # fill the blanks
            coords[3:-1:6, 0] = coords[4:-1:6, 0]
            coords[3:-1:6, 1] = coords[2:-1:6, 1]
            coords[6:-1:6, 0] = coords[5:-1:6, 0]
            coords[6:-1:6, 1] = coords[7:-1:6, 1]
        else:
            coords = np.zeros((2 * n, 2))

            # alternate between points
            coords[0::4] = coords_lower[0::2]
            coords[1::4] = coords_upper[0::2]
            coords[2::4] = coords_upper[1::2]
            coords[3::4] = coords_lower[1::2]

            # shorten
            coords[1:-1:4, 0] = coords[2:-1:4, 0] = np.minimum(
                coords[1:-1:4, 0], coords[2:-1:4, 0])
            coords[3:-1:4, 0] = coords[4:-1:4, 0] = np.maximum(
                coords[3:-1:4, 0], coords[4:-1:4, 0])

            coords = transform(coords)

        length = coords[1:] - coords[:-1]
        length = np.sum(np.sqrt(length[:, 0]**2 + length[:, 1]**2))

        if length > speed * time:
            n = n - 1
            coords = coords_prev
            break
        else:
            n = n + 1
            coords_prev = coords

    if rotate:
        l, w = w, l
        coords[:, 0], coords[:, 1] = coords[:, 1], l - coords[:, 0]

    X = coords[:, 0]
    X[X < 0] = 0
    X[X > w] = w
    Y = coords[:, 1]
    Y[Y < 0] = 0
    Y[Y > l] = l

    return _util.sample_lines(coords, speed, sample_rate, time)
