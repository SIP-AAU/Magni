"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
uniform_line_sample_image(h, w, scan_length, num_points)
    Function for uniform line sampling an image.
uniform_line_sample_surface(l, w, speed, sample_rate, time)
    Function for uniform line sampling a surface.

"""

from __future__ import division

import numpy as np

from magni.imaging.measurements import _util
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['uniform_line_sample_image', 'uniform_line_sample_surface']

_min_l = _util.min_l
_min_w = _util.min_w
_min_speed = _util.min_speed
_min_sample_rate = _util.min_sample_rate
_min_time = _util.min_time
_min_scan_length = _util.min_scan_length
_min_num_points = _util.min_num_points


def uniform_line_sample_image(h, w, scan_length, num_points):
    """
    Sample an image using a set of uniformly distributed straight lines.

    The coordinates (in units of pixels) resulting from sampling an image of
    size `h` times `w` using a pattern based on a set of uniformly distributed
    straight lines are determined. The `scan_length` determines the length of
    the path scanned whereas `num_points` indicates the number of samples taken
    on that path.

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

    Each of the scanned lines span the entire width of the image with the
    exception of the last line that may only be partially scanned if the
    `scan_length` implies this. The top and bottom lines of the image are
    always included in the scan.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import uniform_line_sample_image
    >>> h = 10
    >>> w = 10
    >>> scan_length = 50.0
    >>> num_points = 12
    >>> np.set_printoptions(suppress=True)
    >>> uniform_line_sample_image(h, w, scan_length, num_points)
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

    validate_input()

    coords = uniform_line_sample_surface(float(h - 1), float(w - 1),
                                         scan_length, float(num_points - 1),
                                         1.0)

    coords = coords + 0.5

    return coords


def uniform_line_sample_surface(l, w, speed, sample_rate, time):
    """
    Sample aa surface area using a set of uniformly distributed straight lines.

    The coordinates (in units of meters) resulting from sampling an area of
    size `l` times `w` using a pattern based on a set of uniformly distributed
    straight lines are determined.  The scanned path is determined from the
    probe `speed` and the scan `time`.

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
    measured along the x-axis whereas the height `l` is measured along the
    y-axis.

    Each of the scanned lines span the entire width of the image with the
    exception of the last line that may only be partially scanned if the
    `scan_length` implies this. The top and bottom lines of the image are
    always included in the scan.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import uniform_line_sample_surface
    >>> l = 2e-6
    >>> w = 2e-6
    >>> speed = 7e-7
    >>> sample_rate = 1.0
    >>> time = 12.0
    >>> np.set_printoptions(suppress=True)
    >>> uniform_line_sample_surface(l, w, speed, sample_rate, time)
    array([[ 0.        ,  0.        ],
           [ 0.00000067,  0.        ],
           [ 0.00000133,  0.        ],
           [ 0.000002  ,  0.        ],
           [ 0.000002  ,  0.00000067],
           [ 0.00000167,  0.000001  ],
           [ 0.000001  ,  0.000001  ],
           [ 0.00000033,  0.000001  ],
           [ 0.        ,  0.00000133],
           [ 0.        ,  0.000002  ],
           [ 0.00000067,  0.000002  ],
           [ 0.00000133,  0.000002  ],
           [ 0.000002  ,  0.000002  ]])

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

    num_lines = int(np.floor((speed * time - l) / w))

    # We should always at least partially scan top and bottom lines.
    if num_lines < 2:
        num_lines = 2

    coords = np.zeros((2 * num_lines, 2))
    coords[1::4, 0] = coords[2::4, 0] = w
    coords[0::2, 1] = coords[1::2, 1] = np.linspace(0, l, num_lines)

    return _util.sample_lines(coords, speed, sample_rate, time)
