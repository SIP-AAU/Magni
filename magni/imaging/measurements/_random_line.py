"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
random_line_sample_image(h, w, scan_length, num_points, discrete=None,
    seed=None)
    Function for random line sampling an image.
random_line_sample_surface(l, w, speed, sample_rate, time, discrete=None,
    seed=None)
    Function for random line sampling a surface.

"""

from __future__ import division

import numpy as np

from magni.imaging.measurements import _util
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['random_line_sample_image', 'random_line_sample_surface']

_min_l = _util.min_l
_min_w = _util.min_w
_min_speed = _util.min_speed
_min_sample_rate = _util.min_sample_rate
_min_time = _util.min_time
_min_scan_length = _util.min_scan_length
_min_num_points = _util.min_num_points


def random_line_sample_image(h, w, scan_length, num_points, discrete=None,
                             seed=None):
    """
    Sample an image using a set of random straight lines.

    The coordinates (in units of pixels) resulting from sampling an image of
    size `h` times `w` using a pattern based on a set of random straight lines
    are determined. The `scan_length` determines the length of the path scanned
    whereas `num_points` indicates the number of samples taken on that path. If
    `discrete` is set, it specifies the finite number of equally spaced lines
    from which the scan lines are be chosen at random. For reproducible
    results, the `seed` may be used to specify a fixed seed of the random
    number generator.

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
    discrete : int or None, optional
        The number of equally spaced lines from which the scan lines are chosen
        (the default is None, which implies that no discritisation is used).
    seed : int or None, optional
        The seed used for the random number generator (the defaul is None,
        which implies that the random number generator is not seeded).

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
    >>> from magni.imaging.measurements import random_line_sample_image
    >>> h = 10
    >>> w = 10
    >>> scan_length = 50.0
    >>> num_points = 12
    >>> seed = 6021
    >>> np.set_printoptions(suppress=True)
    >>> random_line_sample_image(h, w, scan_length, num_points, seed=seed)
    array([[ 0.5       ,  0.5       ],
           [ 4.59090909,  0.5       ],
           [ 8.68181818,  0.5       ],
           [ 7.01473938,  1.28746666],
           [ 2.92383029,  1.28746666],
           [ 0.5       ,  2.95454545],
           [ 0.5       ,  7.04545455],
           [ 4.03665944,  7.59970419],
           [ 8.12756853,  7.59970419],
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
        _numeric('discrete', 'integer', range_='[2;inf)', ignore_none=True)
        _numeric('seed', 'integer', range_='[0;inf)', ignore_none=True)

    validate_input()

    coords = random_line_sample_surface(float(h - 1), float(w - 1),
                                        scan_length, float(num_points - 1),
                                        1.0, discrete=discrete, seed=seed)
    coords = coords + 0.5

    return coords


def random_line_sample_surface(l, w, speed, sample_rate, time, discrete=None,
                               seed=None):
    """
    Sample a surface area using a set of random straight lines.

    The coordinates (in units of meters) resulting from sampling an image of
    size `l` times `w` using a pattern based on a set of random straight lines
    are determined.  The scanned path is determined from the probe `speed` and
    the scan `time`. If `discrete` is set, it specifies the finite number of
    equally spaced lines from which the scan lines are be chosen at random.
    For reproducible results, the `seed` may be used to specify a fixed seed of
    the random number generator.

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
    discrete : int or None, optional
        The number of equally spaced lines from which the scan lines are chosen
        (the default is None, which implies that no discritisation is used).
    seed : int or None, optional
        The seed used for the random number generator (the defaul is None,
        which implies that the random number generator is not seeded).

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

    Each of the scanned lines span the entire width of the image with the
    exception of the last line that may only be partially scanned if the
    `speed` and `time` implies this. The top and bottom lines of the image are
    always included in the scan and are not included in the `discrete` number
    of lines.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import random_line_sample_surface
    >>> l = 2e-6
    >>> w = 2e-6
    >>> speed = 7e-7
    >>> sample_rate = 1.0
    >>> time = 12.0
    >>> seed = 6021
    >>> np.set_printoptions(suppress=True)
    >>> random_line_sample_surface(l, w, speed, sample_rate, time, seed=seed)
    array([[ 0.        ,  0.        ],
           [ 0.00000067,  0.        ],
           [ 0.00000133,  0.        ],
           [ 0.000002  ,  0.        ],
           [ 0.000002  ,  0.00000067],
           [ 0.000002  ,  0.00000133],
           [ 0.00000158,  0.00000158],
           [ 0.00000091,  0.00000158],
           [ 0.00000024,  0.00000158],
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
        _numeric('discrete', 'integer', range_='[2;inf)', ignore_none=True)
        _numeric('seed', 'integer', range_='[0;inf)', ignore_none=True)

        if (speed * time - 2 * w - l) / w <= -1:
            # Estimated number of lines in addition to top and bottom lines
            # must exceed -1 to avoid drawing a negative number of lines at
            # random.
            msg = ('The value of >>(speed * time - 2 * w - l) / w<<, {!r}, '
                   'must be > -1.')
            raise ValueError(msg.format((speed * time - 2 * w - l) / w))

    validate_input()

    if seed is not None:
        np.random.seed(seed)

    num_lines = int(np.floor((speed * time - l) / w))

    if discrete is None:
        lines = np.sort(np.random.rand(num_lines - 2) * l)
    else:
        possible_lines = l / (discrete + 1) * np.arange(1, discrete + 1)

        try:
            lines = np.sort(np.random.choice(
                possible_lines, size=num_lines - 2, replace=False))
        except ValueError:
            raise ValueError('The number of Discrete lines must be large ' +
                             'enough to contain the entire scan path. With ' +
                             'the current settings, a minimun of '
                             '{!r} lines are required.'.format(num_lines - 2))

    coords = np.zeros((2 * num_lines, 2))
    coords[1::4, 0] = coords[2::4, 0] = w
    coords[2:-2:2, 1] = coords[3:-2:2, 1] = lines
    coords[-2:, 1] = l

    return _util.sample_lines(coords, speed, sample_rate, time)
