"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
lissajous_sample_image(h, w, scan_length, num_points, f_y=1., f_x=1.,
    theta_y=0., theta_x=np.pi / 2)
    Function for lissajous sampling an image.
lissajous_sample_surface(l, w, speed, sample_rate, time, f_y=1., f_x=1.,
    theta_y=0., theta_x=np.pi / 2, speed_mode=0)
    Function for lissajous sampling a surface.

"""

from __future__ import division

import numpy as np

from magni.imaging.measurements import _util
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['lissajous_sample_image', 'lissajous_sample_surface']

_min_l = _util.min_l
_min_w = _util.min_w
_min_speed = _util.min_speed
_min_sample_rate = _util.min_sample_rate
_min_time = _util.min_time
_min_scan_length = _util.min_scan_length
_min_num_points = _util.min_num_points


def lissajous_sample_image(h, w, scan_length, num_points, f_y=1., f_x=1.,
                           theta_y=0., theta_x=np.pi / 2):
    """
    Sample an image using a lissajous pattern.

    The coordinates (in units of pixels) resulting from sampling an image of
    size `h` times `w` using a lissajous pattern are determined. The
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
    f_y : float
        The frequency of the y-sinusoid (the default value is 1.0).
    f_x : float
        The frequency of the x-sinusoid (the default value is 1.0).
    theta_y : float
        The starting phase of the y-sinusoid (the default is 0.0).
    theta_x : float
        The starting phase of the x-sinusoid (the default is pi / 2).

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
    >>> from magni.imaging.measurements import lissajous_sample_image
    >>> h = 10
    >>> w = 10
    >>> scan_length = 50.0
    >>> num_points = 12
    >>> np.set_printoptions(suppress=True)
    >>> lissajous_sample_image(h, w, scan_length, num_points)
    array([[ 5.        ,  9.5       ],
           [ 1.40370042,  7.70492686],
           [ 0.67656563,  3.75183526],
           [ 3.39871123,  0.79454232],
           [ 7.39838148,  1.19240676],
           [ 9.48459832,  4.62800824],
           [ 7.99295651,  8.36038857],
           [ 4.11350322,  9.41181634],
           [ 0.94130617,  6.94345168],
           [ 1.0071768 ,  2.92458128],
           [ 4.25856283,  0.56150128],
           [ 8.10147506,  1.7395012 ],
           [ 9.4699986 ,  5.51876059]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('h', 'integer', range_='[2;inf)')
        _numeric('w', 'integer', range_='[2;inf)')
        _numeric('scan_length', 'floating',
                 range_='[{};inf)'.format(_min_scan_length))
        _numeric('num_points', 'integer',
                 range_='[{};inf)'.format(_min_num_points))
        _numeric('f_y', 'floating', range_='(0;inf)')
        _numeric('f_x', 'floating', range_='(0;inf)')
        _numeric('theta_y', 'floating', range_='(-inf;inf)')
        _numeric('theta_x', 'floating', range_='(-inf;inf)')

    validate_input()

    coords = lissajous_sample_surface(
        float(h - 1), float(w - 1), scan_length, float(num_points), 1.,
        f_y=f_y, f_x=f_x, theta_y=theta_y, theta_x=theta_x)
    coords = coords + 0.5

    return coords


def lissajous_sample_surface(l, w, speed, sample_rate, time, f_y=1., f_x=1.,
                             theta_y=0., theta_x=np.pi / 2, speed_mode=0):
    """
    Sample a surface area using a lissajous pattern.

    The coordinates (in units of meters) resulting from sampling an area of
    size `l` times `w` using a lissajous pattern are determined. The scanned
    path is determined from the probe `speed` and the scan `time`.

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
    f_y : float
        The frequency of the y-sinusoid (the default value is 1.0).
    f_x : float
        The frequency of the x-sinusoid (the default value is 1.0).
    theta_y : float
        The starting phase of the y-sinusoid (the default is 0.0).
    theta_x : float
        The starting phase of the x-sinusoid (the default is pi / 2).
    speed_mode : int
        The speed mode used to select sampling points (the default is 0 which
        implies that the speed argument determines the speed, and f_y and f_x
        determine the ratio between the relative frequencies used).

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

    Generally, the lissajous sampling pattern does not provide constant speed,
    and this cannot be compensated for without violating f_y, f_x, or both.
    Therefore, `speed_mode` allows the user to determine how this issue is
    handled: In `speed_mode` 0, constant speed equal to `speed` is ensured by
    non-uniform sampling of a lissajous curve, whereby `f_y` and `f_x` are not
    constant frequencies. In `speed_mode` 1, average speed equal to `speed` is
    ensured by scaling `f_y` and `f_x` by the same constant. In `speed_mode` 2,
    `f_y` and `f_x` are kept constant and the `speed` is only used to determine
    the path length in combination with `time`.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import lissajous_sample_surface
    >>> l = 1e-6
    >>> w = 1e-6
    >>> speed = 7e-7
    >>> sample_rate = 1.0
    >>> time = 12.0
    >>> np.set_printoptions(suppress=True)
    >>> lissajous_sample_surface(l, w, speed, sample_rate, time)
    array([[ 0.0000005 ,  0.000001  ],
           [ 0.00000001,  0.00000058],
           [ 0.00000033,  0.00000003],
           [ 0.00000094,  0.00000025],
           [ 0.00000082,  0.00000089],
           [ 0.00000017,  0.00000088],
           [ 0.00000007,  0.00000024],
           [ 0.00000068,  0.00000003],
           [ 0.00000099,  0.0000006 ],
           [ 0.00000048,  0.000001  ],
           [ 0.        ,  0.00000057],
           [ 0.00000035,  0.00000002],
           [ 0.00000094,  0.00000027]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('l', 'floating', range_='[{};inf)'.format(_min_l))
        _numeric('w', 'floating', range_='[{};inf)'.format(_min_w))
        _numeric('speed', 'floating', range_='[{};inf)'.format(_min_speed))
        _numeric('sample_rate', 'floating',
                 range_='[{};inf)'.format(_min_sample_rate))
        _numeric('time', 'floating', range_='[{};inf)'.format(_min_time))
        _numeric('f_y', 'floating', range_='(0;inf)')
        _numeric('f_x', 'floating', range_='(0;inf)')
        _numeric('theta_y', 'floating', range_='(-inf;inf)')
        _numeric('theta_x', 'floating', range_='(-inf;inf)')
        _numeric('speed_mode', 'integer', range_='[0;2]')

    validate_input()

    s_x = w / 2
    s_y = l / 2

    if speed_mode in (0, 1):
        # The probe moves 4 * s_x * f_x and 4 * s_y * f_y pixels a second in
        # the x-direction and y-direction, respectively, and the 2-norm of this
        # is a lower bound on the distance per second. Thus, t is an upper
        # bound on the scan time.
        t = speed * time / np.sqrt((4 * s_x * f_x)**2 + (4 * s_y * f_y)**2)
        # The above assumes that f_x * t and f_y * t are integral numbers and
        # so t is increased to ensure the upper bound.
        t = max(np.ceil(f_x * t) / f_x, np.ceil(f_y * t) / f_y)
        # The distance between sampling points on the curve is chosen small
        # enough to approximate the curve by straight line segments.
        dt = 1 / (10**4 * max(f_x, f_y))
        t = np.linspace(0, t, int(t / dt))

        x = s_x * np.cos(2 * np.pi * f_x * t + theta_x) + s_x
        y = s_y * np.cos(2 * np.pi * f_y * t + theta_y) + s_y
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        l = np.zeros(t.shape)
        l[1:] = np.cumsum((dx**2 + dy**2)**(1 / 2))

        if speed_mode == 0:
            # Constant speed entails constant distance between samples.
            l_mode_0 = np.linspace(0, speed * time, sample_rate * time + 1)
            t = np.interp(l_mode_0, l, t)
        else:  # speed_mode == 1
            # The value of t where the desired scan length is reached.
            t_end = np.argmax(l > speed * time) * dt
            t = np.linspace(0, t_end, sample_rate * time + 1)
    else:  # speed_mode == 2
        t = np.linspace(0, time, sample_rate * time + 1)

    x = s_x * np.cos(2 * np.pi * f_x * t + theta_x) + s_x
    y = s_y * np.cos(2 * np.pi * f_y * t + theta_y) + s_y

    return np.column_stack((x, y))
