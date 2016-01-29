"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
spiral_sample_image(h, w, scan_length, num_points, rect_area=False)
    Function for spiral sampling an image.
spiral_sample_surface(l, w, speed, sample_rate, time, rect_area=False)
    Function for spiral sampling a surface.

"""

from __future__ import division

import numpy as np

from magni.imaging.measurements import _util
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['spiral_sample_image', 'spiral_sample_surface']

_min_l = _util.min_l
_min_w = _util.min_w
_min_speed = _util.min_speed
_min_sample_rate = _util.min_sample_rate
_min_time = _util.min_time
_min_scan_length = _util.min_scan_length
_min_num_points = _util.min_num_points


def spiral_sample_image(h, w, scan_length, num_points, rect_area=False):
    """
    Sample an image using an archimedean spiral pattern.

    The coordinates (in units of pixels) resulting from sampling an image of
    size `h` times `w` using an archimedean spiral pattern are determined. The
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
    rect_area : bool
        A flag indicating whether or not the full rectangular area is sampled
        (the default value is False which implies that the "corners" of the
        rectangular area are not sampled).

    Returns
    -------
    coords : ndarray
        The coordinates of the samples arranged into a 2D array, such that each
        row is a coordinate pair (x, y).

    Notes
    -----
    The orientation of the coordinate system is such that the width `w` is
    measured along the x-axis whereas the height `h` is measured along the
    y-axis. The width must equal the height for an archimedian spiral to make
    sense.

    If the `rect_area` flag is True, then it is assumed that the sampling
    continues outside of the rectangular area specified by `h` and `w` such
    that the "corners" of the rectangular area are also sampled. The sample
    points outside of the rectangular area are discarded and, hence, not
    returned.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import spiral_sample_image
    >>> h = 10
    >>> w = 10
    >>> scan_length = 50.0
    >>> num_points = 12
    >>> np.set_printoptions(suppress=True)
    >>> spiral_sample_image(h, w, scan_length, num_points)
    array([[ 6.28776846,  5.17074073],
           [ 3.13304898,  5.24133767],
           [ 6.07293751,  2.93873701],
           [ 6.99638041,  6.80851189],
           [ 2.89868434,  7.16724999],
           [ 2.35773914,  3.00320067],
           [ 6.41495385,  1.71018152],
           [ 8.82168896,  5.27557847],
           [ 6.34932919,  8.83624957],
           [ 2.04885699,  8.11199373],
           [ 0.6196052 ,  3.96939755]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('h', 'integer', range_='[2;inf)')
        _numeric('w', 'integer', range_='[2;inf)')

        if h != w:
            msg = ('The value of >>h<<, {!r}, must equal the value of >>w<<, '
                   '{!r}, for an archimedian spiral to make sense.')
            raise ValueError(msg.format(h, w))

        _numeric('scan_length', 'floating',
                 range_='[{};inf)'.format(_min_scan_length))
        _numeric('num_points', 'integer',
                 range_='[{};inf)'.format(_min_num_points))
        _numeric('rect_area', 'boolean')

    validate_input()

    coords = spiral_sample_surface(float(h - 1), float(w - 1),
                                   scan_length, float(num_points), 1.0,
                                   rect_area)
    coords = coords + 0.5

    return coords


def spiral_sample_surface(l, w, speed, sample_rate, time, rect_area=False):
    """
    Sample a surface area using an archimedean spiral pattern.

    The coordinates (in units of meters) resulting from sampling an area of
    size `l` times `w` using an archimedean spiral pattern are determined. The
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
    rect_area : bool
        A flag indicating whether or not the full rectangular area is sampled
        (the default value is False which implies that the "corners" of the
        rectangular area are not sampled).

    Returns
    -------
    coords : ndarray
        The coordinates of the samples arranged into a 2D array, such that each
        row is a coordinate pair (x, y).

    Notes
    -----
    The orientation of the coordinate system is such that the width `w` is
    measured along the x-axis whereas the length `l` is measured along the
    y-axis. The width must equal the length for an archimedian sprial to make
    sense.

    If the `rect_area` flag is True, then it is assumed that the sampling
    continues outside of the rectangular area specified by `l` and `w` such
    that the "corners" of the rectangular area are also sampled. The sample
    points outside of the rectangular area are discarded and, hence, not
    returned.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import spiral_sample_surface
    >>> l = 1e-6
    >>> w = 1e-6
    >>> speed = 7e-7
    >>> sample_rate = 1.0
    >>> time = 12.0
    >>> np.set_printoptions(suppress=True)
    >>> spiral_sample_surface(l, w, speed, sample_rate, time)
    array([[ 0.00000036,  0.00000046],
           [ 0.00000052,  0.00000071],
           [ 0.00000052,  0.00000024],
           [ 0.00000059,  0.00000079],
           [ 0.00000021,  0.00000033],
           [ 0.00000084,  0.00000036],
           [ 0.00000049,  0.0000009 ],
           [ 0.0000001 ,  0.00000036],
           [ 0.00000072,  0.00000011],
           [ 0.00000089,  0.00000077],
           [ 0.00000021,  0.00000091]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('l', 'floating', range_='[{};inf)'.format(_min_l))
        _numeric('w', 'floating', range_='[{};inf)'.format(_min_w))

        if l != w:
            msg = ('The value of >>h<<, {!r}, must equal the value of >>w<<, '
                   '{!r}, for an archimedian spiral to make sense.')
            raise ValueError(msg.format(l, w))

        _numeric('speed', 'floating', range_='[{};inf)'.format(_min_speed))
        _numeric('sample_rate', 'floating',
                 range_='[{};inf)'.format(_min_sample_rate))
        _numeric('time', 'floating', range_='[{};inf)'.format(_min_time))
        _numeric('rect_area', 'boolean')

    validate_input()

    sample_period = 1 / sample_rate
    r_end = np.min([l, w]) / 2

    if rect_area:
        # Sample the "corners" of a rectangular area
        r_end *= np.sqrt(2)

    pitch = np.pi * r_end ** 2 / (speed * time)

    # Starting at t=0 yields a divide-by-zero problem. Therefore we start the
    # spiral at t=sample-period
    sample_times = np.linspace(sample_period, time, sample_rate * time - 1)

    r = np.sqrt(pitch * speed / np.pi * sample_times)
    theta = np.sqrt(4 * np.pi * speed * sample_times / pitch)

    coords = np.zeros((len(sample_times), 2))
    coords[:, 0] = r * np.cos(theta)
    coords[:, 1] = r * np.sin(theta)

    # FIXME: This simple stretch yields a non-constant linear speed, which
    # clashes with the idea of the current interface to all sample methods.
    """
    if w > l:
        coords[:, 0] = coords[:, 0] * w / l
    else:
        coords[:, 1] = coords[:, 1] * l / w
    """

    if rect_area:
        # We assume the tip follows the spiral outside of the rectangular area
        # These coordinates should not be included in the sampled area
        within_rect_area = np.where(
            np.all(np.abs(coords) <= (w / 2, l / 2), axis=1))
        coords = coords[within_rect_area]

    coords = coords + np.array([w / 2, l / 2])

    return coords
