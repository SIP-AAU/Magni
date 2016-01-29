"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing functionality for constructing and visualising scan
patterns for measurements.

This subpackage provides several pairs of scan pattern functions. The first
function, named \*_sample_surface, is used for sampling a given surface. The
second function, named \*_sample_image, is a wrapper that provides a
pixel-oriented interface to the first function. In addition to these pairs of
scan pattern functions, the module provides auxillary functions that may be
used to visualise the scan patterns.

Routine listings
----------------
construct_measurement_matrix(coords, h, w)
    Function for constructing a measurement matrix.
construct_pixel_mask(h, w, pixels)
    Construct a binary pixel mask.
lissajous_sample_image(h, w, scan_length, num_points, f_y=1., f_x=1.,
    theta_y=0., theta_x=np.pi / 2)
    Function for lissajous sampling an image.
lissajous_sample_surface(l, w, speed, sample_rate, time, f_y=1., f_x=1.,
    theta_y=0., theta_x=np.pi / 2, speed_mode=0)
    Function for lissajous sampling a surface.
plot_pattern(l, w, coords, mode, output_path=None)
    Function for visualising a scan pattern.
plot_pixel_mask(h, w, pixels, output_path=None)
    Function for visualising a pixel mask obtained from a scan pattern.
random_line_sample_image(h, w, scan_length, num_points, discrete=None,
    seed=None)
    Function for random line sampling an image.
random_line_sample_surface(l, w, speed, sample_rate, time, discrete=None,
    seed=None)
    Function for random line sampling a surface.
spiral_sample_image(h, w, scan_length, num_points, rect_area=False)
    Function for spiral sampling an image.
spiral_sample_surface(l, w, speed, sample_rate, time, rect_area=False)
    Function for spiral sampling a surface.
square_spiral_sample_image(h, w, scan_length, num_points)
    Function for square spiral sampling an image.
square_spiral_sample_surface(l, w, speed, sample_rate, time)
    Function for square spiral sampling a surface.
uniform_line_sample_image(h, w, scan_length, num_points)
    Function for uniform line sampling an image.
uniform_line_sample_surface(l, w, speed, sample_rate, time)
    Function for uniform line sampling a surface.
uniform_rotated_line_sample_image(h, w, scan_length, num_points, angle=0.,
    follow_edge=True)
    Function for uniform rotated line sampling an image.
uniform_rotated_line_sample_surface(l, w, speed, sample_rate, time, angle=0.,
    follow_edge=True)
    Function for uniform rotated line sampling a surface.
unique_pixels(coords)
    Function for determining unique pixels from a set of coordinates.
zigzag_sample_image(h, w, scan_length, num_points, angle=np.pi / 20)
    Function for zigzag sampling an image.
zigzag_sample_surface(l, w, speed, sample_rate, time, angle=np.pi / 20)
    Function for zigzag sampling a surface.

Notes
-----
In principle, most of the scan pattern related parameters need only be
positive. However, it is assumed that the following requirements are fulfilled:

:Minimum length of scan area: 1 nm
:Minimum width of scan area: 1 nm
:Minimum scan speed: 1 nm/s
:Minimum sample_rate: 1 Hz
:Minimum scan time: 1 s
:Minimum scan length: 1 nm
:Minimum number of scan points: 1

Examples
--------
Sample a surface using a lissajous pattern:

>>> from magni.imaging.measurements import lissajous_sample_surface
>>> l = 13.0; w = 13.0; speed = 4.0; time = 27.0; sample_rate = 3.0;
>>> coords = lissajous_sample_surface(l, w, speed, sample_rate, time,
...                                   speed_mode=1)

Display the resulting pattern:

>>> from magni.imaging.measurements import plot_pattern
>>> plot_pattern(l, w, coords, 'surface')

Sample a 128x128 pixel image using a spiral pattern:

>>> from magni.imaging.measurements import spiral_sample_image
>>> h = 128; w = 128; scan_length = 1000.0; num_points = 200;
>>> coords = spiral_sample_image(h, w, scan_length, num_points)

Display the resulting pattern:

>>> plot_pattern(h, w, coords, 'image')

Find the corresponding unique pixels and plot the pixel mask:

>>> from magni.imaging.measurements import unique_pixels, plot_pixel_mask
>>> unique_pixels = unique_pixels(coords)
>>> plot_pixel_mask(h, w, unique_pixels)

"""

from magni.imaging.measurements._lissajous import *
from magni.imaging.measurements._matrices import *
from magni.imaging.measurements._random_line import *
from magni.imaging.measurements._spiral import *
from magni.imaging.measurements._square_spiral import *
from magni.imaging.measurements._uniform_line import *
from magni.imaging.measurements._uniform_rotated_line import *
from magni.imaging.measurements._util import *
from magni.imaging.measurements._visualisation import *
from magni.imaging.measurements._zigzag import *
