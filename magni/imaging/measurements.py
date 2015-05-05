"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions for constructing scan patterns for measurements.

This module provides several pairs of scan pattern functions. The first
function, named \*_sample_surface, is used for sampling a given surface. The
second function, named \*_sample_image, is a wrapper that provides a
pixel-oriented interface to the first function. In a addition to these pairs of
scan pattern functions, the module provides auxillary functions that may be
used to visualise the scan patterns.

Routine listings
----------------
construct_measurement_matrix(coords, h, w)
    Function for constructing a measurement matrix.
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
unique_pixels(coords)
    Function for determining unique pixels from a set of coordinates.

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
Sample a surface using a spiral pattern:

>>> from magni.imaging.measurements import spiral_sample_surface
>>> l = 13.0; w = 13.0; speed = 4.0; time = 27.0; sample_rate = 3.0;
>>> coords = spiral_sample_surface(l, w, speed, sample_rate, time)

Display the resulting pattern:

>>> from magni.imaging.measurements import plot_pattern
>>> plot_pattern(l, w, coords, 'surface')

Sample a 128x128 pixel image using random lines and a fixed seed:

>>> from magni.imaging.measurements import random_line_sample_image
>>> h = 128; w = 128; scan_length = 1000.0; num_points = 200; seed=6021;
>>> coords = random_line_sample_image(h, w, scan_length, num_points, seed=seed)

Display the resulting pattern:

>>> plot_pattern(h, w, coords, 'image')

Find the corresponding unique pixels and plot the pixel mask:

>>> from magni.imaging.measurements import unique_pixels, plot_pixel_mask
>>> unique_pixels = unique_pixels(coords)
>>> plot_pixel_mask(h, w, unique_pixels)

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from magni.imaging.visualisation import imshow as _imshow
from magni.utils.matrices import Matrix as _Matrix
from magni.utils import plotting as _plotting
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric

# In principle most of the AFM-scanning related parameters should just be
# positive, however we have settled for:
_min_l = 1e-9  # [m]
_min_w = 1e-9  # [m]
_min_speed = 1e-9  # [m/s]
_min_sample_rate = 1.0  # [Hz]
_min_time = 1.0  # [s]
_min_scan_length = 1e-9  # [m]
_min_num_points = 1  # []


def construct_measurement_matrix(coords, h, w):
    """
    Construct a measurement matrix extracting the specified measurements.

    Parameters
    ----------
    coords : ndarray
        The `k` floating point coordinates arranged into a 2D array where each
        row is a coordinate pair (x, y), such that `coords` has size `k` x 2.

    Returns
    -------
    Phi : magni.utils.matrices.Matrix
        The constructed measurement matrix.

    See Also
    --------
    magni.utils.matrices.Matrix : The matrix emulator class.

    Notes
    -----
    The function construct two functions: one for extracting pixels at the
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

    coords = unique_pixels(coords)
    mask = coords[:, 0] * w + coords[:, 1]

    def measure(vec):
        return vec[mask]

    def measure_T(vec):
        output = np.zeros((h * w, 1), dtype=vec.dtype)
        output[mask] = vec
        return output

    return _Matrix(measure, measure_T, [], (len(mask), h * w))


def plot_pattern(l, w, coords, mode, output_path=None):
    """
    Display a plot that shows the pattern given by a set of coordinates.

    The pattern given by the `coords` is displayed on an `w` x `l` area. If
    `mode` is 'surface', `l` and `w` are regarded as measured in meters. If
    `mode` is 'image', `l` and `w` are regarded as measured in pixels. The
    `coords` are marked by filled circles and connected by straight dashed
    lines.

    Parameters
    ----------
    l : float or int
        The length/height of the area. If `mode` is 'surface', it must be a
        float. If `mode` is 'image', it must be an integer.
    w : float or int
        The width of the area. If `mode` is 'surface', it must be a
        float. If `mode` is 'image', it must be an integer.
    coords : ndarray
        The 2D array of pixels that make up the mask. Each row is a coordinate
        pair (x, y).
    mode : {'surface', 'image'}
        The display mode that dertermines the axis labeling and the type of `l`
        and `w`.
    output_path : str, optional
        Path (including file type extension) under which the plot is saved (the
        default value is None which implies, that the plot is not saved).

    Notes
    -----
    The resulting plot is displayed in a figure using `matplotlib`'s
    `pyplot.plot`.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import plot_pattern
    >>> l = 3
    >>> w = 3
    >>> coords = np.array([[0, 0], [1, 1], [2, 1]])
    >>> mode = 'image'
    >>> plot_pattern(l, w, coords, mode)

    """

    @_decorate_validation
    def validate_input():
        _generic('mode', 'string', value_in=('surface', 'image'))

        if mode == 'surface':
            _numeric('l', 'floating', range_='[{};inf)'.format(_min_l))
            _numeric('w', 'floating', range_='[{};inf)'.format(_min_w))
            _numeric('coords', 'floating', shape=(-1, 2))
            _numeric('coords[:, 0]', 'floating', range_='[0;{}]'.format(w),
                     shape=(-1,), var=coords[:, 0])
            _numeric('coords[:, 1]', 'floating', range_='[0;{}]'.format(l),
                     shape=(-1,), var=coords[:, 1])
        elif mode == 'image':
            _numeric('l', 'integer', range_='[2;inf)')
            _numeric('w', 'integer', range_='[2;inf)')
            _numeric('coords', ('integer', 'floating'), shape=(-1, 2))
            _numeric('coords[:, 0]', ('integer', 'floating'),
                     range_='[0;{})'.format(w), shape=(-1,),
                     var=coords[:, 0])
            _numeric('coords[:, 1]', ('integer', 'floating'),
                     range_='[0;{})'.format(l), shape=(-1,),
                     var=coords[:, 1])

        _generic('output_path', 'string', ignore_none=True)

    validate_input()

    figsize = plt.rcParams['figure.figsize']

    if w / l > figsize[0] / figsize[1]:
        figsize_local = [figsize[0], figsize[0] * l / w]
    else:
        figsize_local = [figsize[1] * w / l, figsize[1]]

    _plotting.setup_matplotlib({'figure': {'figsize': figsize_local}})

    fig, axes = plt.subplots(1, 1)
    axes.plot(coords[:, 0], coords[:, 1],
              marker=_plotting.markers[0],
              ls=_plotting.linestyles[1],
              ms=6)

    axes.set_xlim([0, w])
    axes.xaxis.tick_top()
    axes.invert_yaxis()
    axes.set_ylim([l, 0])
    axes.xaxis.set_label_position('top')
    axes.set_aspect('equal')

    if mode == 'surface':
        axes.set_xlabel('width [m]')
        axes.set_ylabel('length [m]')
        axes.ticklabel_format(style='sci', scilimits=(0, 0))

        # Fix position of exponential multipliers in scientific notation
        exp_texts = [axes.xaxis.get_children()[1],
                     axes.yaxis.get_children()[1]]
        vert_aligns = ['bottom', 'center']
        hori_aligns = ['center', 'right']
        xy_texts = [(0, 25), (-35, 0)]
        xys = [(1, 1), (0, 0)]
        annos = [None, None]
        for k, text in enumerate(exp_texts):
            text.set_visible(False)
            annos[k] = axes.annotate('', xys[k], xytext=xy_texts[k],
                                     xycoords='axes fraction',
                                     textcoords='offset points',
                                     va=vert_aligns[k], ha=hori_aligns[k])

        fig.canvas.mpl_connect(
            'draw_event', lambda event: annos[0].set_text(
                exp_texts[0].get_text()))
        fig.canvas.mpl_connect(
            'draw_event', lambda event: annos[1].set_text(
                exp_texts[1].get_text()))

    elif mode == 'image':
        axes.set_xlabel('width [pixels]')
        axes.set_ylabel('height [pixels]')

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)

    _plotting.setup_matplotlib({'figure': {'figsize': figsize}})


def plot_pixel_mask(h, w, pixels, output_path=None):
    """
    Display a binary image that shows the given pixel mask.

    A black image with `w` x `h` pixels is created and the `pixels` are marked
    with white.

    Parameters
    ----------
    h : int
        The height of the image in pixels.
    w : int
        The width of the image in pixels.
    pixels : ndarray
        The 2D array of pixels that make up the mask. Each row is a coordinate
        pair (x, y), such that `coords` has size len(`pixels`) x 2.
    output_path : str, optional
        Path (including file type extension) under which the plot is saved (the
        default value is None which implies, that the plot is not saved).

    Notes
    -----
    The resulting image is displayed in a figure using
    `magni.imaging.visualisation.imshow`.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.measurements import plot_pixel_mask
    >>> h = 3
    >>> w = 3
    >>> pixels = np.array([[0, 0], [1, 1], [2, 1]])
    >>> plot_pixel_mask(h, w, pixels)

    """

    @_decorate_validation
    def validate_input():
        _numeric('h', 'integer', range_='[2;inf)')
        _numeric('w', 'integer', range_='[2;inf)')
        _numeric('pixels', 'integer', shape=(-1, 2))
        _numeric('pixels[:, 0]', 'integer', range_='[0;{}]'.format(w - 1),
                 shape=(-1,), var=pixels[:, 0])
        _numeric('pixels[:, 1]', 'integer', range_='[0;{}]'.format(h - 1),
                 shape=(-1,), var=pixels[:, 1])
        _generic('output_path', 'string', ignore_none=True)

    validate_input()

    mask = np.zeros((h, w))
    mask[pixels[:, 1], pixels[:, 0]] = 1

    figsize = plt.rcParams['figure.figsize']

    if w / h > figsize[0] / figsize[1]:
        figsize_local = [figsize[0], figsize[0] * h / w]
    else:
        figsize_local = [figsize[1] * w / h, figsize[1]]

    _plotting.setup_matplotlib({'figure': {'figsize': figsize_local}})

    fig, axes = plt.subplots(1, 1)
    _imshow(mask, ax=axes, show_axis='top')

    axes.set_xlabel('width [pixels]')
    axes.set_ylabel('height [pixels]')

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)

    _plotting.setup_matplotlib({'figure': {'figsize': figsize}})


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

    samples = int(sample_rate * time) + 1
    scan_length = np.floor((speed * time - l) / w) * w + l
    sample_dist = scan_length / (samples - 1)
    lines_scan_length = scan_length - 2 * w - l
    num_lines = np.int(np.round(lines_scan_length / w))

    if seed is not None:
        np.random.seed(seed)
    lines = np.zeros(num_lines + 2)
    lines[-1] = l
    if discrete is None:
        lines[1:-1] = np.sort(np.random.rand(num_lines) * l)
    else:
        possible_lines = l / (discrete + 1) * np.arange(1, discrete + 1)
        try:
            lines[1:-1] = np.sort(np.random.choice(possible_lines,
                                                   size=num_lines,
                                                   replace=False))
        except ValueError:
            raise ValueError('The number of Discrete lines must be large ' +
                             'enough to contain the entire scan path. With ' +
                             'the current settings, a minimun of '
                             '{!r} lines are required.'.format(num_lines))

    coords = _get_line_scan_coords(lines, samples, sample_dist, l, w)

    return coords


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

    samples = int(sample_rate * time) + 1
    scan_length = np.floor((speed * time - l) / w) * w + l
    sample_dist = scan_length / (samples - 1)
    lines_scan_length = scan_length - l
    num_lines = int(np.round(lines_scan_length / w))

    # We should always at least partially scan top and bottom lines.
    if num_lines < 2:
        num_lines = 2

    lines = np.linspace(0, l, num=num_lines)

    coords = _get_line_scan_coords(lines, samples, sample_dist, l, w)

    return coords


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


def _get_line_scan_coords(lines, samples, sample_dist, l, w):
    """
    Determine the coordinates of the sampled lines in a line scanning.

    Parameters
    ----------
    lines : ndarray
        The position (vertical distance from top of scan area) of the lines to
        scan.
    samples : int
        The number of samples on the scan path.
    sample_dist : float
        The distance (path length) between samples.
    l : float
        The length of the area to scan in units of meters.
    w : float
        The width of the area to scan in units of meters.

    Returns
    -------
    coords : ndarray
        The w and l coordinates arranged into a N_samples-by-2 array.

    """

    dir_right = np.float64((1, 0))
    dir_left = np.float64((-1, 0))
    dir_up = np.float64((0, 1))

    coords = np.zeros((samples, 2))
    position = np.float64((0, 0))
    direction = dir_right

    dist_target = w
    dist_epsilon = 1e-6 * sample_dist
    line = 0

    for sample in range(1, samples):
        dist_remaining = sample_dist

        while dist_remaining > 0:
            if dist_remaining <= dist_target:
                position = position + dist_remaining * direction
                dist_target = dist_target - dist_remaining
                dist_remaining = 0
            elif np.abs(dist_remaining - dist_target) < dist_epsilon:
                position = position + dist_target * direction
                dist_target = 0
                dist_remaining = 0
            else:
                position = position + dist_target * direction
                # dist_target = 0
                dist_remaining = dist_remaining - dist_target

                if ((direction == dir_right).all() or
                        (direction == dir_left).all()):
                    direction = dir_up
                    dist_target = lines[line + 1] - lines[line]
                    line = line + 1
                else:
                    direction = dir_right if line % 2 == 0 else dir_left
                    dist_target = w
                    # line = line

        coords[sample, :] = position

    coords[coords < 0] = 0
    coords[:, 0][coords[:, 0] > w] = w
    coords[:, 1][coords[:, 1] > l] = l

    return coords
