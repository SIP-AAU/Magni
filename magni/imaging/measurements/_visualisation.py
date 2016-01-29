"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing public functions for the magni.imaging.measurements
subpackage.

Routine listings
----------------
plot_pattern(l, w, coords, mode, output_path=None)
    Function for visualising a scan pattern.
plot_pixel_mask(h, w, pixels, output_path=None)
    Function for visualising a pixel mask obtained from a scan pattern.

"""

from __future__ import division

import matplotlib.pyplot as plt

from magni.imaging.measurements import _util
from magni.imaging.visualisation import imshow as _imshow
from magni.utils import plotting as _plotting
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric


__all__ = ['plot_pattern', 'plot_pixel_mask']

_min_l = _util.min_l
_min_w = _util.min_w


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

    mask = _util.construct_pixel_mask(h, w, pixels)

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
