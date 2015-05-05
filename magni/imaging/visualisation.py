"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality for visualising images.

The module provides functionality for adjusting the intensity of an image. It
provides a wrapper of the `matplotlib.pyplot.imshow` function that may exploit
the provided functions for adjusting the image intensity. Also it include a
function may be used to display a set of related images using a common
colormapping.

Routine listings
----------------
imshow(X, ax=None, intensity_func=None, intensity_args=(), \*\*kwargs)
    Function that may be used to display an image.
imsubplot(imgs, rows, titles=None, x_labels=None, y_labels=None,
    x_ticklabels=None, y_ticklabels=None, cbar_label=None,
    normalise=True, \*\*kwargs)
    Function that may be used to display a set of related images.
mask_img_from_coords(img, coords)
    Function for masking certain parts of an image based on coordinates.
shift_mean(x_mod, x_org)
    Function for shifting mean intensity of an image based on another image.
stretch_image(img, max_val, min_val=0)
    Function for stretching the intensity of an image.

"""

from __future__ import division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from magni.utils import plotting as _plotting
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


def imshow(X, ax=None, intensity_func=None, intensity_args=(),
           show_axis='frame', **kwargs):
    """
    Display an image.

    Wrap `matplotlib.pyplot.imshow` to display a possibly intensity manipulated
    version of the image `X`.

    Parameters
    ----------
    X : ndarray
        The image to be displayed.
    ax : matplotlib.axes.Axes, optional
        The axes on which the image is displayed (the default is None, which
        implies that the current axes is used).
    intensity_func : FunctionType, optional
        The handle to the function used to manipulate the image intensity
        before the image is displayed (the default is None, which implies that
        no intensity manipulation is used).
    intensity_args : list or tuple, optional
        The arguments that are passed to the `intensity_func` (the default is
        (), which implies that no arguments are passed).
    show_axis : {'none', 'top', 'inherit', 'frame'}
        How the x- and y-axis are display. If 'none', no axis are displayed. If
        'top', the x-axis is displayed at the top of the image. If 'inherit',
        the axis display is inherited from `matplotlib.pyplot.imshow`. If
        'frame' only the frame is shown and not the ticks.

    Returns
    -------
    im_out : matplotlib.image.AxesImage
        The AxesImage returned by matplotlibs imshow.

    See Also
    --------
    matplotlib.pyplot.imshow : Matplotlib's imshow function.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.visualisation import imshow
    >>> X = np.arange(4).reshape(2, 2)
    >>> add_k = lambda X, k: X + k
    >>> im_out = imshow(X, intensity_func=add_k, intensity_args=(2,))

    """

    @_decorate_validation
    def validate_input():
        _numeric('X', ('boolean', 'integer', 'floating'), shape=(-1, -1))
        _generic('ax', mpl.axes.Axes, ignore_none=True)
        _generic('intensity_func', 'function', ignore_none=True)
        _generic('intensity_args', 'explicit collection')
        _generic('show_axis', 'string', value_in=('none', 'top', 'inherit',
                                                  'frame'))

    validate_input()

    _plotting.setup_matplotlib()

    # Handle ax keyword argument
    ca = plt.gca()
    if ax is not None:
        plt.sca(ax)

    # Intensity manipulation
    if intensity_func is not None:
        im_out = plt.imshow(intensity_func(X, *intensity_args), **kwargs)
    else:
        im_out = plt.imshow(X, **kwargs)

    # Display of axis
    axes = plt.gca()
    if show_axis == 'none':
        axes.axis('off')
    elif show_axis == 'top':
        axes.xaxis.tick_top()
        axes.xaxis.set_label_position('top')
    elif show_axis == 'frame':
        xlabels = axes.get_xticklabels()
        ylabels = axes.get_yticklabels()
        empty_xlabels = ['']*len(xlabels)
        empty_ylabels = ['']*len(ylabels)
        axes.set_xticklabels(empty_xlabels)
        axes.set_yticklabels(empty_ylabels)
        axes.tick_params(length=0)

    plt.sca(ca)

    return im_out


def imsubplot(imgs, rows, titles=None, x_labels=None, y_labels=None,
              x_ticklabels=None, y_ticklabels=None, cbar_label=None,
              normalise=True, **kwargs):
    """
    Display a set of related images as subplots in a figure.

    The images `imgs` are shown in a figure with a subplot layout based on the
    number of `rows`. The `titles`, `x_labels`, and `y_labels` are shown in the
    subplots. If `normalise` is True, all the images will share the same
    normalised colourbar/colormapping i.e. a particular colour will correspond
    to the same value across all images.

    Parameters
    ----------
    imgs : list or tuple
        The images (as ndarrays) that is to be displayed.
    rows : int
        The number of rows to use in the subplot layout.
    titles : list or tuple
        The titles (as strings) to use for each of the subplots (the default is
        None, which implies that no titles are displayed).
    x_labels : list or tuple
        The x_labels (as strings) to use for each of the subplots (the default
        is None, which implies that no x_labels are displayed).
    y_labels : list or tuple
        The y_labels (as strings) to use for each of the subplots (the default
        is None, which implies that no y_labels are displayed).
    x_ticklabels : list or tuple
        The x_ticklabels (as strings) to share across the subplots (the default
        is None, which implies that no x_ticklabels are displayed).
    y_ticklabels : list or tuple
        The y_ticklabels (as strings) to share across the subplots (the default
        is None, which implies tht no y_ticklabels are displayed).
    cbar_label : str
        The colorbar label to use with a normalised colormapping (the default
        is None, which implies that no colorbar label is displayed).
    normalise : bool
        The flag that indicates whether to use a normalised colormapping.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure instance.

    See Also
    --------
    matplotlib.pyplot.subplots : Underlying subplot function.

    Notes
    -----
    Additional kwargs given to the function will be passed to the underlying
    suplot instantiation function `matplotlib.pyplot.subplots`.

    If `normalise` is True, the common colorbar is shown below the subplots.

    The implementation of the normalisation feature is based on the Pylab
    example: http://matplotlib.org/examples/pylab_examples/multi_image.html.

    Examples
    --------
    For example, show to images next to each other with a common colormapping:

    >>> import numpy as np
    >>> from magni.imaging.visualisation import imsubplot
    >>> img1 = np.arange(4).reshape(2, 2)
    >>> img2 = np.ones((4, 4))
    >>> fig = imsubplot([img1, img2], 1, titles=['arange', 'ones'],
    ... x_labels=['x1', 'x2'], y_labels=['y1', 'y2'], normalise=True)

    """

    @_decorate_validation
    def validate_input():
        _levels('imgs', (_generic(None, 'explicit collection'),
                         _numeric(None, ('boolean', 'integer', 'floating'),
                                  shape=(-1, -1))))
        _numeric('rows', 'integer', range_='[1;inf)')
        _levels('titles', (_generic(None, 'explicit collection',
                                    len_=len(imgs), ignore_none=True),
                           _generic(None, 'string')))
        _levels('x_labels', (_generic(None, 'explicit collection',
                                      len_=len(imgs), ignore_none=True),
                             _generic(None, 'string')))
        _levels('y_labels', (_generic(None, 'explicit collection',
                                      len_=len(imgs), ignore_none=True),
                             _generic(None, 'string')))
        _levels('x_ticklabels', (_generic(None, 'explicit collection',
                                          ignore_none=True),
                                 _generic(None, 'string')))
        _levels('y_ticklabels', (_generic(None, 'explicit collection',
                                          ignore_none=True),
                                 _generic(None, 'string')))
        _generic('cbar_label', 'string', ignore_none=True)
        _numeric('normalise', 'boolean')

    validate_input()

    cols = max(1, np.int(np.ceil(len(imgs) / rows)))
    fig, axes = plt.subplots(rows, cols, squeeze=False, **kwargs)
    axs = axes.ravel()

    vmin = 1e40
    vmax = -1e40
    ims = []
    for k, img in enumerate(imgs):
            ims.append(imshow(img, ax=axs[k]))

            fig_strings = (titles, x_labels, y_labels)
            handles = (axs[k].set_title, axs[k].set_xlabel, axs[k].set_ylabel)
            for fig_string, handle in zip(fig_strings, handles):
                if fig_string is not None:
                    handle(fig_string[k])

            if x_ticklabels is not None:
                axs[k].set_xticks(range(len(x_ticklabels)))
                axs[k].set_xticklabels(x_ticklabels, rotation=90)

            if y_ticklabels is not None:
                axs[k].set_yticks(range(len(y_ticklabels)))
                axs[k].set_yticklabels(y_ticklabels)

            vmin = min(vmin, np.amin(img))  # Find common minimum
            vmax = max(vmax, np.amax(img))  # Find common maximum

    if normalise:
        # Connect a Tracker to each image in order to update colormap limits
        common_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        ims[0].set_norm(common_norm)
        for k in range(1, len(ims)):
            ims[k].set_norm(common_norm)
            ims[0].callbacksSM.connect('changed', _ImageColourTracker(ims[k]))

        plt.subplots_adjust(bottom=0.2)
        c_bar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
        c_bar = fig.colorbar(ims[0], c_bar_ax, orientation='horizontal')
        c_bar.solids.set_edgecolor("face")
        if cbar_label is not None:
            c_bar.set_label(cbar_label)

    return fig


def mask_img_from_coords(img, coords):
    """
    Mask coordinates in an image.

    The coordinates `coords` in the image `img` are masked such that only the
    cooordinates are shown.

    Parameters
    ----------
    img : ndarray
        The image to mask.
    coords : ndarray
        The coordinates arranged into a 2D array, such that each row is a
        coordinate pair (x, y).

    Returns
    -------
    masked_img : numpy.ma.MaskedArray
        The masked image.

    See Also
    --------
    magni.imaging.measurements : Further description of the coordinate format.

    Examples
    --------
    For example, display only center pixel in a 3-by-3 image

    >>> import numpy as np
    >>> from magni.imaging.visualisation import mask_img_from_coords
    >>> img = np.arange(9).reshape(3, 3)
    >>> coords = np.array([[1, 1]])
    >>> mask_img_from_coords(img, coords)
    masked_array(data =
     [[-- -- --]
     [-- 4 --]
     [-- -- --]],
                 mask =
     [[ True  True  True]
     [ True False  True]
     [ True  True  True]],
           fill_value = 999999)
    <BLANKLINE>

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _numeric('coord', 'integer', shape=(-1, 2), range_='(0;inf)')

        validate_input()

    mask = np.ones_like(img, dtype=np.bool_)
    mask[coords[:, 1], coords[:, 0]] = False
    masked_img = np.ma.array(img, mask=mask)

    return masked_img


def shift_mean(x_mod, x_org):
    """
    Shift the mean value of `x_mod` such that it equals the mean of `x_org`.

    Parameters
    ----------
    x_org : ndarray
        The array which hold the "true" mean value.
    x_mod : ndarray
        The modified copy of `x_org` which must have its mean value shifted.

    Returns
    -------
    shifted_x_mod : ndarray
        A copy of `x_mod` with the same mean value as `x_org`.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.visualisation import shift_mean
    >>> x_org = np.arange(4).reshape(2, 2)
    >>> x_mod = np.ones((2, 2))
    >>> print('{:.1f}'.format(x_org.mean()))
    1.5
    >>> print('{:.1f}'.format(x_mod.mean()))
    1.0
    >>> shifted_x_mod = shift_mean(x_mod, x_org)
    >>> print('{:.1f}'.format(shifted_x_mod.mean()))
    1.5
    >>> np.set_printoptions(suppress=True)
    >>> shifted_x_mod
    array([[ 1.5,  1.5],
           [ 1.5,  1.5]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('x_mod', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _numeric('x_org', ('integer', 'floating', 'complex'),
                 shape=x_mod.shape)

    validate_input()

    return x_mod + (x_org.mean() - x_mod.mean())


def stretch_image(img, max_val, min_val=0):
    """
    Stretch image such that pixels values are in [`min_val`, `max_val`].

    Parameters
    ----------
    img : ndarray
        The (float) image that is to be stretched.
    max_val : int or float
        The maximum value in the stretched image.
    min_val : int or float
        The minimum value in the stretched image.

    Returns
    -------
    stretched_img : ndarray
        A stretched copy of the input image.

    Notes
    -----
    The pixel values in the input image are scaled to lie in the interval
    [`min_val`, `max_val`] using a linear stretch.

    Examples
    --------
    For example, stretch an image between 0 and 1

    >>> import numpy as np
    >>> from magni.imaging.visualisation import stretch_image
    >>> img = np.arange(4, dtype=np.float).reshape(2, 2)
    >>> stretched_img = stretch_image(img, 1)
    >>> np.set_printoptions(suppress=True)
    >>> stretched_img
    array([[ 0.        ,  0.33333333],
           [ 0.66666667,  1.        ]])

    or stretch the image between -1 and 1

    >>> stretched_img = stretch_image(img, 1.0, min_val=-1.0)
    >>> stretched_img
    array([[-1.        , -0.33333333],
           [ 0.33333333,  1.        ]])

    or re-stretch the strecthed image between -3.0 and -1.5

    >>> stretched_img = stretch_image(stretched_img, -1.5, min_val=-3.0)
    >>> stretched_img
    array([[-3. , -2.5],
           [-2. , -1.5]])

    or re-stretch that image between 1.25 and 8.00

    >>> stretched_img = stretch_image(stretched_img, 8.00, min_val=1.25)
    >>> stretched_img
    array([[ 1.25,  3.5 ],
           [ 5.75,  8.  ]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', 'floating', shape=(-1, -1))
        _numeric('max_val', ('integer', 'floating'))
        _numeric('min_val', ('integer', 'floating'))

        if not max_val > min_val:
            msg = 'max_val ({!r}) must be larger than min_val ({!r})'
            raise ValueError(msg.format(max_val, min_val))

    validate_input()

    min_ = img.min()
    max_ = img.max()

    a = (max_val - min_val) / (max_ - min_)
    b = -a * min_ + min_val

    return a * img + b


class _ImageColourTracker():
    """
    Track a common 'clim' in a set of matplotlib image subplots.

    Parameters
    ----------
    tracker : matplotlib.image.AxesImage
        The image instance that must track a given 'clim'.

    """

    def __init__(self, tracker):
        self.tracker = tracker

    def __call__(self, tracked):
        self.tracker.set_clim(tracked.get_clim())
