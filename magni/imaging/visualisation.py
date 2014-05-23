"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality for visualising images.

The module provides functionality for adjusting the intensity of an image.
Furthermore, it provides a wrapper of the `matplotlib.pyplot.imshow` function
that may exploit the provided functions for adjusting the image intensity.

Routine listings
----------------
imshow(X, ax=None, intensity_func=None, intensity_args=(), \*\*kwargs)
    Function that may be used to display an image.
shift_mean(x_mod, x_org)
    Function for shifting mean intensity of an image based on another image.
stretch_image(img, max_val)
    Function for stretching the intensity of an image.

"""

from __future__ import division
import types

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from magni.utils import plotting as _plotting
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate as _validate
from magni.utils.validation import validate_ndarray as _validate_ndarray


@_decorate_validation
def _validate_imshow(X, ax, intensity_func, intensity_args, show_axis):
    """
    Validate the `imshow` function.

    See Also
    --------
    imshow : The validated function.
    magni.utils.validation.validate : Validation.

    """

    _validate_ndarray(X, 'X')
    _validate(ax, 'ax', {'class': mpl.axes.Axes}, ignore_none=True)
    _validate(intensity_func, 'intensity_func', {'type': types.FunctionType},
              ignore_none=True)
    _validate(intensity_args, 'intensity_args', {'type_in': (list, tuple)})
    _validate(show_axis, 'show_axis', {'type': str,
                                       'val_in': ('none', 'top', 'inherit')})


def imshow(X, ax=None, intensity_func=None, intensity_args=(),
           show_axis='none', **kwargs):
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
    show_axis : {'none', 'top', 'inherit'}
        How the x- and y-axis are display. If 'none', no axis are displayed. If
        'top', the x-axis is displayed at the top of the image. If 'inherit',
        the axis display is inherited from `matplotlib.pyplot.imshow`.

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

    >>> from magni.imaging.visualisation import imshow
    >>> X = np.arange(4).reshape(2, 2)
    >>> add_k = lambda X, k: X + k
    >>> im_out = imshow(X, intensity_func=add_k, intensity_args=(2,))

    """

    _validate_imshow(X, ax, intensity_func, intensity_args, show_axis)

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
        xlabels = axes.get_xticklabels()
        ylabels = axes.get_yticklabels()
        empty_xlabels = ['']*len(xlabels)
        empty_ylabels = ['']*len(ylabels)
        axes.set_xticklabels(empty_xlabels)
        axes.set_yticklabels(empty_ylabels)
        axes.tick_params(length=0)
    elif show_axis == 'top':
        axes.xaxis.tick_top()
        axes.xaxis.set_label_position('top')

    plt.sca(ca)

    return im_out


@_decorate_validation
def _validate_shift_mean(x_mod, x_org):
    """
    Validate the `shift_mean` function.

    See Also
    --------
    shift_mean : The validated function.
    magni.utils.validation.validate : Validation.

    """

    _validate_ndarray(x_org, 'x_mod', {})
    _validate_ndarray(x_mod, 'x_org', {'dtype': x_mod.dtype,
                                       'shape': x_mod.shape})


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

    >>> from magni.imaging.visualisation import shift_mean
    >>> x_org = np.arange(4).reshape(2, 2)
    >>> x_mod = np.ones((2, 2))
    >>> x_org.mean()
    1.5
    >>> x_mod.mean()
    1.0
    >>> shifted_x_mod = shift_mean(x_mod, x_org)
    >>> shifted_x_mod.mean()
    1.5
    >>> shifted_x_mod
    array([[ 1.5,  1.5],
           [ 1.5,  1.5]])

    """

    _validate_shift_mean(x_mod, x_org)

    return x_mod + (x_org.mean() - x_mod.mean())


@_decorate_validation
def _validate_stretch_image(img, max_val):
    """
    Validate the `stretch_image` function.

    See Also
    --------
    stretch_image : The validated function.
    magni.utils.validation.validate : Validation.

    """

    _validate_ndarray(img, 'img', {'subdtype': np.floating})
    _validate(max_val, 'max_val', {'type_in': [int, float], 'min': 0})


def stretch_image(img, max_val):
    """
    Stretch image such that pixels values are in the range [0, `max_val`].

    Parameters
    ----------
    img : ndarray
        The (float) image that is to be stretched.
    max_val : int or float
        The maximum value in the stretched image.

    Returns
    -------
    stretched_img : ndarray
        A stretched copy of the input image.

    Notes
    -----
    The pixel values in the input image are scaled to lie in the interval [0,
    `max_val`] using a linear stretch.

    Examples
    --------
    For example,

    >>> from magni.imaging.visualisation import stretch_image
    >>> img = np.arange(4, dtype=np.float).reshape(2, 2)
    >>> stretched_img = stretch_image(img, 1.0)
    >>> stretched_img
    array([[ 0.        ,  0.33333333],
           [ 0.66666667,  1.        ]])

    """

    _validate_stretch_image(img, max_val)

    min_ = img.min()
    max_ = img.max()

    return max_val / (max_ - min_) * (img - min_)
