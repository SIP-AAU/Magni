"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality for visualising dictionary coefficients.

Routine listings
----------------
visualise_DCT(shape)
    Function for visualising DCT coefficients.
visualise_DFT(shape)
    Function for visualising DFT coefficients.

"""

from __future__ import division

import numpy as np

from magni.imaging import vec2mat as _vec2mat


def visualise_DCT(shape):
    """
    Return utilities for visualising DCT coefficients.

    A handle to a function to transform the coefficients into a 'displayable'
    format is returned along with a tuple of ranges of the axes in the 2D
    coefficient plane.

    Parameters
    ----------
    shape : tuple
        The shape of the 2D DCT being visualised.

    Returns
    -------
    display_coefficients : Function
        The function used to transform coefficients into a 'displayable'
        format.
    axes_extent : tuple
        The ranges of the axes in the 2D coefficient plane.

    Notes
    -----
    The display_coefficients function takes log10 to the absolute value of the
    transform cofficient vector given to it as an argument. The returned
    displayable coefficients is a matrix.

    The axes_extent consists of (abcissa_min, abcissa_max, ordinate_min,
    ordinate_max).

    """

    h, w = shape

    def display_coefficients(x):
        return np.log10(np.abs(_vec2mat(x, (h, w))))

    axes_extent = (0, w - 1, h - 1, 0)

    return display_coefficients, axes_extent


def visualise_DFT(shape):
    """
    Return utilities for visualising DFT coefficients.

    A handle to a function to transform the coefficients into a 'displayable'
    format is returned along with a tuple of ranges of the axes in the 2D
    coefficient plane.

    Parameters
    ----------
    shape : tuple
        The shape of the 2D DFT being visualised.

    Returns
    -------
    display_coefficients : Function
        The function used to transform coefficients into a 'displayable'
        format.
    axes_extent : tuple
        The ranges of the axes in the 2D coefficient plane.

    Notes
    -----
    The display_coefficients function takes log10 to the absolute value of the
    transform cofficient vector given to it as an argument. The returned
    displayable coefficients is a matrix that is flipped up/down and
    fftshifted.

    The axes_extent consists of (abcissa_min, abcissa_max, ordinate_min,
    ordinate_max).

    """

    h, w = shape

    def display_coefficients(x):
        return np.flipud(
            np.fft.fftshift(np.log10(np.abs(_vec2mat(x, (h, w))))))

    axes_extent = (-(w // 2), (w - 1) // 2, -(h // 2), (h - 1) // 2)

    return display_coefficients, axes_extent
