"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions for evaluation of image reconstruction quality.

Routine listings
----------------
calculate_mse(x_org, x_recons)
    Function to calcualte Mean Squared Error (MSE).
calculate_psnr(x_org, x_recons, peak)
    Function to calculate Peak Signal to Noise Ratio (PSNR).
calculate_retained_energy(x_org, x_recons)
    Function to calculate the percentage of energy retained in reconstruction.

"""

from __future__ import division

import numpy as np

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate as _validate
from magni.utils.validation import validate_ndarray as _validate_ndarray


@_decorate_validation
def _validate_calculate_mse(x_org, x_recons):
    """
    Validate the `calculate_mse` function.

    See Also
    --------
    magni.utils.validation.validate : Validation.

    """

    _validate_ndarray(x_org, 'x_org', {})
    _validate_ndarray(x_recons, 'x_recons', {'shape': x_org.shape})


def calculate_mse(x_org, x_recons):
    r"""
    Calculate Mean Squared Error (MSE) between `x_recons` and `x_org`.

    Parameters
    ----------
    x_org : ndarray
        Array of original values.
    x_recons : ndarray
        Array of reconstruction values.

    Returns
    -------
    mse : float
        Mean Squared Error (MSE).

    Notes
    -----
    The Mean Squared Error (MSE) is calculated as:

    .. math::

         \frac{1}{N} \cdot \sum(x_{org} - x_{recons})^2

    where `N` is the number of entries in `x_org`.

    Examples
    --------
    For example,

    >>> from magni.imaging.evaluation import calculate_mse
    >>> x_org = np.arange(4).reshape(2, 2)
    >>> x_recons = np.ones((2,2))
    >>> calculate_mse(x_org, x_recons)
    1.5

    """

    _validate_calculate_retained_energy(x_org, x_recons)

    return ((x_org - x_recons) ** 2).mean()


@_decorate_validation
def _validate_calculate_psnr(x_org, x_recons, peak):
    """
    Validate the `calculate_psnr` function.

    See Also
    --------
    magni.utils.validation.validate : Validation.

    """

    _validate_ndarray(x_org, 'x_org', {})
    _validate_ndarray(x_recons, 'x_recons', {'shape': x_org.shape})
    _validate(peak, 'peak', {'type_in': [float, int], 'min': 0})


def calculate_psnr(x_org, x_recons, peak):
    r"""
    Calculate Peak Signal to Noise Ratio (PSNR) between `x_recons` and `x_org`.

    Parameters
    ----------
    x_org : ndarray
        Array of original values.
    x_recons : ndarray
        Array of reconstruction values.
    peak : int or float
        Peak value.

    Returns
    -------
    psnr : float
        Peak Signal to Noise Ratio (PSNR) in dB.

    Notes
    -----
    The PSNR is as calculated as

    .. math::

         10 \cdot \log_{10}\left(\frac{peak^2}{ 1/N \cdot \sum(x_{org} -
         x_{recons})^2}\right)

    where `N` is the number of entries in `x_org`.

    Examples
    --------
    For example,

    >>> from magni.imaging.evaluation import calculate_psnr
    >>> x_org = np.arange(4).reshape(2, 2)
    >>> x_recons = np.ones((2,2))
    >>> peak = 3
    >>> calculate_psnr(x_org, x_recons, peak)
    7.7815125038364368

    """

    _validate_calculate_psnr(x_org, x_recons, peak)

    return 10 * np.log10(peak**2 / ((x_org - x_recons) ** 2).mean())


@_decorate_validation
def _validate_calculate_retained_energy(x_org, x_recons):
    """
    Validate the `calculate_retained_energy` function.

    See Also
    --------
    magni.utils.validation.validate : Validation.

    """

    _validate_ndarray(x_org, 'x_org', {})
    _validate_ndarray(x_recons, 'x_recons', {'shape': x_org.shape})


def calculate_retained_energy(x_org, x_recons):
    r"""
    Calculate percentage of energy retained in reconstruction.

    Parameters
    ----------
    x_org : ndarray
        Array of original values.
    x_recons : ndarray
        Array of reconstruction values.

    Returns
    -------
    energy : float
        Percentage of retained energy in reconstruction.

    Notes
    -----
    The retained energy is as calculated as

    .. math::
         \frac{\sum x_{recons}^2}{\sum x_{org}^2} \cdot 100\%

    Examples
    --------
    For example,

    >>> from magni.imaging.evaluation import calculate_retained_energy
    >>> x_org = np.arange(4).reshape(2, 2)
    >>> x_recons = np.ones((2,2))
    >>> calculate_retained_energy(x_org, x_recons)
    28.571428571428569

    """

    _validate_calculate_retained_energy(x_org, x_recons)

    return (x_recons ** 2).sum() / (x_org ** 2).sum() * 100
