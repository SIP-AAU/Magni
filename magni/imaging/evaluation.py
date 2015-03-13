"""
..
    Copyright (c) 2014-2015, Magni developers.
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
from magni.utils.validation import validate_numeric as _numeric


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

    >>> import numpy as np
    >>> from magni.imaging.evaluation import calculate_mse
    >>> x_org = np.arange(4).reshape(2, 2)
    >>> x_recons = np.ones((2,2))
    >>> print('{:.2f}'.format(calculate_mse(x_org, x_recons)))
    1.50

    """

    @_decorate_validation
    def validate_input():
        _numeric('x_org', ('integer', 'floating'), shape=None)
        _numeric('x_recons', ('integer', 'floating'), shape=x_org.shape)

    validate_input()

    return ((x_org - x_recons) ** 2).mean()


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

    If :math:`|x_{org} - x_{recons}| <= (10^{-8} + 1^{-5} * |x_{recons}|)`
    then `np.inf` is returned.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.imaging.evaluation import calculate_psnr
    >>> x_org = np.arange(4).reshape(2, 2)
    >>> x_recons = np.ones((2,2))
    >>> peak = 3
    >>> print('{:.2f}'.format(calculate_psnr(x_org, x_recons, peak)))
    7.78

    """

    @_decorate_validation
    def validate_input():
        _numeric('x_org', ('integer', 'floating'), shape=None)
        _numeric('x_recons', ('integer', 'floating'), shape=x_org.shape)
        _numeric('peak', ('integer', 'floating'), range_='(0;inf)')

    validate_input()

    if np.allclose(x_org, x_recons):
        psnr = np.inf
    else:
        psnr = 10 * np.log10(peak**2 / ((x_org - x_recons) ** 2).mean())

    return psnr


def calculate_retained_energy(x_org, x_recons):
    r"""
    Calculate percentage of energy retained in reconstruction.

    Parameters
    ----------
    x_org : ndarray
        Array of original values (must not be all zeros).
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

    >>> import numpy as np
    >>> from magni.imaging.evaluation import calculate_retained_energy
    >>> x_org = np.arange(4).reshape(2, 2)
    >>> x_recons = np.ones((2,2))
    >>> print('{:.2f}'.format(calculate_retained_energy(x_org, x_recons)))
    28.57

    """

    @_decorate_validation
    def validate_input():
        _numeric('x_org', ('integer', 'floating'), shape=None)

        if np.count_nonzero(x_org) == 0:
            raise ValueError('x_org must not be all zeros')

        _numeric('x_recons', ('integer', 'floating'), shape=x_org.shape)

    validate_input()

    return (x_recons ** 2).sum() / (x_org ** 2).sum() * 100
