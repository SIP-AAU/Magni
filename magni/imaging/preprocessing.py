"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality to remove tilt in images.

Routine listings
----------------
detilt(img, mask=None, mode='plane_flatten', degree=1, return_tilt=False)
    Function to remove tilt from an image.

"""

from __future__ import division

import numpy as np

import magni.imaging
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric


def detilt(img, mask=None, mode='plane_flatten', degree=1, return_tilt=False):
    """
    Estimate the tilt in an image and return the detilted image.

    Parameters
    ----------
    img : ndarray
        The image that is to be detilted.
    mask : ndarray, optional
        Bool array of the same size as `img` indicating the pixels to use in
        detilt (the default is None, which implies, that the the entire image
        is used)
    mode : {'line_flatten', 'plane_flatten'}, optional
        The type of detilting applied (the default is plane_flatten).
    degree : int, optional
        The degree of the polynomial used in line flattening
        (the default is 1).
    return_tilt : bool, optional
        If True, the detilted image and the estimated tilt is returned (the
        default is False).

    Returns
    -------
    img_detilt : ndarray
        Detilted image.
    tilt : ndarray, optional
        The estimated tilt (image). Only returned if return_tilt is True.

    Notes
    -----
    If `mode` is line flatten, the tilt in each horizontal line of pixels in
    the image is estimated by a polynomial fit independently of all other
    lines. If `mode` is plane flatten, the tilt is estimated by fitting a plane
    to all pixels.

    If a custom `mask` is specified, only the masked (True) pixels are used in
    the estimation of the tilt.

    Examples
    --------
    For example, line flatten an image using a degree 1 polynomial

    >>> import numpy as np
    >>> from magni.imaging.preprocessing import detilt
    >>> img = np.array([[0, 2, 3], [1, 5, 7], [3, 6, 8]], dtype=np.float)
    >>> np.set_printoptions(suppress=True)
    >>> detilt(img, mode='line_flatten', degree=1)
    array([[-0.16666667,  0.33333333, -0.16666667],
           [-0.33333333,  0.66666667, -0.33333333],
           [-0.16666667,  0.33333333, -0.16666667]])

    Or plane flatten the image based on a mask and return the tilt

    >>> mask = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 1]], dtype=np.bool)
    >>> im, ti = detilt(img, mask=mask, mode='plane_flatten', return_tilt=True)
    >>> np.set_printoptions(suppress=True)
    >>> im
    array([[ 0.11111111, -0.66666667, -2.44444444],
           [-0.33333333,  0.88888889,  0.11111111],
           [ 0.22222222,  0.44444444, -0.33333333]])
    >>> ti
    array([[-0.11111111,  2.66666667,  5.44444444],
           [ 1.33333333,  4.11111111,  6.88888889],
           [ 2.77777778,  5.55555556,  8.33333333]])

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', 'floating', shape=(-1, -1))
        _numeric('mask', 'boolean', shape=img.shape, ignore_none=True)
        _generic('mode', 'string', value_in=('plane_flatten', 'line_flatten'))
        _numeric('degree', 'integer', range_='[1;inf)')
        _numeric('return_tilt', 'boolean')

    validate_input()

    if mode == 'line_flatten':
        tilt = _line_flatten_tilt(img, mask, degree)

    elif mode == 'plane_flatten':
        tilt = _plane_flatten_tilt(img, mask)

    if return_tilt:
        return (img - tilt, tilt)
    else:
        return img - tilt


def _line_flatten_tilt(img, mask, degree):
    """
    Estimate tilt using the line flatten method.

    Parameters
    ----------
    img : ndarray
        The image from which the tilt is estimated.
    mask : ndarray, or None
        If not None, a bool ndarray of the the shape as `img` indicating which
        pixels should be used in estimate of tilt.
    degree : int
        The degree of the polynomial in the estimated line tilt.

    Returns
    -------
    tilt : ndarray
        The estimated tilt.

    """

    m, n = img.shape
    x = np.arange(n)

    # Shapes of matrices used in the detilting:
    # vander.shape=(n, degree+1), coef.shape=(degree+1, len(m_masked))
    vander = np.fliplr(np.vander(x, degree + 1))  # [1, x, x**2, ...]

    if mask is not None:
        tilt = np.zeros_like(img)
        for l in range(m):
            if mask[l, :].sum() >= degree + 1:  # Skip if underdetermined
                coef, res, rank, s = np.linalg.lstsq(vander[mask[l, :]],
                                                     img[l, mask[l, :]])
                tilt[l, :] = vander.dot(coef).T

    else:
        coef, res, rank, s = np.linalg.lstsq(vander, img.T)
        tilt = vander.dot(coef).T

    return tilt


def _plane_flatten_tilt(img, mask):
    """
    Estimate tilt using the plane flatten method.

    Parameters
    ----------
    img : ndarray
        The image from which the tilt is estimated.
    mask : ndarray, or None
        If not None, a bool ndarray of the the shape as `img` indicating which
        pixels should be used in estimate of tilt.

    Returns
    -------
    tilt : ndarray
        The estimated tilt.

    """

    m, n = img.shape
    x = np.arange(n)
    y = np.arange(m)
    Y = np.tile(y, n)
    X = np.repeat(x, m)

    # ---------->  x-axis (second numpy axis)
    # |
    # |  image
    # |
    # v
    #
    # y-axis (first numpy axis)

    # Plane equation: by + ax + c = z
    Q = np.column_stack([Y, X, np.ones(X.shape[0])])  # [y, x, 1]
    img_as_vec = magni.imaging.mat2vec(img)  # image values corresponding to z

    if mask is not None:
        mask = magni.imaging.mat2vec(mask).ravel()
        Q_mask = Q[mask]
        img_as_vec_mask = img_as_vec[mask]
    else:
        Q_mask = Q
        img_as_vec_mask = img_as_vec

    # Least squares solve: [Y, X, 1] [b, a, c].T = img_as_vec
    coef, res, rank, s = np.linalg.lstsq(Q_mask, img_as_vec_mask)

    z = Q.dot(coef)
    tilt = magni.imaging.vec2mat(z.reshape(m * n, 1), (m, n))

    return tilt
