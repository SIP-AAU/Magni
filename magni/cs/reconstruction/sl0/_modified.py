"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the modified SL0 reconstruction algorithm.

Routine listings
----------------
run(y, A)
    Run the modified SL0 reconstruction algorithm.

See Also
--------
magni.cs.reconstruction.sl0.config : Configuration options.

Notes
-----
| The modified SL0 reconstruction algorithm is described in [1]_.
|     For delta < 0.55: Modified projection algorithm
|     For delta >= 0.55: Modified constraint elimination algorithm

References
----------
.. [1] C. S. Oxvig, P. S. Pedersen, T. Arildsen, and T. Larsen, "Surpassing the
   Theoretical 1-norm Phase Transition in Compressive Sensing by Tuning the
   Smoothed l0 Algorithm", *in IEEE International Conference on Acoustics,
   Speech and Signal Processing (ICASSP)*, Vancouver, Canada, May 26-31, 2013,
   pp. 6019-6023.

"""

from __future__ import division

import numpy as np
import scipy

from magni.cs.reconstruction.sl0 import config as _config


def run(y, A):
    """
    Run the modified SL0 reconstruction algorithm.

    Parameters
    ----------
    y : ndarray
        The m x 1 measurement vector.
    A : ndarray
        The m x n matrix which is the product of the measurement matrix and the
        dictionary matrix.

    Returns
    -------
    alpha : ndarray
        The n x 1 reconstructed coefficient vector.

    See Also
    --------
    _run_proj : The original projection algorithm.
    _run_feas : The original constraint elimination algorithm.

    Examples
    --------
    For example, recovering a vector from random measurements

    >>> from magni.cs.reconstruction.sl0._modified import run
    >>> np.random.seed(seed=6021)
    >>> A = 1 / np.sqrt(80) * np.random.randn(80, 200)
    >>> x = np.zeros((200, 1))
    >>> x[:10] = 1
    >>> y = A.dot(x)
    >>> x_hat = run(y, A)
    >>> x_hat[:12]
    array([[  9.99997941e-01],
           [  9.99999463e-01],
           [  1.00000090e+00],
           [  9.99998622e-01],
           [  1.00000078e+00],
           [  9.99998433e-01],
           [  1.00000025e+00],
           [  1.00000346e+00],
           [  1.00000088e+00],
           [  9.99995474e-01],
           [ -4.12075673e-07],
           [  2.17244596e-07]])
    >>> (np.abs(x_hat) > 1e-2).sum()
    10

    """

    if A.shape[0] / A.shape[1] < 0.55:
        x = _run_proj(y, A)
    else:
        x = _run_feas(y, A)

    return x


def _calc_sigma_start(delta):
    """
    Calculate the initial sigma factor for a given indeterminacy.

    Parameters
    ----------
    delta : float
        The indeterminacy, m / n, of a system of equations of size m x n.

    Returns
    -------
    sigma_start : float
        The initial sigma factor for the given indeterminacy.

    """

    return 1.0 / (2.75 * delta)


def _run_feas(y, A):
    """
    Run the modified *feasibility* SL0 reconstruction algorithm.

    This function implements the algorithm with a search on the feasible set.

    Parameters
    ----------
    y : ndarray
        The m x 1 measurement vector.
    A : ndarray
        The m x n matrix which is the product of the measurement matrix and the
        dictionary matrix.

    Returns
    -------
    alpha : ndarray
        The n x 1 reconstructed coefficient vector.

    """

    param = _config.get()

    sigma_update = param['sigma_update']
    sigma_min = param['sigma_min']

    L = param['L']
    L_update = param['L_update']

    mu_start = param['mu_start']
    mu_end = param['mu_end']

    epsilon = param['epsilon']

    Q, R = scipy.linalg.qr(A.T)
    Q1 = Q[:, :A.shape[0]]
    Q2 = Q[:, A.shape[0]:]
    R = R[:R.shape[1], :]
    x = Q1.dot(scipy.linalg.solve_triangular(R, y, trans='T'))

    mult = _calc_sigma_start(A.shape[0] / A.shape[1])
    sigma = mult * np.abs(x).max()

    x_zeros = np.zeros(x.shape)
    i = 0

    while sigma > sigma_min:
        if i < 4 or mult * sigma_update**i > 0.75:
            if A.shape[0] / A.shape[1] <= 0.5:
                mu = 50 * mu_start
            else:
                mu = mu_start
        else:
            mu = mu_end

        x_prev = x_zeros
        j = 0

        while scipy.linalg.norm(x - x_prev) > sigma * epsilon and j <= L:
            x_prev = x.copy()

            d = np.exp(-(x ** 2) / (2 * sigma ** 2)) * x
            nabla = Q2.T.dot(d)
            x = x - Q2.dot(mu * nabla)  # Search on feasible set

            j = j + 1

        sigma = sigma * sigma_update
        L = L * L_update
        i = i + 1

    return x


def _run_proj(y, A):
    """
    Run the original *projection* SL0 reconstruction algorithm.

    This function implements the algorithm with an unconstrained gradient step
    followed by a projection back onto the feasible set.

    Parameters
    ----------
    y : ndarray
        The m x 1 measurement vector.
    A : ndarray
        The m x n matrix which is the product of the measurement matrix and the
        dictionary matrix.

    Returns
    -------
    alpha : ndarray
        The n x 1 reconstructed coefficient vector.

    """

    param = _config.get()

    sigma_update = param['sigma_update']
    sigma_min = param['sigma_min']

    L = param['L']
    L_update = param['L_update']

    mu_start = param['mu_start']
    mu_end = param['mu_end']

    epsilon = param['epsilon']

    Q, R = scipy.linalg.qr(A.T, mode='economic')
    A_pinv = Q.dot(scipy.linalg.inv(R.T))
    x = A_pinv.dot(y)

    mult = _calc_sigma_start(A.shape[0] / A.shape[1])
    sigma = mult * np.abs(x).max()

    x_zeros = np.zeros(x.shape)
    i = 0

    while sigma > sigma_min:
        if i < 4 or mult * sigma_update**i > 0.75:
            if A.shape[0] / A.shape[1] <= 0.5:
                mu = 50 * mu_start
            else:
                mu = mu_start
        else:
            mu = mu_end

        x_prev = x_zeros
        j = 0

        while scipy.linalg.norm(x - x_prev) > sigma * epsilon and j <= L:
            x_prev = x.copy()

            d = np.exp(-(x ** 2) / (2 * sigma ** 2)) * x
            x = x - mu * d
            x = x - A_pinv.dot(A.dot(x) - y)  # Projection

            j = j + 1

        sigma = sigma * sigma_update
        L = L * L_update
        i = i + 1

    return x
