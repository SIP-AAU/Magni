"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the original SL0 reconstruction algorithm.

Routine listings
----------------
run(y, A)
    Run the original SL0 reconstruction algorithm.

See Also
--------
magni.cs.reconstruction.sl0.config : Configuration options.

Notes
-----
| The original SL0 reconstruction algorithm is described in [1]_ and [2]_.
|     For delta < 0.55: Standard projection algorithm by Mohimani et. al [1]_
|     For delta >= 0.55: Standard constraint elimination algorithm by Cui et.
      al. [2]_

References
----------
.. [1] H. Mohimani, M. Babaie-Zadeh, and C. Jutten, "A Fast Approach for
   Overcomplete Sparse Decomposition Based on Smoothed l0 Norm", *IEEE
   Transactions on Signal Processing*, vol. 57, no. 1, pp. 289-301, Jan. 2009.
.. [2] Z. Cui, H. Zhang, and W. Lu, "An Improved Smoothed l0-norm Algorithm
   Based on Multiparameter Approximation Function", *in 12th IEEE International
   Conference on Communication Technology (ICCT)*, Nanjing, China, Nov. 11-14,
   2011, pp. 942-945.

"""

from __future__ import division

import numpy as np
import scipy

from magni.cs.reconstruction.sl0 import config as _conf


def run(y, A):
    """
    Run the SL0 reconstruction algorithm.

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

    >>> import numpy as np
    >>> from magni.cs.reconstruction.sl0._original import run
    >>> np.random.seed(seed=6021)
    >>> A = 1 / np.sqrt(80) * np.random.randn(80, 200)
    >>> x = np.zeros((200, 1))
    >>> x[:10] = 1
    >>> y = A.dot(x)
    >>> x_hat = run(y, A)
    >>> np.set_printoptions(suppress=True)
    >>> x_hat[:12]
    array([[ 0.99984076],
           [ 0.99984986],
           [ 0.99995544],
           [ 0.99996633],
           [ 1.00010956],
           [ 1.00000432],
           [ 0.9999957 ],
           [ 1.00016335],
           [ 0.99992732],
           [ 0.99984163],
           [-0.00003011],
           [ 0.0000041 ]])
    >>> (np.abs(x_hat) > 1e-2).sum()
    10

    """

    if A.shape[0] / A.shape[1] < 0.55:
        x = _run_proj(y, A)
    else:
        x = _run_feas(y, A)

    return x


def _run_feas(y, A):
    """
    Run the original *feasibility* SL0 reconstruction algorithm.

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

    sigma_min = _conf['sigma_stop_fixed']
    L = int(np.round(_conf['L_fixed']))
    mu = _conf['mu_fixed']
    sigma_update = _conf['sigma_geometric']

    Q, R = scipy.linalg.qr(A.T, mode='economic')
    IP = np.eye(A.shape[1]) - Q.dot(Q.T)
    x = Q.dot(scipy.linalg.solve_triangular(R, y, trans='T'))
    sigma = (_conf['precision_float'](_conf['sigma_start_fixed']) *
             np.abs(x).max())

    while sigma > sigma_min:
        for j in range(L):
            d = np.exp(-x ** 2 / (2 * sigma ** 2)) * x
            x = x - mu * IP.dot(d)  # Search on feasible set

        sigma = sigma * sigma_update

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

    sigma_min = _conf['sigma_stop_fixed']
    L = int(np.round(_conf['L_fixed']))
    mu = _conf['mu_fixed']
    sigma_update = _conf['sigma_geometric']

    Q, R = scipy.linalg.qr(A.T, mode='economic')
    A_pinv = Q.dot(scipy.linalg.inv(R.T))
    x = A_pinv.dot(y)
    sigma = (_conf['precision_float'](_conf['sigma_start_fixed']) *
             np.abs(x).max())

    while sigma > sigma_min:
        for j in range(L):
            d = x * np.exp(-x ** 2 / (2 * sigma ** 2))
            x = x - mu * d
            x = x - A_pinv.dot(A.dot(x) - y)  # Projection

        sigma = sigma * sigma_update

    return x
