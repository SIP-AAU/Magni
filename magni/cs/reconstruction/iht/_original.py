"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the actual reconstruction algorithm.

Routine listings
----------------
run(y, A)
    Run the IHT reconstruction algorithm.

See Also
--------
magni.cs.reconstruction.iht.config : Configuration options.

Notes
-----
The IHT reconstruction algorithm is described in [1]_.

References
----------
.. [1] A. Maleki and D.L. Donoho, "Optimally Tuned Iterative Reconstruction
   Algorithms for Compressed Sensing", *IEEE Journal Selected Topics in Signal
   Processing*, vol. 3, no. 2, pp. 330-341, Apr. 2010.

"""

from __future__ import division

import numpy as np
import scipy.stats

from magni.cs.reconstruction.iht import config as _config
from magni.utils.matrices import Matrix as _Matrix
from magni.utils.matrices import MatrixCollection as _MatrixC
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_ndarray as _validate_ndarray


@_decorate_validation
def _validate_run(y, A):
    """
    Validate the `run` function.

    See Also
    --------
    run : The validated function.
    magni.utils.validation.validate : Validation.

    """

    try:
        _validate_ndarray(A, 'A')
    except TypeError:
        if not isinstance(A, _Matrix) and not isinstance(A, _MatrixC):
            raise TypeError('A must be a matrix.')

    _validate_ndarray(y, 'y', {'shape': (A.shape[0], 1)})


def run(y, A):
    """
    Run the IHT reconstruction algorithm.

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
    _calculate_far : Optimal False Acceptance Rate calculation.
    _normalise : Matrix normalisation.

    Notes
    -----
    In each iteration, the threshold is robustly calculated as a fixed multiple
    of the standard deviation of the calculated correlations. The fixed
    multiple is based on the False Acceptance Rate (FAR) assuming a Gaussian
    distribution of the correlations.

    The algorithm terminates after a fixed number of iterations or if the ratio
    between the 2-norm of the residual and the 2-norm of the measurements falls
    below the specified `tolerance`.

    Examples
    --------
    For example, recovering a vector from random measurements

    >>> from magni.cs.reconstruction.iht._original import run
    >>> np.random.seed(seed=6021)
    >>> A = 1 / np.sqrt(80) * np.random.randn(80, 200)
    >>> x = np.zeros((200, 1))
    >>> x[:10] = 1
    >>> y = A.dot(x)
    >>> x_hat = run(y, A)
    >>> x_hat[:12]
    array([[ 0.99836297],
           [ 1.00029086],
           [ 0.99760224],
           [ 0.99927175],
           [ 0.99899124],
           [ 0.99899434],
           [ 0.9987368 ],
           [ 0.99801849],
           [ 1.00059408],
           [ 0.9983772 ],
           [ 0.        ],
           [ 0.        ]])
    >>> (np.abs(x_hat) > 1e-2).sum()
    10

    """

    _validate_run(y, A)

    _param = _config.get()
    convert = _param['precision_float']
    kappa = _param['kappa']
    tol = _param['tolerance']
    far = _calculate_far(A.shape[0] / A.shape[1])
    Lambda = convert(scipy.stats.norm.ppf(1 - far / 2))
    stdQ1 = convert(scipy.stats.norm.ppf(1 - 0.25))
    k = int(_param['threshold_rho'] * A.shape[0])

    x = np.zeros((A.shape[1], 1), dtype=convert)
    r = y.copy()

    for it in range(_param['iterations']):
        c = A.T.dot(r)
        x = x + kappa * c

        if _param['threshold'] == 'far':
            thres = (kappa * Lambda * convert(np.median(np.abs(c.ravel())))
                     / stdQ1)
        elif _param['threshold'] == 'oracle':
            if k == 0:
                thres = np.abs(x.ravel()).max() + 1
            elif k == A.shape[0]:
                thres = 0
            else:
                thres = np.sort(np.abs(x.ravel()))[-(k + 1)]

        x[np.abs(x) <= thres] = 0
        r = y - A.dot(x)

        if np.linalg.norm(r) < tol * np.linalg.norm(y):
            break

    return x


def _calculate_far(delta):
    """
    Calculate the optimal False Acceptance Rate for a given indeterminacy.

    Parameters
    ----------
    delta : float
        The indeterminacy, m / n, of a system of equations of size m x n.

    Returns
    -------
    FAR : float
        The optimal False Acceptance Rate for the given indeterminacy.

    Notes
    -----
    The optimal False Acceptance Rate to be used in connection with the
    interference heuristic presented in the paper "Optimally Tuned Iterative
    Reconstruction Algorithms for Compressed Sensing" [2]_ is calculated from
    a set of optimal values presented in the same paper. The calculated value
    is found from a linear interpolation or extrapolation on the known set of
    optimal values.

    References
    ----------
    .. [2] A. Maleki and D.L. Donoho, "Optimally Tuned Iterative Reconstruction
       Algorithms for Compressed Sensing", *IEEE Journal Selected Topics in
       Signal Processing*, vol. 3, no. 2, pp. 330-341, Apr. 2010.

    """

    # Known optimal values (x - indeterminacy / y - FAR)
    x = [0.05, 0.11, 0.21, 0.41, 0.50, 0.60, 0.70, 0.80, 0.93]
    y = [0.0015, 0.002, 0.004, 0.011, 0.015, 0.02, 0.027, 0.035, 0.043]

    for i in range(len(x) - 1):
        if delta <= x[i + 1]:
            break

    return y[i] + (delta - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
