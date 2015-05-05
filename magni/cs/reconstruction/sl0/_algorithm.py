"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the core Smoothed l0 (SL0) algorithm.

Routine listings
----------------
run(y, A)
    Run the SL0 reconstruction algoritm.

See Also
--------
magni.cs.reconstruction.sl0._config : Configuration options.

Notes
-----
Implementations of the original SL0 reconstruction algorithm [1]_ and a
modified Sl0 reconstruction algorithm [3]_ are available. It is also possible
to configure the subpackage to provide customised versions of the SL0
reconstruction algorithm. The projection algorithm [1]_ is used for small delta
(< 0.55) whereas the contraint elimination algorithm [2]_ is used for large
delta (>= 0.55) which merely affects the computation time.

References
----------
.. [1] H. Mohimani, M. Babaie-Zadeh, and C. Jutten, "A Fast Approach for
   Overcomplete Sparse Decomposition Based on Smoothed l0 Norm", *IEEE
   Transactions on Signal Processing*, vol. 57, no. 1, pp. 289-301, Jan. 2009.
.. [2] Z. Cui, H. Zhang, and W. Lu, "An Improved Smoothed l0-norm Algorithm
   Based on Multiparameter Approximation Function", *in 12th IEEE International
   Conference on Communication Technology (ICCT)*, Nanjing, China, Nov. 11-14,
   2011, pp. 942-945.
.. [3] C. S. Oxvig, P. S. Pedersen, T. Arildsen, and T. Larsen, "Surpassing the
   Theoretical 1-norm Phase Transition in Compressive Sensing by Tuning the
   Smoothed l0 Algorithm", *in IEEE International Conference on Acoustics,
   Speech and Signal Processing (ICASSP)*, Vancouver, Canada, May 26-31, 2013,
   pp. 6019-6023.

"""

from __future__ import division

import numpy as np
import scipy.linalg

from magni.cs.reconstruction.sl0 import config as _conf
from magni.cs.reconstruction.sl0 import _L_start
from magni.cs.reconstruction.sl0 import _L_update
from magni.cs.reconstruction.sl0 import _mu_start
from magni.cs.reconstruction.sl0 import _mu_update
from magni.cs.reconstruction.sl0 import _sigma_start
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


def run(y, A):
    """
    Run the SL0 reconstruction algorithm.

    Parameters
    ----------
    y : ndarray
        The m x 1 measurement vector.
    A : ndarray or magni.utils.matrices.{Matrix, MatrixCollection}
        The m x n matrix which is the product of the measurement matrix and the
        dictionary matrix.

    Returns
    -------
    alpha : ndarray
        The n x 1 reconstructed coefficient vector.

    See Also
    --------
    magni.cs.reconstruction.sl0.config : Configuration options.

    Notes
    -----
    The algorithm terminates after a fixed number of iterations or if the ratio
    between the 2-norm of the residual and the 2-norm of the measurements falls
    below the specified `tolerance`.

    Examples
    --------
    For example, recovering a vector from random measurements using the
    original SL0 reconstruction algorithm

    >>> import numpy as np, magni
    >>> from magni.cs.reconstruction.sl0 import run
    >>> np.set_printoptions(suppress=True)
    >>> magni.cs.reconstruction.sl0.config['L'] = 'fixed'
    >>> magni.cs.reconstruction.sl0.config['mu'] = 'fixed'
    >>> magni.cs.reconstruction.sl0.config['sigma_start'] = 'fixed'
    >>> np.random.seed(seed=6021)
    >>> A = 1 / np.sqrt(80) * np.random.randn(80, 200)
    >>> alpha = np.zeros((200, 1))
    >>> alpha[:10] = 1
    >>> y = A.dot(alpha)
    >>> alpha_hat = run(y, A)
    >>> alpha_hat[:12]
    array([[ 0.99993202],
           [ 0.99992793],
           [ 0.99998107],
           [ 0.99998105],
           [ 1.00005882],
           [ 1.00000843],
           [ 0.99999138],
           [ 1.00009479],
           [ 0.99995889],
           [ 0.99992509],
           [-0.00001509],
           [ 0.00000275]])
    >>> (np.abs(alpha_hat) > 1e-2).sum()
    10

    Or recover the same vector as above using the modified SL0 reconstruction
    algorithm

    >>> magni.cs.reconstruction.sl0.config['L'] = 'geometric'
    >>> magni.cs.reconstruction.sl0.config['mu'] = 'step'
    >>> magni.cs.reconstruction.sl0.config['sigma_start'] = 'reciprocal'
    >>> alpha_hat = run(y, A)
    >>> alpha_hat[:12]
    array([[ 0.9999963 ],
           [ 1.00000119],
           [ 1.00000293],
           [ 0.99999661],
           [ 1.00000021],
           [ 0.9999951 ],
           [ 1.00000103],
           [ 1.00000662],
           [ 1.00000404],
           [ 0.99998937],
           [-0.00000075],
           [ 0.00000037]])
    >>> (np.abs(alpha_hat) > 1e-2).sum()
    10

    """

    @_decorate_validation
    def validate_input():
        _numeric('y', ('integer', 'floating', 'complex'), shape=(-1, 1))
        _numeric('A', ('integer', 'floating', 'complex'),
                 shape=(y.shape[0], -1))

    @_decorate_validation
    def validate_output():
        _numeric('alpha', ('integer', 'floating', 'complex'),
                 shape=(A.shape[1], 1))

    validate_input()

    if not isinstance(A, np.ndarray):
        A = A.A

    param = dict(_conf.items())
    convert = param['precision_float']
    epsilon = param['epsilon']
    sigma_geometric = param['sigma_geometric']
    sigma_stop = param['sigma_stop_fixed']

    L = _L_start.get_function_handle(param['L'], locals())()
    mu = _mu_start.get_function_handle(param['mu'], locals())()
    sigma_start = _sigma_start.get_function_handle(
        param['sigma_start'], locals())()

    calculate_L_update = _L_update.get_function_handle(param['L'], locals())
    calculate_mu_update = _mu_update.get_function_handle(param['mu'], locals())

    if A.shape[0] / A.shape[1] < 0.55:
        Q, R = scipy.linalg.qr(A.T, mode='economic')
        A_pinv = Q.dot(scipy.linalg.inv(R.T))
        alpha = A_pinv.dot(y)
    else:
        Q, R = scipy.linalg.qr(A.T)
        Q1 = Q[:, :A.shape[0]]
        Q2 = Q[:, A.shape[0]:]
        R = R[:R.shape[1], :]
        alpha = Q1.dot(scipy.linalg.solve_triangular(R, y, trans='T'))

    sigma = convert(sigma_start) * np.abs(alpha).max()

    alpha_zeros = np.zeros(alpha.shape)
    i = 0

    while sigma > sigma_stop:
        alpha_prev = alpha_zeros
        j = 0

        while (scipy.linalg.norm(alpha - alpha_prev) > sigma * epsilon and
               j <= L):
            alpha_prev = alpha.copy()
            d = np.exp(-alpha**2 / (2 * sigma**2)) * alpha

            if A.shape[0] / A.shape[1] < 0.55:
                alpha = alpha - mu * d
                alpha = alpha - A_pinv.dot(A.dot(alpha) - y)  # Projection
            else:
                nabla = Q2.T.dot(d)
                alpha = alpha - Q2.dot(mu * nabla)  # Search on feasible set

            j = j + 1

        sigma = sigma * sigma_geometric
        L = calculate_L_update(locals())
        mu = calculate_mu_update(locals())
        i = i + 1

    validate_output()

    return alpha
