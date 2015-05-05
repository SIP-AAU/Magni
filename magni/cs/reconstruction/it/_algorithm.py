"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the core Iterative Thresholding (IT) algorithm.

Routine listings
----------------
run(y, A)
    Run the IT reconstruction algorithm.

See Also
--------
magni.cs.reconstruction.it._config : Configuration options.

Notes
-----
The default configuration of the IT algorithm provides the Iterative Hard
Thresholding (IHT) algorithm [1]_ using the False Alarm Rate (FAR) heuristic
from [2]_. The algorithm may also be configured to act as the Iterative Soft
Thresholding (IST) [3]_ algorithm with the soft threshold from [4]_.

References
----------
.. [1] T. Blumensath and M.E. Davies, "Iterative Thresholding for Sparse
   Approximations", *Journal of Fourier Analysis and Applications*, vol. 14,
   pp. 629-654, Sep. 2008.
.. [2] A. Maleki and D.L. Donoho, "Optimally Tuned Iterative Reconstruction
   Algorithms for Compressed Sensing", *IEEE Journal Selected Topics in Signal
   Processing*, vol. 3, no. 2, pp. 330-341, Apr. 2010.
.. [3] I. Daubechies, M. Defrise, and C. D. Mol, "An Iterative Thresholding
   Algorithm for Linear Inverse Problems with a Sparsity Constraint",
   *Communications on Pure and Applied Mathematics*, vol. 57, no. 11,
   pp. 1413-1457, Nov. 2004.
.. [4] D.L. Donoho, "De-Noising by Soft-Thresholding", *IEEE Transactions on
   Information Theory*, vol. 41, no. 3, pp. 613-627, May. 1995.

"""

from __future__ import division

import numpy as np

from magni.cs.reconstruction.it import config as _conf
from magni.cs.reconstruction.it import _step_size
from magni.cs.reconstruction.it import _threshold
from magni.cs.reconstruction.it import _threshold_operators
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


def run(y, A):
    """
    Run the IT reconstruction algorithm.

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
    magni.cs.reconstruction.it._config : Configuration options.
    magni.cs.reconstruction.it._step_size : Step-size calculation.
    magni.cs.reconstruction.it._threshold : Threshold calculation.
    magni.cs.reconstruction.it._threshold_operators : Thresholding operators.

    Notes
    -----
    The algorithm terminates after a fixed number of iterations or if the ratio
    between the 2-norm of the residual and the 2-norm of the measurements falls
    below the specified `tolerance`.

    Examples
    --------
    For example, recovering a vector from random measurements using IHT with
    the FAR heuristic

    >>> import numpy as np, magni
    >>> from magni.cs.reconstruction.it import run
    >>> np.random.seed(seed=6021)
    >>> A = 1 / np.sqrt(90) * np.random.randn(90, 200)
    >>> alpha = np.zeros((200, 1))
    >>> alpha[:10] = 1
    >>> y = A.dot(alpha)
    >>> alpha_hat = run(y, A)
    >>> alpha_hat[:12]
    array([[ 0.99748945],
           [ 1.00082074],
           [ 0.99726507],
           [ 0.99987834],
           [ 1.00025857],
           [ 1.00003266],
           [ 1.00021666],
           [ 0.99838884],
           [ 1.00018356],
           [ 0.99859105],
           [ 0.        ],
           [ 0.        ]])
    >>> (np.abs(alpha_hat) > 1e-2).sum()
    10

    Or recover the same vector as above using IST with a fixed number of
    non-thresholded coefficients

    >>> it_config = {'threshold_operator': 'soft', 'threshold': 'fixed',
    ... 'threshold_fixed': 10}
    >>> magni.cs.reconstruction.it.config.update(it_config)
    >>> alpha_hat = run(y, A)
    >>> alpha_hat[:12]
    array([[ 0.99822443],
           [ 0.99888724],
           [ 0.9982493 ],
           [ 0.99928642],
           [ 0.99964131],
           [ 0.99947346],
           [ 0.9992809 ],
           [ 0.99872624],
           [ 0.99916842],
           [ 0.99903033],
           [ 0.        ],
           [ 0.        ]])
    >>> (np.abs(alpha_hat) > 1e-2).sum()
    10

    Or use a weighted IHT method with a fixed number of non-thresholded
    coefficietns to recover the above signal

    >>> weights = np.linspace(1.0, 0.9, 200).reshape(200, 1)
    >>> it_config = {'threshold_operator': 'weighted_hard',
    ... 'threshold_weights': weights}
    >>> magni.cs.reconstruction.it.config.update(it_config)
    >>> alpha_hat = run(y, A)
    >>> alpha_hat[:12]
    array([[ 0.99853888],
           [ 0.99886104],
           [ 0.99843149],
           [ 1.0000333 ],
           [ 1.00060273],
           [ 1.00035539],
           [ 0.99966219],
           [ 0.99912209],
           [ 0.99961541],
           [ 0.99952029],
           [ 0.        ],
           [ 0.        ]])
    >>> (np.abs(alpha_hat) > 1e-2).sum()
    10
    >>> magni.cs.reconstruction.it.config.reset()

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

    # Configured setup
    param = dict(_conf.items())
    convert = param['precision_float']
    iterations = param['iterations']
    tolerance = param['tolerance']
    threshold_weights = param['threshold_weights']
    kappa = param['kappa_fixed']
    threshold_alpha = _threshold_operators.get_function_handle(
        param['threshold_operator'])

    # Initialisation based on configuration
    if param['warm_start'] is not None:
        alpha = param['warm_start']
    else:
        alpha = np.zeros((A.shape[1], 1), dtype=convert)

    run.calculate_step_size = _step_size.get_function_handle(
        param['kappa'], locals())
    run.calculate_threshold = _threshold.get_function_handle(
        param['threshold'], locals())

    r = y.copy()

    # IT iterations
    for it in range(param['iterations']):
        c = A.T.dot(r)

        # Step-size (relaxation parameter)
        kappa = run.calculate_step_size(locals())
        alpha = alpha + kappa * c

        # Threshold
        threshold = run.calculate_threshold(locals())
        threshold_alpha(locals())

        # Residual
        r = y - A.dot(alpha)

        if np.linalg.norm(r) < tolerance * np.linalg.norm(y):
            break

    validate_output()

    return alpha
