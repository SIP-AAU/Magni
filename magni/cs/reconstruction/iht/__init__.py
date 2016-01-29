"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing an implementation of Iterative Hard Thresholding (IHT).

.. note:: Deprecated in Magni 1.3.0.
          `magni.cs.reconstruction.iht` will be removed in a future version.
          Use the more general `magni.cs.reconstruction.it` instead.

.. warning:: Change of variable interpretation.
             In `magni.cs.reconstruction.it` the config variable
             `threshold_fixed` has a different interpretation than in
             `magni.cs.reconstruction.iht`.

Routine listings
----------------
config
    Configger providing configuration options for this subpackage.
run(y, A)
    Run the IHT reconstruction algorithm.

Notes
-----
The IHT reconstruction algorithm is described in [1]_. The default
configuration uses the False Alarm Rate heuristic described in [2]_.

References
----------
.. [1] T. Blumensath and M.E. Davies, "Iterative Thresholding for Sparse
   Approximations", *Journal of Fourier Analysis and Applications*, vol. 14,
   pp. 629-654, Sep. 2008.
.. [2] A. Maleki and D.L. Donoho, "Optimally Tuned Iterative Reconstruction
   Algorithms for Compressed Sensing", *IEEE Journal Selected Topics in Signal
   Processing*, vol. 3, no. 2, pp. 330-341, Apr. 2010.

"""

import warnings

from magni.cs.reconstruction.iht._config import configger as config
from magni.cs.reconstruction import it as _it


def run(y, A):
    """
    Run the IHT reconstruction algorithm.

    .. note:: Deprecated in Magni 1.3.0
          `magni.cs.reconstruction.iht` will be removed in a future version.
          Use the more general `magni.cs.reconstruction.it` instead.

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

    Examples
    --------
    For example, recovering a vector from random measurements

    >>> import warnings
    >>> import numpy as np
    >>> from magni.cs.reconstruction.iht import run, config
    >>> np.random.seed(seed=6021)
    >>> A = 1 / np.sqrt(80) * np.random.randn(80, 200)
    >>> alpha = np.zeros((200, 1))
    >>> alpha[:10] = 1
    >>> y = A.dot(alpha)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore')
    ...     alpha_hat = run(y, A)
    ...
    >>> np.set_printoptions(suppress=True)
    >>> alpha_hat[:12]
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
    >>> (np.abs(alpha_hat) > 1e-2).sum()
    10

    Or recovering the same only using a fixed threshold level:

    >>> config.update({'threshold': 'oracle', 'threshold_fixed': 10./80})
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore')
    ...     alpha_hat_2 = run(y, A)
    ...
    >>> alpha_hat_2[:12]
    array([[ 0.99877706],
           [ 0.99931441],
           [ 0.9978366 ],
           [ 0.99944973],
           [ 1.00052762],
           [ 1.00033436],
           [ 0.99943286],
           [ 0.99952526],
           [ 0.99941578],
           [ 0.99942908],
           [ 0.        ],
           [ 0.        ]])


    >>> (np.abs(alpha_hat_2) > 1e-2).sum()
    10

    """

    warnings.warn(
        '`magni.cs.reconstruction.iht` is deprecated in  magni 1.3.0. It ' +
        'will be removed in a future version. Use the more general ' +
        '`magni.cs.reconstruction.it` instead.', DeprecationWarning)

    current_it_config = dict(_it.config.items())

    iht_config = dict(config.items())
    k = int(iht_config['threshold_fixed'] * A.shape[0])
    iht_config['threshold_fixed'] = k
    if iht_config['threshold'] == 'oracle':
        iht_config['threshold'] = 'fixed'

    try:
        _it.config.update(iht_config)
        alpha = _it.run(y, A)
    finally:
        _it.config.update(current_it_config)

    return alpha
