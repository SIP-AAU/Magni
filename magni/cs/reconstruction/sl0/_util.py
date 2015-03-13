"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the public function of the magni.cs.reconstruction.sl0
subpackage.

"""

from __future__ import division

import numpy as np

from magni.cs.reconstruction.sl0 import config as _conf
from magni.cs.reconstruction.sl0 import _modified
from magni.cs.reconstruction.sl0 import _original
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


def run(y, A):
    """
    Run the specified SL0 reconstruction algorithm.

    The available SL0 reconstruction algorithms are the original SL0 and the
    modified SL0. Which of the available SL0 reconstruction algorithms is used,
    is specified as configuration options.

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
    magni.cs.reconstruction.sl0.config : Configuration options.
    magni.cs.reconstruction.sl0._original.run : The original SL0 reconstruction
        algorithm.
    magni.cs.reconstruction.sl0._modified.run : The modified SL0 reconstruction
        algorithm.

    Examples
    --------
    See the individual run functions in the implementations of the original and
    modified SL0 reconstruction algorithms.

    """

    @_decorate_validation
    def validate_input():
        _numeric('y', ('integer', 'floating', 'complex'), shape=(-1, 1))
        _numeric('A', ('integer', 'floating', 'complex'),
                 shape=(y.shape[0], -1))

    validate_input()

    if not isinstance(A, np.ndarray):
        A = A.A

    if _conf['L'] == _conf['mu'] == _conf['sigma_start'] == 'fixed':
        x = _original.run(y, A)
    elif (_conf['L'] == 'geometric' and _conf['mu'] == 'step' and
          _conf['sigma_start'] == 'reciprocal'):
        x = _modified.run(y, A)
    else:
        raise NotImplementedError(
            "Currently, only the following configuration combinations of (L, "
            "mu, sigma_start) are implemented: ('fixed', 'fixed', 'fixed') "
            "and ('geometric', 'step', 'reciprocal').")

    return x
