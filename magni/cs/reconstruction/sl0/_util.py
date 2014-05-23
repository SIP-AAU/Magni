"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the public function of the magni.cs.reconstruction.sl0
subpackage.

"""

from __future__ import division

from magni.cs.reconstruction.sl0 import config as _config
from magni.cs.reconstruction.sl0 import _modified
from magni.cs.reconstruction.sl0 import _original
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

    _validate_ndarray(A, 'A', {'dim': 2})
    _validate_ndarray(y, 'y', {'shape': (A.shape[0], 1)})


def run(y, A):
    """
    Run the specified SL0 reconstruction algorithm.

    The available SL0 reconstruction algorithms are the original SL0 and the
    modified SL0. Which of the available SL0 reconstruction algorithms is used,
    is specified as a configuration option.

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

    _validate_run(y, A)

    algorithm = _config.get('algorithm')

    if algorithm == 'std':
        x = _original.run(y, A)
    elif algorithm == 'mod':
        x = _modified.run(y, A)

    return x
