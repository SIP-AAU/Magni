"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing problem suite instance generation functionality.

The problem suite instances consist of a matrix, A, and a coefficient vector,
alpha, with which the measurement vector, y, can be generated.

Routine listings
----------------
generate_matrix(m, n)
    Generate a matrix belonging to a specific problem suite.
generate_vector(n, k)
    Generate a vector belonging to a specific problem suite.

See also
--------
magni.cs.phase_transition._config: Configuration options.

Notes
-----
The matrices and vectors generated in this module use the numpy.random
submodule. Consequently, the calling script or function should control the seed
to ensure reproducibility.

Examples
--------
Generate a problem suite instance:

>>> import numpy as np
>>> from magni.cs.phase_transition import _data
>>> m, n, k = 400, 800, 100
>>> A = _data.generate_matrix(m, n)
>>> alpha = _data.generate_vector(n, k)
>>> y = np.dot(A, alpha)

"""

from __future__ import division

import numpy as np

from magni.cs.phase_transition import config as _conf


def generate_matrix(m, n):
    """
    Generate a matrix belonging to a specific problem suite.

    The available ensemble is the Uniform Spherical Ensemble. See Notes for a
    description of the ensemble.

    Parameters
    ----------
    m : int
        The number of rows.
    n : int
        The number of columns.

    Returns
    -------
    A : ndarray
        The generated matrix.

    Notes
    -----
    The Uniform Spherical Ensemble:
        The matrices of this ensemble have i.i.d. Gaussian entries and its
        columns are normalised to have unit length.

    """

    A = np.float64(np.random.randn(m, n))

    for i in range(n):
        A[:, i] = A[:, i] / np.linalg.norm(A[:, i])

    return A


def generate_vector(n, k):
    """
    Generate a vector belonging to a specific problem suite.

    The available ensembles are the Gaussian ensemble and the Rademacher
    ensemble. See Notes for a description of the ensembles. Which of the
    available ensembles is used, is specified as a configuration option. Note,
    that the non-zero `k` non-zero coefficients are the `k` first entries.

    Parameters
    ----------
    n : int
        The length of the vector.
    k : int
        The number of non-zero coefficients.

    Returns
    -------
    alpha : ndarray
        The generated vector.

    See Also
    --------
    magni.cs.phase_transition.config : Configuration options.

    Notes
    -----
    The Gaussian ensemble :
        The non-zero coefficients are drawn from the normal Gaussian
        distribution.
    The Rademacher ensemble:
        The non-zero coefficients are drawn from the constant amplitude with
        random signs ensemble.

    """

    alpha = np.zeros((n, 1))
    coefficients = _conf['coefficients']

    if coefficients == 'rademacher':
        alpha[:k, 0] = np.random.randint(0, 2, k) * 2 - 1
    elif coefficients == 'gaussian':
        alpha[:k, 0] = np.random.randn(k)

    return alpha
