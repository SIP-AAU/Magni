"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing utility functions for initilisation of in- and output channels
in the Generalised Approximate Message Passing (GAMP) algorithm.

Routine listings
----------------
get_em_bg_amp_initialisation(problem_params, method='vila')
    Get initial parameters for EM Bernoulli-Guassian AMP.
rho_se(delta, zeta, resolution=1000)
    Return the theoretical noiseless LASSO phase transition.

"""

from __future__ import division

import numpy as np

from scipy.stats import norm as _gaus

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.matrices import norm as _norm


def get_em_bg_amp_initialisation(problem_params, method='vila'):
    """
    Get initial parameters for EM Bernoulli-Guassian AMP.

    If the initialisation `method` is `vila` then the scheme from [1]_ is used.
    If it is `krzakala` then the scheme from [2]_ is used.

    Parameters
    ----------
    problem_params : dict
        The problem parameters used to compute the initialisation.
    method : str
        The initialisation method to use.

    Returns
    -------
    tau : float
        The initial sparsity level.
    sigma_sq : float
        The initial AWGN noise level.
    theta_tilde : float
        The initial Gaussian prior variance.

    See Also
    --------
    magni.cs.reconstruction.gamp._output_channel.wrap_calculate_using_AWGN :\
        Related output channel.
    magni.cs.reconstruction.gamp._input_channels.wrap_calculate_using_iidsGB :\
        Related input channel.

    Notes
    -----
    Independently of the choice of `method`, the `problem_params` are:

    * y: the measurements
    * A: the system matrix

    If `method` is `vila`, one must also specify:

    * SNR: the signal-to-noise ratio

    References
    ----------
    .. [1] J. P. Vila and P. Schniter, "Expectation-Maximization
       Gaussian-Mixture Approximate Message Passing", *IEEE Transactions on
       Signal Processing*, 2013, vol. 61, no. 19, pp. 4658-4672, Oct. 2013.
    .. [2] F. Krzakala, M. Mezard, F. Sausset, Y. Sun, and L. Zdeborova,
       "Probabilistic reconstruction in compressed sensing: algorithms, phase
       diagrams, and threshold achieving matrices", *Journal of Statistical
       Mechanics: Theory and Experiment*, vol. P08009, pp. 1-57, Aug. 2012.

    Examples
    --------
    For example, get the "vila" initialisation for a SNR of 100

    >>> import numpy as np
    >>> from magni.cs.reconstruction.gamp.channel_initialisation import \
    ... get_em_bg_amp_initialisation as get_init
    >>> np.random.seed(6012)
    >>> A = np.random.randn(20, 40)
    >>> y = np.random.randn(20, 1)
    >>> problem_params = {'A': A, 'y': y, 'SNR': 100}
    >>> init_tup = get_init(problem_params, method='vila')
    >>> [round(float(elem), 3) for elem in init_tup]
    [0.193, 0.01, 0.123]

    or get the corresponding "krzakala" initialisation

    >>> del problem_params['SNR']
    >>> init_tup = get_init(problem_params, method='krzakala')
    >>> [round(float(elem), 3) for elem in init_tup]
    [0.05, 1.0, 0.479]

    """

    @_decorate_validation
    def validate_input():
        _generic('method', 'string', value_in=('vila', 'krzakala'))
        param_keys = {'y', 'A'}
        if method == 'vila':
            param_keys.add('SNR')
        _generic('problem_params', 'mapping', keys_in=tuple(param_keys))

    validate_input()

    y = problem_params['y']
    A = problem_params['A']
    m = A.shape[0]
    n = A.shape[1]
    delta = m / n

    norm_y_sq = np.linalg.norm(y)**2
    norm_A_sq = _norm(A, 'fro')**2

    if method == 'vila':
        SNR = problem_params['SNR']
        tau = delta * rho_se(delta, 2)
        sigma_sq = norm_y_sq / ((SNR + 1) * m)
        theta_tilde = (norm_y_sq - m * sigma_sq) / (norm_A_sq * tau)

    elif method == 'krzakala':
        tau = delta / 10
        sigma_sq = 1.0  # Our best guess. Does not seem to be documented in [2]
        theta_tilde = norm_y_sq / (norm_A_sq * tau)

    return tau, sigma_sq, theta_tilde


def rho_se(delta, zeta, resolution=1000):
    """
    Return the theoretical noiseless LASSO phase transition.

    Parameters
    ----------
    delta : float
        The under sampling ratio.
    zeta : {1, 2}
        The "problem" to get the phase transition for.
    resolution : int
        The resolution used in the brute force optimisation.

    Returns
    -------
    rho_se : float
        The phase transition value.

    Notes
    -----
    The theoretical noiseless LASSO phase transition is computed based on eq. 5
    in [3]_. A simple brute force optimisation with the specified `resolution`
    of that expression is used to find the phase transition. The "problems",
    for which the phase transition may be computed, are:

    1. Sparse nonnegative vectors
    2. Sparse signed vectors

    References
    ----------
    .. [3] D.L. Donoho, A. Maleki, and A. Montanari, "Message-passing
       algorithms for compressed sensing", *Proceedings of the National Academy
       of Sciences of the United States of America*, vol. 106, no. 45,
       pp. 18914-18919, Nov. 2009.

    Examples
    --------
    For example, find a phase transition value for a sparse signed vector:

    >>> from magni.cs.reconstruction.gamp.channel_initialisation import rho_se
    >>> round(float(rho_se(0.19, 2)), 3)
    0.238

    or find the corresponding value for a sparse nonnegative vector

    >>> round(float(rho_se(0.19, 1)), 3)
    0.318

    and a few more examples

    >>> round(float(rho_se(0.0, 1)), 3)
    0.0
    >>> round(float(rho_se(0.0, 2)), 3)
    0.0
    >>> round(float(rho_se(0.5, 1)), 3)
    0.558
    >>> round(float(rho_se(0.5, 2)), 3)
    0.386
    >>> round(float(rho_se(1.0, 1)), 3)
    0.95
    >>> round(float(rho_se(1.0, 2)), 3)
    0.95

    """

    @_decorate_validation
    def validate_input():
        _numeric('delta', ('integer', 'floating'), range_='[0;1]')
        _numeric('zeta', ('integer'), range_='[1;2]')
        _numeric('resolution', ('integer'), range_='[1;inf)')

    validate_input()

    gaus = _gaus()

    if delta < 1e-12:
        rho_se_z = 0.0
    elif delta > 0.99:
        rho_se_z = 0.95
    else:
        z = np.linspace(1e-12, 10, resolution)
        gaus_elem = (1 + z**2) * gaus.cdf(-z) - z * gaus.pdf(-z)
        rho_se_z = ((1 - zeta / delta * gaus_elem) /
                    (1 + z**2 - zeta * gaus_elem))

    return np.nanmax(rho_se_z)
