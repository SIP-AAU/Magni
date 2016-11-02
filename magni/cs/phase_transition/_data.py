"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing problem suite instance and noise generation functionality.

The problem suite instances consist of a matrix, A, and a coefficient vector,
alpha, with which the measurement vector, y, can be generated (with or without
noise from the noise vector e)

Routine listings
----------------
generate_matrix(m, n)
    Generate a matrix belonging to a specific problem suite.
generate_noise(m, n, k)
    Generate a noise vector of a specific type.
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

The choice of non-zero indices in the coefficient vector is controlled by the
configuration option 'support_structure' whereas the distribution of the
non-zero coefficients is controlled by the configuration option 'coefficient'.

Examples
--------
For example generate a sample from the USE/Rademacher problem suite:

>>> import numpy as np, magni
>>> from magni.cs.phase_transition import _data
>>> m, n, k = 400, 800, 100
>>> A = _data.generate_matrix(m, n)
>>> alpha = _data.generate_vector(n, k)
>>> y = np.dot(A, alpha)

Or generate a problem suite instance with "linear" support distribution.

>>> support_distrib = np.reshape(np.arange(n, dtype=np.float) + 1, (n, 1))
>>> support_distrib /= np.sum(support_distrib)
>>> magni.cs.phase_transition.config['support_distribution'] = support_distrib
>>> A = _data.generate_matrix(m, n)
>>> alpha = _data.generate_vector(n, k)
>>> y = np.dot(A, alpha)

Or generate an AWGN noise vector based on a 40 dB SNR

>>> magni.cs.phase_transition.config['noise'] = 'AWGN'
>>> e = _data.generate_noise(m, n, k)

"""

from __future__ import division

import numpy as np

import magni.imaging as _imaging
from magni.cs.phase_transition import config as _conf
from magni.utils.matrices import MatrixCollection as _MatrixCollection
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


def generate_matrix(m, n):
    """
    Generate a matrix belonging to a specific problem suite.

    The available matrices are

    * A random matrix drawn from the Uniform Spherical Ensemble (USE).
    * A fixed uniformly row sub-sampled DCT matrix ensemble (RandomDCT2D).
    * An option to use custom matrix factory (see notes below).

    See Notes for a description of these matrices. Which of the available
    matrices is used, is specified as a configuration option.

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

    See Also
    --------
    magni.cs.phase_transition.config : Configuration options.
    magni.utils.matrices.MatrixCollection : Fast transform implementation.

    Notes
    -----
    The Uniform Spherical Ensemble:
        The matrices of this ensemble have i.i.d. Gaussian entries of mean zero
        and variance one. Its columns are then normalised to have unit length.
    Fixed uniformly row sub-sampled DCT ensemble:
        The matrices of this ensemble correspond to the combination of a 2D
        array sub-sampled using a uniform point pattern and a 2D Discrete
        Cosine Transform (DCT) matrix. The combined matrix is implemented as a
        fast transform with a DCT based on an FFT routine.
    Custom matrix factory:
        The matrix generation is delegated to the configured
        `custom_system_matrix_factory` callable which is expected to take the
        arguments `m`, `n` and return `A`.

    """

    @_decorate_validation
    def validate_input():
        _numeric('n', 'integer', range_='[1;inf)')
        _numeric('m', 'integer', range_='[1;{}]'.format(n))

    @_decorate_validation
    def validate_output():
        _numeric('A', ('integer', 'floating', 'complex'), shape=(m, n))

    validate_input()

    system_matrix = _conf['system_matrix']

    if system_matrix == 'USE':
        A = np.float64(np.random.randn(m, n))
        A = A / np.linalg.norm(A, axis=0)

    elif system_matrix == 'RandomDCT2D':
        n_sqrt = np.sqrt(n)
        if not n_sqrt.is_integer():
            raise ValueError(
                'When using a RandomDCT2D system matrix, ' +
                'the value of >>{!r}<<, {!r} must be a square number'.format(
                    'n', n))

        n_sqrt = int(n_sqrt)

        # Random pixel subsampling operator
        points = np.random.choice(np.arange(n), size=m, replace=False)
        points.sort()
        coords = np.vstack([points // n_sqrt, points % n_sqrt]).T
        Phi = _imaging.measurements.construct_measurement_matrix(
            coords, n_sqrt, n_sqrt)
        assert Phi.shape[0] == m

        # DCT dictionary
        Psi = _imaging.dictionaries.get_DCT((n_sqrt, n_sqrt))

        A = _MatrixCollection((Phi, Psi))

    elif system_matrix == 'custom':
        A = _conf['custom_system_matrix_factory'](m, n)

    validate_output()

    return A


def generate_noise(m, n, k):
    """
    Generate a noise vector of a specific type.

    The available types are:

    * AWGN : Additive White Gaussian Noise
    * AWLN : Additive White Laplacian Noise
    * custom : The noise generation is delegated to the configured
      `custom_noise_factory` callable which is expected to take the arguments
      `m`, `n`, `k`, `noise_power`.

    Which of the available types is used, is specified as a configuration
    option.

    Parameters
    ----------
    m : int
        The number of rows.
    n : int
        The number of columns.
    k : int
        The number of non-zero coefficients.

    Returns
    -------
    e : ndarray
        The generated noise vector.

    See Also
    --------
    magni.cs.phase_transition.config : Configuration options.

    Notes
    -----
    The noise power is computed from the configured `SNR` and the theoretical
    ensemble variance of the coefficients generated by `generate_cofficients`.

    """

    @_decorate_validation
    def validate_input():
        _numeric('n', 'integer', range_='[1;inf)')
        _numeric('m', 'integer', range_='[1;{}]'.format(n))
        _numeric('k', 'integer', range_='[1;{}]'.format(n))

    @_decorate_validation
    def validate_output():
        _numeric('e', ('integer', 'floating', 'complex'), shape=(m, 1))

    validate_input()

    coefficients = _conf['coefficients']
    noise_type = _conf['noise']
    SNR = _conf['SNR']
    if noise_type is None:
        raise RuntimeError('The noise type has not been configured.')

    tau = k / n
    size = (m, 1)

    noise_generators = {
        'AWGN': lambda s: np.random.normal(scale=np.sqrt(s), size=size),
        'AWLN': lambda s: np.random.laplace(scale=np.sqrt(s / 2), size=size),
        'custom': lambda s: _conf['custom_noise_factory'](m, n, k, s)
    }

    # Generally the signal power is the coefficient variance scaled by the
    # signal density tau (number of non-zero coefficients).
    # This dictionary must be synced with the coeffient generators dictionary
    # in the generate_vector function below.
    signal_powers = {
        'rademacher': 1.0 * tau,
        'gaussian': 1.0 * tau,
        'laplace': 2.0 * tau,
        'bernoulli': 1.0 * tau
        }

    noise_power = 10**(-SNR/10) * signal_powers[coefficients]
    e = noise_generators[noise_type](noise_power)

    validate_output()

    return e


def generate_vector(n, k):
    """
    Generate a vector belonging to a specific problem suite.

    The available ensembles are:

    * Gaussian
    * Rademacher
    * Laplace
    * Bernoulli

    See Notes for a description of the ensembles. Which of the available
    ensembles is used, is specified as a configuration option. Note, that the
    non-zero `k` non-zero coefficients are the `k` first entries if no support
    structure specified in the configuration.

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
    The Gaussian ensemble:
        The non-zero coefficients are drawn from the normal Gaussian
        distribution.
    The Rademacher ensemble:
        The non-zero coefficients are drawn from the constant amplitude with
        random signs ensemble.
    The Laplace ensemble:
        The non-zero coefficients are drawn from the zero-mean, unit scale
        Laplace distribution (variance = 2).
    The Bernoulli ensemble:
        The non-zero coefficients are all equal to one.

    """

    @_decorate_validation
    def validate_input():
        _numeric('n', 'integer', range_='[1;inf)')
        _numeric('k', 'integer', range_='[1;{}]'.format(n))

    @_decorate_validation
    def validate_output():
        _numeric('alpha', ('integer', 'floating', 'complex'), shape=(n, 1))

    validate_input()

    alpha = np.zeros((n, 1))
    coefficients = _conf['coefficients']
    support_distribution = _conf['support_distribution']

    # This dictionary must be synced with the signal powers dictionary in the
    # generate_noise function above.
    coefficient_generators = {
        'rademacher': lambda k: np.random.randint(0, 2, k) * 2 - 1,
        'gaussian': lambda k: np.random.randn(k),
        'laplace': lambda k: np.random.laplace(size=k),  # Variance = 2
        'bernoulli': lambda k: np.ones(k)}

    if support_distribution is not None:
        ix = np.random.choice(np.arange(n), size=k, replace=False,
                              p=support_distribution.reshape(-1))
        alpha[ix, 0] = coefficient_generators[coefficients](k)

    else:
        alpha[:k, 0] = coefficient_generators[coefficients](k)

    validate_output()

    return alpha
