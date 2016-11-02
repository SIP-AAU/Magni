"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the phase_transition subpackage.

See also
--------
magni.utils.config.Configger : The Configger class used

Notes
-----
This module instantiates the `Configger` class provided by
`magni.utils.config`. The configuration options are the following:

algorithm_kwargs : dict
    The keyword arguments passed on to the reconstruction algorithm.
coefficients : {'rademacher', 'gaussian', 'laplace', 'bernoulli'}
    The distribution which the non-zero coefficients in the coefficient vector
    are drawn from.
custom_noise_factory : callable
    The callable that generates a custom m-by-1 vector of noise from the four
    arguments "m", "n", "k", "noise_power".
custom_system_matrix_factory : callable
    The callable that generates a custom m-by-n system matrix from the two
    arguments "m", "n".
delta : list or tuple
    The delta values of the delta-rho grid whose points are used for the monte
    carlo simulations (the default is [0., 1.]).
logit_solver : {'built-in', 'sklearn'}
    The solver to use in the logistic regression fit of the phase transition.
maxpoints : int
    The maximum number of phase space grid points to be handled by a process
    before it is replaced by a new process to free ressources (the default is
    None, which implies that processes are not replaced).
monte_carlo : int
    The number of monte carlo simulations to run in each point of the delta-rho
    grid (the default is 1).
noise : {'AWGN', 'AWLN', 'custom'}
    The type of noise to use (the default is None, which implies that
    noiseless measurements are used).
problem_size : int
    The length of the coefficient vector (the default is 800).
rho : list or tuple
    The rho values of the delta-rho grid whose points are used for the monte
    carlo simulations (the default is [0., 1.]).
seed : int
    The seed used when picking seeds for generating data for the monte carlo
    simulations or using the scikit-learn logistic regression solver (the
    default is None, which implies an arbitrary seed).
SNR : int or float
    The signal-to-noise ratio in decibel to use (the default is 40 dB).
support_distribution : ndarray
    The n x 1 support distribution array to use in the generation of test
    vectors (the default is None, which implies that the active entries in test
    vectors are the low index entries).
system_matrix : {'USE', 'RandomDCT2D', 'custom'}
    The system matrix to use in the simulation (the default is USE, which
    implies that the system matrix is drawn from the Uniform Spherical Ensemble
    ).

"""

from __future__ import division
from types import FunctionType as _FunctionType
try:
    from collections import Callable as _Callable  # Python 2
except ImportError:
    from collections.abc import Callable as _Callable  # Python 3

from magni.utils.config import Configger as _Configger
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


configger = _Configger(
    {'algorithm_kwargs': {},
     'coefficients': 'rademacher',
     'custom_noise_factory': (
         lambda noise_power, m: 'Remember to configure custom noise factory'),
     'custom_system_matrix_factory': (
         lambda m, n: 'Remember to configure custom system matrix factory'),
     'delta': [0.0, 1.0],
     'logit_solver': 'built-in',
     'maxpoints': None,
     'monte_carlo': 1,
     'noise': None,
     'problem_size': 800,
     'rho': [0.0, 1.0],
     'seed': None,
     'SNR': 40,
     'support_distribution': None,
     'system_matrix': 'USE'},
    {'algorithm_kwargs': _generic(None, 'mapping'),
     'coefficients': _generic(
         None, 'string',
         value_in=('rademacher', 'gaussian', 'laplace', 'bernoulli')),
     'custom_noise_factory': _generic(
         None, (_FunctionType, _Callable)),
     'custom_system_matrix_factory': _generic(
         None, (_FunctionType, _Callable)),
     'delta': _levels(None, (
         _generic(None, 'explicit collection'),
         _numeric(None, 'floating', range_='[0;1]'))),
     'logit_solver': _generic(None, 'string',
                              value_in=('built-in', 'sklearn')),
     'maxpoints': _numeric(
         None, 'integer', range_='[1;inf)', ignore_none=True),
     'monte_carlo': _numeric(None, 'integer', range_='[1;inf)'),
     'noise': _generic(None, 'string',
                       value_in=('AWGN', 'AWLN', 'custom'), ignore_none=True),
     'problem_size': _numeric(None, 'integer', range_='[1;inf)'),
     'rho': _levels(None, (
         _generic(None, 'explicit collection'),
         _numeric(None, 'floating', range_='[0;1]'))),
     'seed': _numeric(None, 'integer', range_='[0;inf)', ignore_none=True),
     'SNR': _numeric(None, ('integer', 'floating')),
     'support_distribution': _numeric(None, 'floating', range_='[0;1]',
                                      shape=(-1, 1), ignore_none=True),
     'system_matrix': _generic(None, 'string',
                               value_in=('USE', 'RandomDCT2D', 'custom'))})
