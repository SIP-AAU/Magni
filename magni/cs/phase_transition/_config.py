"""
..
    Copyright (c) 2014-2015, Magni developers.
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

coefficients : {'rademacher', 'gaussian'}
    The distribution which the non-zero coefficients in the coefficient vector
    are drawn from.
delta : list or tuple
    The delta values of the delta-rho grid whose points are used for the monte
    carlo simulations (the default is [0., 1.]).
monte_carlo : int
    The number of monte carlo simulations to run in each point of the delta-rho
    grid (the default is 1).
problem_size : int
    The length of the coefficient vector (the default is 800).
rho : list or tuple
    The rho values of the delta-rho grid whose points are used for the monte
    carlo simulations (the default is [0., 1.]).
seed : int
    The seed used when picking seeds for generating data for the monte carlo
    simulations (the default is None, which implies an arbitrary seed).

"""

from __future__ import division

from magni.utils.config import Configger as _Configger
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric

configger = _Configger(
    {'coefficients': 'rademacher',
     'delta': [0.0, 1.0],
     'monte_carlo': 1,
     'problem_size': 800,
     'rho': [0.0, 1.0],
     'seed': None},
    {'coefficients': _generic(None, 'string',
                              value_in=('rademacher', 'gaussian')),
     'delta': _levels(None, (
         _generic(None, 'explicit collection'),
         _numeric(None, 'floating', range_='[0;1]'))),
     'monte_carlo': _numeric(None, 'integer', range_='[1;inf)'),
     'problem_size': _numeric(None, 'integer', range_='[1;inf)'),
     'rho': _levels(None, (
         _generic(None, 'explicit collection'),
         _numeric(None, 'floating', range_='[0;1]'))),
     'seed': _numeric(None, 'integer', range_='[0;inf)', ignore_none=True)})
