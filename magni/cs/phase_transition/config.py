"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the phase_transition subpackage.

Routine listings
----------------
get(key=None)
    Get the value of one or more configuration options.
set(dictionary={}, \*\*kwargs)
    Set the value of one or more configuration options.

See Also
--------
magni.utils.config : The Configger class used.

Notes
-----
This module instantiates the `Configger` class provided by `magni.utils.config`
and assigns handles for the `get` and `set` methods of that class instance. The
configuration options are the following:

seed : int
    The seed used when picking seeds for generating data for the monte carlo
    simulations (the default is None, which implies an arbitrary seed).
n : int
    The length of the coefficient vector (the default is 800).
delta : list or tuple
    The delta values of the delta-rho grid whose points are used for the monte
    carlo simulations (the default is [0., 1.]).
rho : list or tuple
    The rho values of the delta-rho grid whose points are used for the monte
    carlo simulations (the default is [0., 1.]).
monte_carlo : int
    The number of monte carlo simulations to run in each point of the delta-rho
    grid (the default is 1).
coefficients : {'rademacher', 'gaussian'}
    The distribution which the non-zero coefficients in the coefficient vector
    are drawn from.

"""

from __future__ import division

from magni.utils.config import Configger as _Configger

_configger = _Configger(
    {'seed': None, 'n': 800, 'delta': [0.0, 1.0], 'rho': [0.0, 1.0],
     'monte_carlo': 1, 'coefficients': 'rademacher'},
    {'seed': {'type': int},
     'n': {'type': int, 'min': 1},
     'delta': [{'type_in': (list, tuple)},
               {'type': float, 'min': 0.0, 'max': 1.0}],
     'rho': [{'type_in': (list, tuple)},
             {'type': float, 'min': 0.0, 'max': 1.0}],
     'monte_carlo': {'type': int, 'min': 1},
     'coefficients': {'type': str, 'val_in': ['rademacher', 'gaussian']}})

get = _configger.get
set = _configger.set
