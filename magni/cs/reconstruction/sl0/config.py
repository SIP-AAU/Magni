"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the `magni.cs.reconstruction.sl0`
subpackage.

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

epsilon : float
    The precision parameter used in centering (the default is 0.01).
L : float
    The maximum number of gradient descent iterations for each sigma (the
    default is 2.0).
L_update : float
    The additive/multiplicative update of L when it is not fixed (the default
    is 2.0).
mu : float
    The relative step-size used in gradient descent iteration (the default is
    1.0).
mu_start : float
    The relative step-size used in gradient descent iteration for the first
    iterations (the default is 0.001).
mu_end :float
    The relative step-size used in gradient descent iteration for the last
    iterations (the default is 1.5).
precision_float : dtype
    The floating point precision used for the computations (the default is
    float64).
seq_algorithm : {'mod', 'std'}
    The sequential implementation of SL0 used (the default is 'mod').
sigma_update : float
    The constant used in the geometric sequence of sigmas (the default is 0.7).
sigma_min : float
    The minimum value of std. dev. in Gaussian l0 approx (the default 0.01).

"""

from __future__ import division

import numpy as np

from magni.utils.config import Configger as _Configger


_configger = _Configger(
    {'sigma_update': 0.7, 'sigma_min': 0.01, 'L': 2.0, 'L_update': 2.0,
     'mu': 1.0, 'mu_start': 0.001, 'mu_end': 1.5, 'epsilon': 1e-2,
     'precision_float': np.float64, 'algorithm': 'mod'},
    {'sigma_update': {'type': float},
     'sigma_min': {'type': float},
     'L': {'type': float},
     'L_update': {'type': float},
     'mu': {'type': float},
     'mu_start': {'type': float},
     'mu_end': {'type': float},
     'epsilon': {'type': float},
     'precision_float': {'type': type},
     'algorithm': {'type': str, 'val_in': ['std', 'mod']}})

set = _configger.set


def get(key=None):
    """
    Get the value of one or more configuration options.

    This function wraps 'Configger.get' in order to convert any float options
    to the specified precision before returning the option(s).

    See Also
    --------
    magni.utils.config.Configger.get : The wrapped function.

    """

    value = _configger.get(key)
    convert = _configger.get('precision_float')

    if isinstance(value, dict):
        for key, val in value.items():
            if isinstance(val, float):
                value[key] = convert(val)
    elif isinstance(value, float):
        value = convert(value)

    return value
