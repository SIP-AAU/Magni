"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the `magni.cs.reconstruction.iht`
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

iterations : int
    The maximum number of iterations to do (the default is 300).
kappa : float
    The relaxation parameter used in the algorithm (the default is 0.65).
precision_float : dtype
    The floating point precision used for the computations (the default is
    float64).
threshold : ['far', 'oracle']
    The method for selecting the threshold value.
threshold_rho : float
    The assumed rho value used for selecting the threshold value if using the
    oracle method.
tolerance : float
    The least acceptable ratio of residual to measurements (in 2-norm) to break
    the interations (the default is 0.001).

"""

from __future__ import division

import numpy as np

from magni.utils.config import Configger as _Configger


_configger = _Configger(
    {'kappa': 0.65, 'tolerance': 1e-3, 'iterations': 300, 'threshold': 'far',
     'threshold_rho': 0.1, 'precision_float': np.float64},
    {'kappa': {'type': float, 'min': 0},
     'tolerance': {'type': float, 'min': 0},
     'iterations': {'type': int, 'min': 1},
     'threshold': {'type': str, 'val_in': ('far', 'oracle')},
     'threshold_rho': {'type': float, 'min': 0, 'max': 1},
     'precision_float': {'type': type}})

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
