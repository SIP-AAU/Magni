"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the `magni.cs.reconstruction.iht`
subpackage.

See also
--------
magni.cs.reconstruction._config.Configger : The Configger class used

Notes
-----
This module instantiates the `Configger` class provided by
`magni.utils.config`. The configuration options are the following:

iterations : int
    The maximum number of iterations to do (the default is 300).
kappa_fixed : float
    The relaxation parameter used in the algorithm (the default is 0.65).
precision_float : {np.float, np.float16, np.float32, np.float64, np.float128}
    The floating point precision used for the computations (the default is
    np.float64).
threshold : {'far', 'oracle'}
    The method for selecting the threshold value.
threshold_fixed : float
    The assumed rho value used for selecting the threshold value if using the
    oracle method.
tolerance : float
    The least acceptable ratio of residual to measurements (in 2-norm) to break
    the interations (the default is 0.001).

"""

from __future__ import division

import numpy as np

from magni.cs.reconstruction._config import Configger as _Configger
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric


configger = _Configger(
    {'iterations': 300,
     'kappa_fixed': 0.65,
     'precision_float': np.float64,
     'threshold': 'far',
     'threshold_fixed': 0.1,
     'tolerance': 1e-3},
    {'iterations': _numeric(None, 'integer', range_='[1;inf)'),
     'kappa_fixed': _numeric(None, 'floating', range_='[0;1]'),
     'precision_float': _generic(None, type, value_in=(
         np.float,
         getattr(np, 'float16', np.float_),
         getattr(np, 'float32', np.float_),
         getattr(np, 'float64', np.float_),
         getattr(np, 'float128', np.float_))),
     'threshold': _generic(None, 'string', value_in=('far', 'oracle')),
     'threshold_fixed': _numeric(None, 'floating', range_='[0;1]'),
     'tolerance': _numeric(None, 'floating', range_='[0;inf]')})
