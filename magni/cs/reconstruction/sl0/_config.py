r"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the `magni.cs.reconstruction.sl0`
subpackage.

See also
--------
magni.cs.reconstruction._config.Configger : The Configger class used

Notes
-----
This module instantiates the `Configger` class provided by
`magni.utils.config`. The configuration options are the following:

epsilon : float
    The precision parameter used in centering (the default is 0.01).
L : {'geometric', 'fixed'}
    The method for selecting the maximum number of gradient descent iterations
    for each sigma.
L_fixed : int
    The value used for L if using the 'fixed' method (the default is 2.0).
L_geometric_ratio : float
    The common ratio used for the L update if using the 'geometric' method (the
    default is 2.0).
L_geometric_start : float
    The starting value used for the L if using the 'geometric' method (the
    default is 2.0).
mu : {'step', 'fixed'}
    The method for selecting the relative step-size used in gradient descent
    iteration.
mu_fixed : float
    The value used for mu if using the 'fixed' method (the default is 1.0).
mu_step_end : float
    The value used for mu for the last iterations if using the 'step' method
    (the default is 1.5).
mu_step_iteration : int
    The iteration where the value used for mu changes if using the 'step'
    method
mu_step_start : float
    The value used for mu for the first iterations if using the 'step' method
    (the default is 0.001).
precision_float : {np.float, np.float16, np.float32, np.float64, np.float128}
    The floating point precision used for the computations (the default is
    np.float64).
sigma_geometric : float
    The common ratio used for the sigma update (the default is 0.7).
sigma_start : {'reciprocal', 'fixed'}
    The method for selecting the starting value of sigma.
sigma_start_fixed : float
    The fixed factor multiplied onto the maximum coefficient of a least squares
    solution to obtain the value if using the 'fixed' method (the
    default is 2.0).
sigma_start_reciprocal : float
    The constant in the factor :math:`\frac{1}{constant \cdot \delta}`
    multiplied onto the maximum coefficient of a least squares solution to
    obtain the value if using the 'reciprocal' method (the default is 2.75).
sigma_stop_fixed : float
    The minimum value of std. dev. in Gaussian l0 approx (the default 0.01).

"""

from __future__ import division

import numpy as np

from magni.cs.reconstruction._config import Configger as _Configger
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric


configger = _Configger(
    {'epsilon': 1e-2,
     'L': 'geometric',
     'L_fixed': 2,
     'L_geometric_ratio': 2.0,
     'L_geometric_start': 2.0,
     'mu': 'step',
     'mu_fixed': 1.0,
     'mu_step_end': 1.5,
     'mu_step_iteration': 4,
     'mu_step_start': 0.001,
     'precision_float': np.float64,
     'sigma_geometric': 0.7,
     'sigma_start': 'reciprocal',
     'sigma_start_fixed': 2.0,
     'sigma_start_reciprocal': 2.75,
     'sigma_stop_fixed': 0.01},
    {'epsilon': _numeric(None, 'floating', range_='(0;inf)'),
     'L': _generic(None, 'string', value_in=('fixed', 'geometric')),
     'L_fixed': _numeric(None, 'integer', range_='[1;inf)'),
     'L_geometric_ratio': _numeric(None, 'floating', range_='(0;inf)'),
     'L_geometric_start': _numeric(None, 'floating', range_='[1;inf)'),
     'mu': _generic(None, 'string', value_in=('fixed', 'step')),
     'mu_fixed': _numeric(None, 'floating', range_='(0;inf)'),
     'mu_step_end': _numeric(None, 'floating', range_='(0;inf)'),
     'mu_step_iteration': _numeric(None, 'integer', range_='(0;inf)'),
     'mu_step_start': _numeric(None, 'floating', range_='(0;inf)'),
     'precision_float': _generic(None, type, value_in=(
         np.float,
         getattr(np, 'float16', np.float_),
         getattr(np, 'float32', np.float_),
         getattr(np, 'float64', np.float_),
         getattr(np, 'float128', np.float_))),
     'sigma_geometric': _numeric(None, 'floating', range_='(0;1)'),
     'sigma_start': _generic(None, 'string', value_in=('fixed', 'reciprocal')),
     'sigma_start_fixed': _numeric(None, 'floating', range_='(0;inf)'),
     'sigma_start_reciprocal': _numeric(None, 'floating', range_='(0;inf)'),
     'sigma_stop_fixed': _numeric(None, 'floating', range_='(0;inf)')})
