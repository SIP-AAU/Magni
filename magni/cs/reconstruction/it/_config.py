"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the `magni.cs.reconstruction.it`
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
kappa : {'fixed', 'adaptive'}
    The method used to calculate the step-size (relaxation) parameter kappa.
kappa_fixed : float
    The step-size (relaxation parameter) used in the algorithm when a fixed
    step-size is used (the default is 0.65).
precision_float : {np.float, np.float16, np.float32, np.float64, np.float128}
    The floating point precision used for the computations (the default is
    np.float64).
threshold : {'far', 'fixed'}
    The method used for calculating the threshold value.
threshold_fixed : int
    The number of non-zero coefficients in the signal vector when this number
    is assumed fixed.
threshold_operator : {'hard', 'soft', 'weighted_hard', 'weighted_soft', 'none'}
    The threshold operator used in the backprojection step.
threshold_weights : ndarray
    Array of weights to be used in one of the weighted threshold operators (the
    default is array([[1]]), which implies that all coefficients are weighted
    equally).
tolerance : float
    The least acceptable ratio of residual to measurements (in 2-norm) to break
    the interations (the default is 0.001).
warm_start : ndarray
    The initial guess of the solution vector (the default is None, which
    implies that a vector of zeros is used).

"""

from __future__ import division

import numpy as np

from magni.cs.reconstruction._config import Configger as _Configger
from magni.cs.reconstruction.it import _util
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric

configger = _Configger(
    {'iterations': 300,
     'kappa': 'fixed',
     'kappa_fixed': 0.65,
     'precision_float': np.float64,
     'threshold': 'far',
     'threshold_fixed': 1,
     'threshold_operator': 'hard',
     'threshold_weights': np.array([[1]]),
     'tolerance': 1e-3,
     'warm_start': None},
    {'iterations': _numeric(None, 'integer', range_='[1;inf)'),
     'kappa': _generic(
         None, 'string', value_in=_util._get_methods('step_size')),
     'kappa_fixed': _numeric(None, 'floating', range_='(0;inf)'),
     'precision_float': _generic(None, type, value_in=(
         np.float,
         getattr(np, 'float16', np.float_),
         getattr(np, 'float32', np.float_),
         getattr(np, 'float64', np.float_),
         getattr(np, 'float128', np.float_))),
     'threshold': _generic(
         None, 'string', value_in=_util._get_methods('threshold')),
     'threshold_fixed': _numeric(None, 'integer', range_='(0;inf)'),
     'threshold_operator': _generic(
         None, 'string', value_in=_util._get_operators('threshold_operators')),
     'threshold_weights': _numeric(
         None, ('integer', 'floating'), shape=(-1, 1)),
     'tolerance': _numeric(None, 'floating', range_='[0;inf]'),
     'warm_start': _numeric(
         None, ('integer', 'floating', 'complex'), shape=(-1, 1),
         ignore_none=True)})
