"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the `magni.cs.reconstruction.amp`
subpackage.

See also
--------
magni.cs.reconstruction._config.Configger : The Configger class used.

Notes
-----
This module instantiates the `Configger` class provided by
`magni.cs.reconstruction._config.Configger`. The configuration options are the
following:

iterations : int
    The maximum number of iterations to do (the default is 300).
precision_float : {np.float, np.float16, np.float32, np.float64, np.float128,
    np.complex64, np.complex128, np.complex256}
    The floating point precision used for the computations (the default is
    np.float64).
report_history : bool
    The indicator of whether or not to return the progress history along with
    the result (the default is False).
stop_criterion : magni.utils.validation.types.StopCriterion
    The stop criterion to use in the iterations (the default is
    magni.cs.reconstruction.gamp.stop_criterion.MSEConvergence).
threshold : magni.utils.validation.types.ThresholdOperator
    The threshold operator to use (the default is
    magni.cs.reconstruction.amp.threshold_operator.SoftThreshold).
threshold_parameters : dict
    The parameters used in the threshold operator (no default is provided,
    which implies that this must be specified by the user).
tolerance : float
    The least acceptable stop criterion tolerance to break the interations (the
    default is 1e-6).
true_solution : ndarray or None
    The true solution to allow for tracking the convergence of the algorithm in
    the artificial setup where the true solution is known a-priori (the default
    is None, which implies that no true solution tracking is used).
warm_start : ndarray
    The initial guess of the solution vector (alpha_bar) (the default is
    None, which implies that alpha_bar is taken to be a vector of zeros).

"""

from __future__ import division

import numpy as np

from magni.cs.reconstruction._config import Configger as _Configger
from magni.cs.reconstruction.amp.stop_criterion import (
    MSEConvergence as _MSEConvergence)
from magni.cs.reconstruction.amp.threshold_operator import (
    SoftThreshold as _SoftThreshold)
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.validation.types import (
    ThresholdOperator as _ThresholdOperator, StopCriterion as _StopCriterion)


configger = _Configger(
    {'iterations': 300,
     'precision_float': np.float64,
     'report_history': False,
     'stop_criterion': _MSEConvergence,
     'threshold': _SoftThreshold,
     'threshold_parameters': dict(),
     'tolerance': 1e-6,
     'true_solution': None,
     'warm_start': None},
    {'iterations': _numeric(None, 'integer', range_='[1;inf)'),
     'precision_float': _generic(None, type, value_in=(
         np.float,
         getattr(np, 'float16', np.float_),
         getattr(np, 'float32', np.float_),
         getattr(np, 'float64', np.float_),
         getattr(np, 'float128', np.float_),
         getattr(np, 'complex64', np.complex_),
         getattr(np, 'complex128', np.complex_),
         getattr(np, 'complex256', np.complex_))),
     'report_history': _numeric(None, 'boolean'),
     'stop_criterion': _generic(None, 'class', superclass=_StopCriterion),
     'threshold': _generic(None, 'class', superclass=_ThresholdOperator),
     'threshold_parameters': _generic(None, 'mapping'),
     'tolerance': _numeric(None, 'floating', range_='[0;inf)'),
     'true_solution': _numeric(
         None, ('integer', 'floating', 'complex'), shape=(-1, 1),
         ignore_none=True),
     'warm_start': _numeric(None, ('integer', 'floating', 'complex'),
                            shape=(-1, 1), ignore_none=True)})
