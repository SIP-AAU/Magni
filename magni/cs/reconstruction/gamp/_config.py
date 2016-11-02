"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the `magni.cs.reconstruction.gamp`
subpackage.

See also
--------
magni.cs.reconstruction._config.Configger : The Configger class used

Notes
-----
This module instantiates the `Configger` class provided by
`magni.cs.reconstruction._config.Configger`. The configuration options are the
following:

damping : float
    The damping applied to the variable side updates (the default is 0.0).
input_channel : magni.utils.validation.types.MMSEInputChannel
    The input channel to use (the default is
    magni.cs.reconstruction.gamp.input_channel.IIDBG).
input_channel_parameters : dict
    The parameters used in the input channel (no default is provided, which
    implies that this must be specified by the user).
iterations : int
    The maximum number of iterations to do (the default is 300).
output_channel : magni.utils.validation.types.MMSEOutputChannel
    The output channel to use (the default is
    magni.cs.reconstruction.gamp.output_channel.AWGN).
output_channel_parameters : dict
    The parameters used in the output channel (no default is provided, which
    implies that this must be specified by the user).
precision_float : {np.float, np.float16, np.float32, np.float64, np.float128,
    np.complex64, np.complex128, np.complex256}
    The floating point precision used for the computations (the default is
    np.float64).
report_A_asq_setup : bool
    The indicator of whether or not to print the A_asq details (the default is
    False).
report_history : bool
    The indicator of whether or not to return the progress history along with
    the result (the default is False).
stop_criterion : magni.utils.validation.types.StopCriterion
    The stop criterion to use in the iterations (the default is
    magni.cs.reconstruction.gamp.stop_criterion.MSEConvergence).
sum_approximation_constant : dict
    The method and constant used in a sum approximation of the squared system
    transform (the default is {'rangan': 1.0}, which implies that Rangan's
    uniform variance methods with the system matrix adapted ||A||_F^2/(m*n)
    constant is used).
tolerance : float
    The least acceptable stop criterion tolerance to break the interations (the
    default is 1e-6).
true_solution : ndarray or None
    The true solution to allow for tracking the convergence of the algorithm in
    the artificial setup where the true solution is known a-priori (the default
    is None, which implies that no true solution tracking is used).
warm_start : list or tuple
    The collection containing the initial guess of the solution vector
    (alpha_bar) and the solution variance vector (alpha_tilde) (the default is
    None, which implies that alpha_bar is taken to be a vector of zeros and
    alpha_tilde is taken to be a vector of ones).

"""

from __future__ import division

import numpy as np

from magni.cs.reconstruction._config import Configger as _Configger
from magni.cs.reconstruction.gamp.stop_criterion import (
    MSEConvergence as _MSEConvergence)
from magni.cs.reconstruction.gamp.input_channel import IIDBG as _IIDBG
from magni.cs.reconstruction.gamp.output_channel import AWGN as _AWGN
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.validation.types import (
    MMSEInputChannel as _MMSEInputChannel,
    MMSEOutputChannel as _MMSEOutputChannel, StopCriterion as _StopCriterion)


configger = _Configger(
    {'damping': 0.0,
     'input_channel': _IIDBG,
     'input_channel_parameters': dict(),
     'iterations': 300,
     'output_channel': _AWGN,
     'output_channel_parameters': dict(),
     'precision_float': np.float64,
     'report_A_asq_setup': False,
     'report_history': False,
     'stop_criterion': _MSEConvergence,
     'sum_approximation_constant': {'rangan': 1.0},
     'tolerance': 1e-6,
     'true_solution': None,
     'warm_start': None},
    {'damping': _numeric(None, 'floating', range_='[0;1)'),
     'input_channel': _generic(None, 'class', superclass=_MMSEInputChannel),
     'input_channel_parameters': _generic(None, 'mapping'),
     'iterations': _numeric(None, 'integer', range_='[1;inf)'),
     'output_channel': _generic(None, 'class', superclass=_MMSEOutputChannel),
     'output_channel_parameters': _generic(None, 'mapping'),
     'precision_float': _generic(None, type, value_in=(
         np.float,
         getattr(np, 'float16', np.float_),
         getattr(np, 'float32', np.float_),
         getattr(np, 'float64', np.float_),
         getattr(np, 'float128', np.float_),
         getattr(np, 'complex64', np.complex_),
         getattr(np, 'complex128', np.complex_),
         getattr(np, 'complex256', np.complex_))),
     'report_A_asq_setup': _numeric(None, 'boolean'),
     'report_history': _numeric(None, 'boolean'),
     'stop_criterion': _generic(None, 'class', superclass=_StopCriterion),
     'sum_approximation_constant': _levels(None, (
         _generic(None, 'mapping', keys_in=('rangan', 'krzakala'), len_=1),
         _numeric(None, ('integer', 'floating'), range_='(0;inf)'))),
     'tolerance': _numeric(None, 'floating', range_='[0;inf)'),
     'true_solution': _numeric(
         None, ('integer', 'floating', 'complex'), shape=(-1, 1),
         ignore_none=True),
     'warm_start': _levels(None, (
         _generic(None, 'explicit collection', len_=2, ignore_none=True),
         _numeric(None, ('integer', 'floating', 'complex'), shape=(-1, 1))))})
