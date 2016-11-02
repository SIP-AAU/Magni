"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the core Approximate Message Passing (AMP) algorithm.

Routine listings
----------------
run(y, A)
    Run the AMP reconstruction algorithm.

See Also
--------
magni.cs.reconstruction.amp._config : Configuration options.
magni.cs.reconstruction.amp.threshold_operator : Threshold operators.
magni.cs.reconstruction.amp.stop_criterion : Stop criteria.
magni.cs.reconstruction.amp.util : Utilities.

Notes
-----
The default configuration of the AMP algorithm provides the Donoho, Maleki,
Montanari (DMM) AMP from [1]_ using the soft threshold with the "residual"
thresholding strategy from [2]_.

References
----------
.. [1] D. L. Donoho, A. Maleki, and A. Montanari, "Message-passing algorithms
   for compressed sensing", *Proceedings of the National Academy of Sciences of
   the United States of America*, vol. 106, no. 45, p. 18914-18919, Nov. 2009.
.. [2] A. Montanari, "Graphical models concepts in compressed sensing" *in
   Compressed Sensing: Theory and Applications*, Y. C. Eldar and G. Kutyniok
   (Ed.), Cambridge University Press, ch. 9, pp. 394-438, 2012.

"""

from __future__ import division
import copy

import numpy as np

from magni.cs.reconstruction.amp import config as _conf
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric


def run(y, A):
    """
    Run the AMP reconstruction algorithm.

    Parameters
    ----------
    y : ndarray
        The m x 1 measurement vector.
    A : ndarray or magni.utils.matrices.{Matrix, MatrixCollection}
        The m x n matrix which is the product of the measurement matrix and the
        dictionary matrix.

    Returns
    -------
    alpha : ndarray
        The n x 1 reconstructed coefficient vector.
    history : dict, optional
        The dictionary of various measures tracked in the GAMP iterations.

    See Also
    --------
    magni.cs.reconstruction.amp._config : Configuration options.
    magni.cs.reconstruction.amp.threshold_operator : Threshold operators.
    magni.cs.reconstruction.amp.stop_criterion : Stop criteria.
    magni.cs.reconstruction.amp.util : Utilities.

    Notes
    -----
    Optionally, the algorithm may be configured to save and return the
    iteration history along with the reconstruction result. The returned
    history contains the following:

    * alpha_bar : Coefficient estimates (the reconstruction coefficients).
    * MSE : solution mean squared error (if the true solution is known).
    * threshold_parameters : The state of the threshold parameters.
    * stop_criterion : The currently used stop criterion.
    * stop_criterion_value : The value of the stop criterion.
    * stop_iteration : The iteration at which the algorithm stopped.
    * stop_reason : The reason for termination of the algorithm.

    Examples
    --------
    For example, recovering a vector from noiseless measurements using AMP
    with soft thresholding

    >>> import numpy as np
    >>> from magni.cs.reconstruction.amp import run, config
    >>> from magni.cs.reconstruction.amp.util import theta_mm
    >>> np.random.seed(seed=6028)
    >>> k, m, n = 10, 200, 400
    >>> A = 1 / np.sqrt(m) * np.random.randn(m, n)
    >>> alpha = np.zeros((n, 1))
    >>> alpha[:k] = np.random.normal(scale=2, size=(k, 1))
    >>> np_printoptions = np.get_printoptions()
    >>> np.set_printoptions(suppress=True, threshold=k+2)
    >>> alpha[:k + 2]
    array([[ 1.92709461],
           [ 0.74378508],
           [-3.2418159 ],
           [-1.32277347],
           [ 0.90118   ],
           [-0.19157262],
           [ 0.82855712],
           [ 0.24817994],
           [-1.43034777],
           [-0.21232344],
           [ 0.        ],
           [ 0.        ]])
    >>> y = A.dot(alpha)
    >>> threshold_params = {'threshold_level_update_method': 'residual',
    ... 'theta': theta_mm(float(m) / n), 'tau_hat_sq': 1.0}
    >>> config['threshold_parameters'] = threshold_params
    >>> alpha_hat = run(y, A)
    >>> alpha_hat[:k + 2]
    array([[ 1.92612707],
           [ 0.74185284],
           [-3.24179313],
           [-1.32200929],
           [ 0.90089208],
           [-0.19097308],
           [ 0.82658038],
           [ 0.24515825],
           [-1.42980997],
           [-0.2111469 ],
           [ 0.00002815],
           [ 0.00047293]])
    >>> np.sum(np.abs(alpha - alpha_hat) > 1e-2)
    0

    or recover the same vector returning a history comparing the pr. iteration
    solution to the true vector

    >>> config['report_history'] = True
    >>> config['true_solution'] = alpha
    >>> alpha_hat, history = run(y, A)
    >>> alpha_hat[:k + 2]
    array([[ 1.92612707],
           [ 0.74185284],
           [-3.24179313],
           [-1.32200929],
           [ 0.90089208],
           [-0.19097308],
           [ 0.82658038],
           [ 0.24515825],
           [-1.42980997],
           [-0.2111469 ],
           [ 0.00002815],
           [ 0.00047293]])
    >>> np.array(history['MSE']).reshape(-1, 1)[1:11]
    array([[ 0.01273323],
           [ 0.0053925 ],
           [ 0.00276334],
           [ 0.00093472],
           [ 0.0004473 ],
           [ 0.00021142],
           [ 0.00009618],
           [ 0.00004371],
           [ 0.00002275],
           [ 0.00001053]])

    or recover the same vector using the median based update method

    >>> config['report_history'] = False
    >>> threshold_params['threshold_level_update_method'] = 'median'
    >>> config['threshold_parameters'] = threshold_params
    >>> alpha_hat = run(y, A)
    >>> alpha_hat[:k + 2]
    array([[ 1.92577079],
           [ 0.7413161 ],
           [-3.24131125],
           [-1.32044146],
           [ 0.89998407],
           [-0.18981235],
           [ 0.82584016],
           [ 0.24445332],
           [-1.42934708],
           [-0.21043221],
           [ 0.        ],
           [ 0.        ]])
    >>> np.sum(np.abs(alpha - alpha_hat) > 1e-2)
    0

    >>> np.set_printoptions(**np_printoptions)

    """

    @_decorate_validation
    def validate_input():
        _numeric('y', ('integer', 'floating', 'complex'), shape=(-1, 1))
        _numeric('A', ('integer', 'floating', 'complex'), shape=(
            y.shape[0],
            _conf['true_solution'].shape[0]
            if _conf['true_solution'] is not None else -1))

    @_decorate_validation
    def validate_output():
        _numeric('alpha', ('integer', 'floating'), shape=(A.shape[1], 1))
        _generic('history', 'mapping',
                 keys_in=('alpha_bar', 'MSE', 'threshold_parameters',
                          'stop_criterion', 'stop_criterion_value',
                          'stop_iteration', 'stop_reason'))

    validate_input()

    # Initialisation
    init = _get_amp_initialisation(y, A)
    AH = init['AH']
    chi = init['chi']
    delta = init['delta']
    alpha_bar = init['alpha_bar']
    AH_dot_chi = init['AH_dot_chi']
    threshold = init['threshold']
    threshold_parameters = init['threshold_parameters']
    stop_criterion = init['stop_criterion']
    stop_criterion_name = init['stop_criterion_name']
    iterations = init['iterations']
    tolerance = init['tolerance']
    convert = init['convert']
    report_history = init['report_history']
    history = init['history']
    true_solution = init['true_solution']

    # AMP iterations
    for it in range(iterations):
        # Save previous state
        alpha_bar_prev = alpha_bar

        # AMP state updates
        # Basic AMP treshold operand: alpha_bar + AH_dot_chi
        alpha_bar = threshold.compute_threshold(locals())
        A_dot_alpha_bar = A.dot(alpha_bar)
        # Basic AMP deriv threshold operand: alpha_bar_prev + AH_dot_chi
        chi = y - A_dot_alpha_bar + 1 / delta * np.mean(
            threshold.compute_deriv_threshold(locals())) * chi
        AH_dot_chi = AH.dot(chi)

        # Threshold level update
        threshold.update_threshold_level(locals())

        # Stop criterion
        stop, stop_criterion_value = stop_criterion.compute(locals())

        # History reporting
        if report_history:
            history['alpha_bar'].append(alpha_bar)
            history['threshold_parameters'].append(
                copy.deepcopy(threshold.__dict__))
            history['stop_criterion_value'].append(stop_criterion_value)
            history['stop_iteration'] = it
            if true_solution is not None:
                history['MSE'].append(
                    1/A.shape[1] * np.linalg.norm(
                        true_solution - alpha_bar)**2)

        if stop:
            history['stop_reason'] = stop_criterion_name.upper()
            break

    alpha = alpha_bar

    validate_output()

    if report_history:
        return alpha, history
    else:
        return alpha


def _get_amp_initialisation(y, A):
    """
    Return an initialisation of the AMP algorithm.

    Parameters
    ----------
    y : ndarray
        The m x 1 measurement vector.
    A : ndarray or magni.utils.matrices.{Matrix, MatrixCollection}
        The m x n matrix which is the product of the measurement matrix and the
        dictionary matrix.

    Returns
    -------
    init : dict
        The initialisation of the the AMP algorithm.

    """

    init = dict()
    param = dict(_conf.items())

    # Configured setup
    init['convert'] = param['precision_float']
    init['report_history'] = param['report_history']
    init['tolerance'] = param['tolerance']
    init['iterations'] = param['iterations']
    init['true_solution'] = param['true_solution']
    init['threshold_parameters'] = param['threshold_parameters']
    init['stop_criterion_name'] = param['stop_criterion'].__name__

    # Arguments based configuration
    init['AH'] = A.conj().T
    init['delta'] = A.shape[0] / A.shape[1]

    # threshold and stop criterion configuration
    threshold_init = copy.copy(init)
    threshold_init['A'] = A
    threshold_init['y'] = y
    init['threshold'] = param['threshold'](threshold_init)
    init['stop_criterion'] = param['stop_criterion'](threshold_init)

    # Configuration of remaining states
    if param['warm_start'] is not None:
        init['alpha_bar'] = init['convert'](param['warm_start'])
    else:
        init['alpha_bar'] = np.zeros((A.shape[1], 1), dtype=init['convert'])

    init['history'] = {'alpha_bar': [init['alpha_bar']],
                       'MSE': [np.nan],
                       'threshold_parameters': [
                           init['threshold_parameters']],
                       'stop_criterion': init['stop_criterion_name'].upper(),
                       'stop_criterion_value': [np.nan],
                       'stop_iteration': 0,
                       'stop_reason': 'MAX_ITERATIONS'}
    init['chi'] = np.zeros_like(y)  # by convention
    init['AH_dot_chi'] = init['AH'].dot(y - A.dot(init['alpha_bar']))

    return init
