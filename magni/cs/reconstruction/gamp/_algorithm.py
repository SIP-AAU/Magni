"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the core Generalised Approximate Message Passing (GAMP)
algorithm.

Routine listings
----------------
run(y, A, A_asq=None)
    Run the GAMP reconstruction algorithm.

See Also
--------
magni.cs.reconstruction.gamp._config : Configuration options.
magni.cs.reconstruction.gamp.input_channel : Available input channels.
magni.cs.reconstruction.gamp.output_channel : Available output channels.
magni.cs.reconstruction.gamp.stop_criterion : Available stop critria.

Notes
-----
The default configuration of the GAMP algorithm provides the s-GB AMP algorithm
from [1]_ using an MSE convergence based stop criterion. Both the input
channel, the output channel, and the stop criterion may be changed.

This implementation allows for the use of sum approximations of the squared
system matrix as detailed in [1]_ and [2]_. Furthermore, a simple damping
option is available based on the description in [3]_ (see also [4]_ for more
details on damping in GAMP).

References
----------
.. [1] F. Krzakala, M. Mezard, F. Sausset, Y. Sun, and L. Zdeborova,
   "Probabilistic reconstruction in compressed sensing: algorithms, phase
   diagrams, and threshold achieving matrices", *Journal of Statistical
   Mechanics: Theory and Experiment*, vol. P08009, pp. 1-57, Aug. 2012.
.. [2] S. Rangan, "Generalized Approximate Message Passing for Estimation
   with Random Linear Mixing", arXiv:1010.5141v2, pp. 1-22, Aug. 2012.
.. [3] S. Rangan, P. Schniter, and A. Fletcher. "On the Convergence of
   Approximate Message Passing with Arbitrary Matrices", *in IEEE International
   Symposium on Information Theory (ISIT)*, pp. 236-240, Honolulu, Hawaii, USA,
   Jun. 29 - Jul. 4, 2014.
.. [4] J. Vila, P. Schniter, S. Rangan, F. Krzakala, L. Zdeborova, "Adaptive
   Damping and Mean Removal for the Generalized Approximate Message Passing
   Algorithm", *in IEEE International Conference on Acoustics, Speech, and
   Signal Processing (ICASSP)*, South Brisbane, Queensland, Australia, Apr.
   19-24, 2015, pp. 2021-2025.

"""

from __future__ import division
import copy

import numpy as np

from magni.cs.reconstruction.gamp import config as _conf
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.validation.types import MatrixBase as _MatrixBase
from magni.utils.matrices import norm as _norm
from magni.utils.matrices import (SumApproximationMatrix as
                                  _SumApproximationMatrix)


def run(y, A, A_asq=None):
    """
    Run the GAMP reconstruction algorithm.

    Parameters
    ----------
    y : ndarray
        The m x 1 measurement vector.
    A : ndarray or magni.utils.matrices.{Matrix, MatrixCollection}
        The m x n matrix which is the product of the measurement matrix and the
        dictionary matrix.
    A_asq : ndarray or magni.utils.matrices.{Matrix, MatrixCollection} or None
        The m x n matrix which is the entrywise absolute value squared product
        of the measurement matrix and the dictionary matrix (the default is
        None, which implies that a sum approximation is used).

    Returns
    -------
    alpha : ndarray
        The n x 1 reconstructed coefficient vector.
    history : dict, optional
        The dictionary of various measures tracked in the GAMP iterations.

    See Also
    --------
    magni.cs.reconstruction.gamp._config : Configuration options.
    magni.cs.reconstruction.gamp.input_channel : Input channels.
    magni.cs.reconstruction.gamp.output_channel : Output channels.
    magni.cs.reconstruction.gamp.stop_criterion : Stop criteria.

    Notes
    -----
    Optionally, the algorithm may be configured to save and return the
    iteration history along with the reconstruction result. The returned
    history contains the following:

    * alpha_bar : Mean coefficient estimates (the reconstruction coefficients).
    * alpha_tilde : Variance coefficient estimates.
    * MSE : solution Mean squared error (if the true solution is known).
    * input_channel_parameters : The state of the input channel.
    * output_channel_parameters : The state of the output channel.
    * stop_criterion : The currently used stop criterion.
    * stop_criterion_value : The value of the stop criterion.
    * stop_iteration : The iteration at which the algorithm stopped.
    * stop_reason : The reason for termination of the algorithm.

    Examples
    --------
    For example, recovering a vector from AWGN noisy measurements using GAMP

    >>> import numpy as np
    >>> from magni.cs.reconstruction.gamp import run, config
    >>> np.random.seed(seed=6028)
    >>> k, m, n = 10, 200, 400
    >>> tau = float(k) / n
    >>> A = 1 / np.sqrt(m) * np.random.randn(m, n)
    >>> A_asq = np.abs(A)**2
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
    >>> sigma = 0.15
    >>> y = A.dot(alpha) + np.random.normal(scale=sigma, size=(A.shape[0], 1))
    >>> input_channel_params = {'tau': tau, 'theta_bar': 0, 'theta_tilde': 4,
    ... 'use_em': False}
    >>> config['input_channel_parameters'] = input_channel_params
    >>> output_channel_params = {'sigma_sq': sigma**2,
    ... 'noise_level_estimation': 'fixed'}
    >>> config['output_channel_parameters'] = output_channel_params
    >>> alpha_hat = run(y, A, A_asq)
    >>> alpha_hat[:k + 2]
    array([[ 1.93810961],
           [ 0.6955502 ],
           [-3.39759349],
           [-1.35533562],
           [ 1.10524227],
           [-0.00594848],
           [ 0.79274671],
           [ 0.04895264],
           [-1.08726071],
           [-0.00142911],
           [ 0.00022861],
           [-0.00004272]])
    >>> np.sum(np.abs(alpha - alpha_hat) > sigma * 3)
    0

    or recover the same vector returning a history comparing the pr. iteration
    solution to the true vector and printing the A_asq details

    >>> config['report_A_asq_setup'] = True
    >>> config['report_history'] = True
    >>> config['true_solution'] = alpha
    >>> alpha_hat, history = run(y, A, A_asq) # doctest: +NORMALIZE_WHITESPACE
    GAMP is using the A_asq: [[ 0.024 ..., 0.002]
     ...,
     [ 0. ..., 0.014]]
    The sum approximation method is: None
    >>> alpha_hat[:k + 2]
    array([[ 1.93810961],
           [ 0.6955502 ],
           [-3.39759349],
           [-1.35533562],
           [ 1.10524227],
           [-0.00594848],
           [ 0.79274671],
           [ 0.04895264],
           [-1.08726071],
           [-0.00142911],
           [ 0.00022861],
           [-0.00004272]])
    >>> np.array(history['MSE']).reshape(-1, 1)[1:11]
    array([[ 0.04562729],
           [ 0.01328304],
           [ 0.00112098],
           [ 0.00074968],
           [ 0.00080175],
           [ 0.00076615],
           [ 0.00077043]])

    or recover the same vector using sample variance AWGN noise level
    estimation

    >>> config['report_A_asq_setup'] = False
    >>> config['report_history'] = False
    >>> output_channel_params['noise_level_estimation'] = 'sample_variance'
    >>> config['output_channel_parameters'] = output_channel_params
    >>> alpha_hat = run(y, A, A_asq)
    >>> alpha_hat[:k + 2]
    array([[ 1.94820622],
           [ 0.72162206],
           [-3.39978431],
           [-1.35357001],
           [ 1.10701779],
           [-0.00834467],
           [ 0.79790879],
           [ 0.08441384],
           [-1.08946306],
           [-0.0015894 ],
           [ 0.00020561],
           [-0.00003623]])
    >>> np.sum(np.abs(alpha - alpha_hat) > sigma * 3)
    0

    or recover the same vector using median AWGN noise level estimation

    >>> output_channel_params['noise_level_estimation'] = 'median'
    >>> config['output_channel_parameters'] = output_channel_params
    >>> alpha_hat = run(y, A, A_asq)
    >>> alpha_hat[:k + 2]
    array([[ 1.93356483],
           [ 0.65232347],
           [-3.39440429],
           [-1.35437724],
           [ 1.10312573],
           [-0.0050555 ],
           [ 0.78743162],
           [ 0.03616397],
           [-1.08589927],
           [-0.00136802],
           [ 0.00024121],
           [-0.00004498]])
    >>> np.sum(np.abs(alpha - alpha_hat) > sigma * 3)
    0

    or recover the same vector learning the AWGN noise level using expectation
    maximization (EM)

    >>> output_channel_params['noise_level_estimation'] = 'em'
    >>> config['output_channel_parameters'] = output_channel_params
    >>> alpha_hat = run(y, A, A_asq)
    >>> alpha_hat[:k + 2]
    array([[ 1.94118089],
           [ 0.71553983],
           [-3.40076165],
           [-1.35662005],
           [ 1.1099417 ],
           [-0.00688125],
           [ 0.79442879],
           [ 0.06258856],
           [-1.08792606],
           [-0.00148811],
           [ 0.00022266],
           [-0.00003785]])
    >>> np.sum(np.abs(alpha - alpha_hat) > sigma * 3)
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

        if isinstance(A_asq, _MatrixBase):
            # It is not possible to validate the range of an implicit matrix
            # Thus allow all possible values
            range_ = '[-inf;inf]'
        else:
            range_ = '[0;inf)'
        _numeric('A_asq', ('integer', 'floating', 'complex'), range_=range_,
                 shape=A.shape, ignore_none=True)

    @_decorate_validation
    def validate_output():
        # complex128 is two float64 (real and imaginary part) each taking 8*8
        # bits. Thus, in total 2*8*8=128 bits. However, we only consider it to
        # be "64 bit precision" since that is what each part is.
        bits_pr_nbytes = 4 if np.iscomplexobj(convert(0)) else 8
        _numeric('alpha', ('integer', 'floating', 'complex'),
                 shape=(A.shape[1], 1),
                 precision=convert(0).nbytes * bits_pr_nbytes)
        _generic('history', 'mapping',
                 keys_in=('alpha_bar', 'alpha_tilde', 'MSE',
                          'input_channel_parameters',
                          'output_channel_parameters', 'stop_criterion',
                          'stop_criterion_value', 'stop_iteration',
                          'stop_reason'))

    validate_input()

    # Initialisation
    init = _get_gamp_initialisation(y, A, A_asq)
    AH = init['AH']
    A_asq = init['A_asq']
    AT_asq = init['AT_asq']
    o = init['o']
    q = init['q']
    s = init['s']
    r = init['r']
    m = init['m']
    n = init['n']
    alpha_bar = init['alpha_bar']
    alpha_tilde = init['alpha_tilde']
    alpha_breve = init['alpha_breve']
    A_dot_alpha_bar = init['A_dot_alpha_bar']
    z_bar = init['z_bar']
    damping = init['damping']
    sum_approximation_method = init['sum_approximation_method']
    A_frob_sq = init['A_frob_sq']
    output_channel = init['output_channel']
    output_channel_parameters = init['output_channel_parameters']
    input_channel = init['input_channel']
    input_channel_parameters = init['input_channel_parameters']
    stop_criterion = init['stop_criterion']
    stop_criterion_name = init['stop_criterion_name']
    iterations = init['iterations']
    tolerance = init['tolerance']
    convert = init['convert']
    report_history = init['report_history']
    history = init['history']
    true_solution = init['true_solution']

    # GAMP iterations
    for it in range(iterations):
        # Save previous state
        alpha_bar_prev = alpha_bar  # Used in stop criterion

        # GAMP state updates
        if sum_approximation_method == 'rangan':
            # Rangan's scalar variance sum approximation.

            # Factor side updates
            v = 1.0 / m * A_frob_sq * alpha_breve
            o = A_dot_alpha_bar - v * q
            z_bar, z_tilde = output_channel.compute(locals())
            q = (z_bar - o) / v
            u = 1.0 / m * np.sum((v - z_tilde) / v**2)

            # Variable side updates (including damping)
            s_full = 1 / (1.0 / n * A_frob_sq * u)
            s = (1 - damping) * s_full + damping * s
            r = alpha_bar + s * AH.dot(q)
            alpha_bar_full, alpha_tilde = input_channel.compute(locals())
            alpha_bar = (1 - damping) * alpha_bar_full + damping * alpha_bar
            alpha_breve = 1.0 / n * np.sum(alpha_tilde)

        else:
            # Either "full" GAMP or Krzakala's sum approximation.
            # For "full" GAMP, A_asq is the entrywise absolute value squared
            # matrix.
            # For Krzakala's sum approximation, A_asq is a
            # magni.utils.matrices.SumApproximationMatrix that implements the
            # scaled sum approximation to the full A_asq matrix.

            # Factor side updates
            v = A_asq.dot(alpha_tilde)
            o = A_dot_alpha_bar - v * q
            z_bar, z_tilde = output_channel.compute(locals())
            q = (z_bar - o) / v
            u = (v - z_tilde) / v**2

            # Variable side updates (including damping)
            s_full = 1 / (AT_asq.dot(u))
            s = (1 - damping) * s_full + damping * s
            r = alpha_bar + s * AH.dot(q)
            alpha_bar_full, alpha_tilde = input_channel.compute(locals())
            alpha_bar = (1 - damping) * alpha_bar_full + damping * alpha_bar

        # Stop criterion
        A_dot_alpha_bar = A.dot(alpha_bar)  # Used in residual stop criteria
        stop, stop_criterion_value = stop_criterion.compute(locals())

        # History reporting
        if report_history:
            history['alpha_bar'].append(alpha_bar)
            history['alpha_tilde'].append(alpha_tilde)
            history['input_channel_parameters'].append(
                copy.deepcopy(input_channel.__dict__))
            history['output_channel_parameters'].append(
                copy.deepcopy(output_channel.__dict__))
            history['stop_criterion_value'].append(stop_criterion_value)
            history['stop_iteration'] = it
            if true_solution is not None:
                history['MSE'].append(
                    1/n * np.linalg.norm(true_solution - alpha_bar)**2)

        if stop:
            history['stop_reason'] = stop_criterion_name.upper()
            break

    alpha = alpha_bar

    validate_output()

    if report_history:
        return alpha, history
    else:
        return alpha


def _get_gamp_initialisation(y, A, A_asq):
    """
    Return an initialisation of the GAMP algorithm.

    Parameters
    ----------
    y : ndarray
        The m x 1 measurement vector.
    A : ndarray or magni.utils.matrices.{Matrix, MatrixCollection}
        The m x n matrix which is the product of the measurement matrix and the
        dictionary matrix.
    A_asq : ndarray or magni.utils.matrices.{Matrix, MatrixCollection} or None
        The m x n matrix which is the entrywise absolute value squared product
        of the measurement matrix and the dictionary matrix (the default is
        None, which implies that a sum approximation is used).

    Returns
    -------
    init : dict
        The initialisation of the the GAMP algorithm.

    """

    init = dict()
    param = dict(_conf.items())

    # Configured setup
    init['convert'] = param['precision_float']
    init['damping'] = init['convert'](param['damping'])
    init['report_history'] = param['report_history']
    init['tolerance'] = param['tolerance']
    init['iterations'] = param['iterations']
    init['true_solution'] = param['true_solution']
    init['output_channel_parameters'] = param['output_channel_parameters']
    init['input_channel_parameters'] = param['input_channel_parameters']
    init['stop_criterion_name'] = param['stop_criterion'].__name__

    # Arguments based configuration
    init['AH'] = A.conj().T
    init['m'], init['n'] = A.shape

    init['A_frob_sq'] = None
    init['sum_approximation_method'] = None

    if A_asq is None:
        init['sum_approximation_method'], sum_approx_const = tuple(
            param['sum_approximation_constant'].items())[0]

        # Krzakala's sum approx
        init['A_asq'] = _SumApproximationMatrix(sum_approx_const)
        init['AT_asq'] = _SumApproximationMatrix(sum_approx_const)

        # Rangan's sum approx
        if init['sum_approximation_method'] == 'rangan':
            # _norm generally forms A explicitly to compute the Frobenius norm
            # which may be infeasible memory-wise
            init['A_frob_sq'] = _norm(A, ord='fro')**2

    else:
        # Full GAMP
        init['A_asq'] = A_asq
        init['AT_asq'] = A_asq.T

    # Channel and stop criterion configuration
    channel_init = copy.copy(init)
    channel_init['A'] = A
    channel_init['y'] = y
    init['input_channel'] = param['input_channel'](channel_init)
    init['output_channel'] = param['output_channel'](channel_init)
    init['stop_criterion'] = param['stop_criterion'](channel_init)

    # Configuration of remaining states
    if param['warm_start'] is not None:
        init['alpha_bar'] = init['convert'](param['warm_start'][0])
        init['alpha_tilde'] = init['convert'](param['warm_start'][1])
    else:
        init['alpha_bar'] = np.zeros((init['n'], 1), dtype=init['convert'])
        init['alpha_tilde'] = np.ones_like(init['alpha_bar'])

    init['alpha_breve'] = 1.0 / init['n'] * np.sum(init['alpha_tilde'])
    init['history'] = {'alpha_bar': [init['alpha_bar']],
                       'alpha_tilde': [init['alpha_tilde']],
                       'MSE': [np.nan],
                       'input_channel_parameters': [
                           init['input_channel_parameters']],
                       'output_channel_parameters': [
                           init['output_channel_parameters']],
                       'stop_criterion': init['stop_criterion_name'].upper(),
                       'stop_criterion_value': [np.nan],
                       'stop_iteration': 0,
                       'stop_reason': 'MAX_ITERATIONS'}
    init['A_dot_alpha_bar'] = A.dot(init['alpha_bar'])
    init['q'] = np.zeros_like(y)
    init['o'] = y.copy()
    init['s'] = np.ones_like(init['alpha_bar'])
    init['r'] = np.zeros_like(init['alpha_bar'])
    init['z_bar'] = y

    # Report on A_asq setup
    if param['report_A_asq_setup']:
        np_printoptions = np.get_printoptions()
        np.set_printoptions(precision=3, threshold=5, edgeitems=1)
        print('GAMP is using the A_asq: {}'.format(init['A_asq']))
        print('The sum approximation method is: {}'.format(
            init['sum_approximation_method']))
        np.set_printoptions(**np_printoptions)

    return init
