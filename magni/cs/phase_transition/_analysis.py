"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality for analysing the simulation results.

Routine listings
----------------
run(path, label)
    Determine the phase transition from the simulation results.

See Also
--------
magni.cs.phase_transition.config : Configuration options.

Notes
-----
For a description of the concept of phase transition, see [1]_.

References
----------
.. [1] C. S. Oxvig, P. S. Pedersen, T. Arildsen, and T. Larsen, "Surpassing the
   Theoretical 1-norm Phase Transition in Compressive Sensing by Tuning the
   Smoothed l0 Algorithm", *in IEEE International Conference on Acoustics,
   Speech and Signal Processing (ICASSP)*, Vancouver, Canada, May 26-31, 2013,
   pp. 6019-6023.

"""

from __future__ import division

import numpy as np

from magni.cs.phase_transition import config as _conf
from magni.utils.multiprocessing import File as _File


def run(path, label):
    """
    Determine the phase transition from the simulation results.

    The simulation results should be present in the HDF5 database specified by
    `path` in the pytables group specified by `label` in an array named 'dist'.
    The determined phase transition (50% curve) is stored in the same HDF5
    database, in the same HDF group in an array named 'phase_transition'.
    Additionally, the 10%, 25%, 75%, and 90% percentiles are stored in an array
    named 'phase_transition_percentiles'.

    Parameters
    ----------
    path : str
        The path of the HDF5 database.
    label : str
        The path of the pytables group in the HDF5 database.

    See Also
    --------
    _estimate_PT : The actual phase transition estimation.

    Notes
    -----
    A simulation is considered successful if the simulation result is less than
    a normalised mean squared error tolerance computed as 10^(-SNR/10) wtih SNR
    configured in the configuration module.

    """

    percentiles = [0.5, 0.1, 0.25, 0.75, 0.9]

    with _File(path, 'a') as f:
        if not '/' + label + '/phase_transition' in f:
            dist = f.get_node('/' + label + '/dist')[:]
            # Set NaNs to value > NMSE_tolerance, i.e. assume failure
            dist[np.isnan(dist)] = 10**(-_conf['SNR'] / 10) * 2

            if _conf['logit_solver'] == 'built-in':
                # Use "simple" built-in solver
                rho = _built_in_logit_solver(dist, percentiles)
            elif _conf['logit_solver'] == 'sklearn':
                # Use scikit learn solver
                rho = _sklearn_logit_solver(dist, percentiles)

            f.create_array('/' + label, 'phase_transition', rho[0])
            f.create_array('/' + label, 'phase_transition_percentiles',
                           rho[1:])


def _built_in_logit_solver(dist, percentiles):
    """
    Fit a logistic regression model using the built-in solver.

    Parameters
    ----------
    dist : ndarray
        The simulated signal "distances" in the phase space.
    percentiles : list or tuple
        The percentiles to estimate.

    Returns
    -------
    rho : ndarray
        The "len(percentiles)"-by-"len(delta)" array of estimated phase
        transition rho vectors. The phase transition rho vectors are
        (in order): 50% (the phase transition esitmate), smaller to larger
        percentiles.

    """

    monte_carlo = dist.shape[2]
    points = len(_conf['rho']) * monte_carlo
    NMSE_tolerance = 10**(-_conf['SNR'] / 10)

    z = np.zeros(points)
    y = np.zeros(points)

    rho = np.zeros((len(percentiles), len(_conf['delta'])))

    for i in range(len(_conf['delta'])):
        n = np.round(_conf['delta'][i] * _conf['problem_size'])

        for j in range(len(_conf['rho'])):
            if n > 0:
                var = _conf['rho'][j]
                var = np.round(var * n) / n
            else:
                var = 0.

            for m in range(monte_carlo):
                z[j * monte_carlo + m] = var
                y[j * monte_carlo + m] = dist[i, j, m] < NMSE_tolerance

        rho[:, i] = _estimate_PT(z, y, percentiles)

    return rho


def _estimate_PT(rho, success, percentiles):
    """
    Estimate the phase transition location for a given delta.

    The phase transition location is estimated using logistic regression. The
    algorithm used for this is Newton's method.

    Parameters
    ----------
    rho : ndarray
        The rho values.
    success : ndarray
        The success indicators.
    percentiles : list or tuple
        The percentiles to estimate.

    Returns
    -------
    rho : float
        The estimated phase transition location.

    Notes
    -----
    The function includes a number of non-standard ways of handling numerical
    and convergence related issues. This will be changed in a future version of
    the code.

    """

    points = len(success)

    if success.sum() < 0.5:
        # if none of the simulations were successful
        return [0] * len(percentiles)
    elif success.sum() > points - 0.5:
        # if all of the simulations were successful
        return [1] * len(percentiles)

    y = np.zeros((points + 2, 1))
    y[:points, 0], y[points:, 0] = success, [1, 0]

    # note: z_i in the algorithm is z[i, :]^T in the implementation
    z = np.ones((points + 2, 2))
    z[:points, 1], z[points:, 1] = rho, [0, 1]

    b = np.zeros((2, 1))

    for l in range(100):
        # p = [p_i]^T
        # p_i = exp(b^T z_i) / (1 + exp(b^T z_i))
        # p_i = 1 / (1 + exp(-z_i^T b))
        p = 1 / (1 + np.exp(-z.dot(b)))

        # gradient
        # g = sum([(y_i - p_i) * z_i])
        g = z.T.dot(y - p)

        # Hessian
        # H = -sum([p_i * (1 - p_i) * z_i z_i^T])
        H = -(p * (1 - p) * z).T.dot(z)

        try:
            # step
            s = np.linalg.inv(H).dot(g)
        except np.linalg.LinAlgError:
            # it results in convergence but is hardly a standard solution
            det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
            det = np.sign(np.sign(det) + 0.5) * 1e-100

            Hinv = np.float64([[H[1, 1], -H[0, 1]], [-H[1, 0], H[0, 0]]]) / det
            s = Hinv.dot(g)

        b = b - s
        # constrained to non-positive since the model would otherwise suggest
        # a better chance of reconstruction for higher rho
        b[1] = min(b[1], 0)

        g_len = np.linalg.norm(g)
        s_len = np.linalg.norm(s)
        b_len = np.linalg.norm(b)

        # g_len < 1e-12 : convergence
        # s_len < 1e-3 : convergence
        # s_len / b_len < 1e-12 : ... in case of large coefficients
        # the last two should probably be replaced by
        #     delta (-b[0] / b[1]) < 1e-12
        # or something like that
        if g_len < 1e-12 or s_len < 1e-3 or s_len / b_len < 1e-12:
            val = np.zeros(len(percentiles))
            for l, p in enumerate(percentiles):
                val[l] = (np.log(p / (1 - p)) - b[0]) / b[1]
                val[l] = max(val[l], 0)
                val[l] = min(val[l], 1)

            return val

    raise RuntimeWarning('analysis.py: phase transition does not converge.')


def _sklearn_logit_solver(dist, percentiles):
    """
    Fit a logistic regression model using the solver from scikit-learn.

    Parameters
    ----------
    dist : ndarray
        The simulated signal "distances" in the phase space.
    percentiles : list or tuple
        The percentiles to estimate.

    Returns
    -------
    rho : ndarray
        The "len(percentiles)"-by-"len(delta)" array of estimated phase
        transition rho vectors. The phase transition rho vectors are
        (in order): 50% (the phase transition esitmate), smaller to larger
        percentiles.

    """

    from sklearn.linear_model import LogisticRegression
    NMSE_tolerance = 10**(-_conf['SNR'] / 10)

    lr = LogisticRegression(C=1e3, fit_intercept=False,
                            random_state=_conf['seed'])
    rho = np.zeros((len(percentiles), len(_conf['delta'])))
    monte_carlo = dist.shape[2]

    for k in range(len(_conf['delta'])):
        successes = (dist[k, :, :] < NMSE_tolerance).reshape(-1)

        if not np.any(successes):
            rho[:, k] = 0.0
        elif np.all(successes):
            rho[:, k] = 1.0
        else:
            X = np.column_stack(
                (np.ones_like(successes),
                 np.repeat(_conf['rho'], monte_carlo)))
            lr.fit(X, successes)
            b = [lr.coef_[0, 0], lr.coef_[0, 1]]

            for l, p in enumerate(percentiles):
                rho[l, k] = (np.log(p / (1 - p)) - b[0]) / b[1]
                rho[l, k] = max(rho[l, k], 0.0)
                rho[l, k] = min(rho[l, k], 1.0)

    return rho
