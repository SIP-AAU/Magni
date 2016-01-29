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
    The determined phase transition is stored in the same HDF5 database, in the
    same pytables group in an array named 'phase_transition'.

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
    10 to the power of -4.

    """

    with _File(path, 'a') as f:
        if not '/' + label + '/phase_transition' in f:
            points = len(_conf['rho']) * _conf['monte_carlo']

            z = np.zeros(points)
            y = np.zeros(points)

            rho = np.zeros(len(_conf['delta']))

            dist = f.get_node('/' + label + '/dist')[:]

            for i in range(len(_conf['delta'])):
                n = np.round(_conf['delta'][i] * _conf['problem_size'])

                for j in range(len(_conf['rho'])):
                    if n > 0:
                        var = _conf['rho'][j]
                        var = np.round(var * n) / n
                    else:
                        var = 0.

                    for m in range(_conf['monte_carlo']):
                        z[j * _conf['monte_carlo'] + m] = var
                        y[j * _conf['monte_carlo'] + m] = dist[i, j, m] < 1e-4

                rho[i] = _estimate_PT(z, y)

            f.create_array('/' + label, 'phase_transition', rho)


def _estimate_PT(rho, success):
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
        return 0
    elif success.sum() > points - 0.5:
        # if all of the simulations were successful
        return 1

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
            val = -b[0] / b[1]
            val = max(val, 0)
            val = min(val, 1)

            return val

    raise RuntimeWarning('analysis.py: phase transition does not converge.')
