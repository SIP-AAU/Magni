"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions for calculating a threshold (level) used in
Iterative Threshold algorithms.

Routine listings
----------------
calculate_far(delta)
    Calculate the optimal False Acceptance Rate for a given indeterminacy.
calculate_using_far(var)
    Calculate a threshold level using the FAR heuristic.
calculate_using_fixed(var)
    Calculate a threshold level using a given fixed support size.
get_function_handle(method)
    Return a function handle to a given calculation method.

"""

from __future__ import division

import numpy as np
import scipy.stats
try:
    import bottleneck as bn
    calculate_median = bn.median
    partsort = bn.partsort
except ImportError:
    calculate_median = np.median
    partsort = np.sort


def calculate_far(delta, it_algorithm):
    """
    Calculate the optimal False Acceptance Rate for a given indeterminacy.

    Parameters
    ----------
    delta : float
        The indeterminacy, m / n, of a system of equations of size m x n.
    it_algorithm : {IHT, ITS}
        The iterative thresholding algorithm to calculate the FAR for.

    Returns
    -------
    FAR : float
        The optimal False Acceptance Rate for the given indeterminacy.

    Notes
    -----
    The optimal False Acceptance Rate to be used in connection with the
    interference heuristic presented in the paper "Optimally Tuned Iterative
    Reconstruction Algorithms for Compressed Sensing" [1]_ is calculated from
    a set of optimal values presented in the same paper. The calculated value
    is found from a linear interpolation or extrapolation on the known set of
    optimal values.

    References
    ----------
    .. [1] A. Maleki and D.L. Donoho, "Optimally Tuned Iterative Reconstruction
       Algorithms for Compressed Sensing", *IEEE Journal Selected Topics in
       Signal Processing*, vol. 3, no. 2, pp. 330-341, Apr. 2010.

    """

    # Known optimal values (x - indeterminacy / y - FAR)
    x = [0.05, 0.11, 0.21, 0.41, 0.50, 0.60, 0.70, 0.80, 0.93]
    if it_algorithm == 'IHT':
        y = [0.0015, 0.002, 0.004, 0.011, 0.015, 0.02, 0.027, 0.035, 0.043]
    else:
        y = [0.02, 0.037, 0.07, 0.12, 0.16, 0.2, 0.25, 0.32, 0.37, 0.42]

    i = next((i for i in range(len(x) - 1) if delta <= x[i + 1]), len(x) - 2)

    FAR = y[i] + (delta - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i])

    return FAR


def wrap_calculate_using_far(var):
    """
    Arguments wrapper for calculate_using_far.

    Calculate a threshold level using the FAR heuristic.

    """

    if 'hard' in var['threshold_alpha'].__name__:
        it_algorithm = 'IHT'
    else:
        it_algorithm = 'IST'

    far = calculate_far(var['A'].shape[0] / var['A'].shape[1],
                        it_algorithm)

    convert = var['convert']
    Lambda = convert(scipy.stats.norm.ppf(1 - far / 2))
    stdQ1 = convert(scipy.stats.norm.ppf(1 - 0.25))

    def calculate_using_far(var):
        """
        Calculate a threshold level using the FAR heuristic.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the threshold.

        Returns
        -------
        thres : float
            The threshold to be used in the Iterative Thresholding algorithm.

        Notes
        -----
        The threhold is calculated using a False Acceptance Ratio (FAR)
        heuristic as described in [1]_.

        References
        ----------
        .. [1] A. Maleki and D.L. Donoho, "Optimally Tuned Iterative
        Reconstruction Algorithms for Compressed Sensing", *IEEE Journal
        Selected Topics in Signal Processing*, vol. 3, no. 2, pp. 330-341,
        Apr. 2010.

        """

        c_median = calculate_median(np.abs(var['c'].ravel()))
        thres = var['kappa'] * Lambda * convert(c_median) / stdQ1

        return thres

    return calculate_using_far


def wrap_calculate_using_fixed(var):
    """
    Arguments wrapper for calculate_using_fixed.

    Calculate a threshold level using a given fixed support size.

    """

    k = var['param']['threshold_fixed']
    threshold_weights = var['threshold_weights']

    if partsort.__name__ == 'sort':
        # Fallback numpy sort
        def find_k_largest(coefs, k):
            return np.sort(coefs)[::-1][k]
    else:
        # Bottleneck partsort
        def find_k_largest(coefs, k):
            return partsort(coefs, len(coefs) - k)[len(coefs) - k - 1]

    def calculate_using_fixed(var):
        """
        Calculate a threshold level using a given fixed support size.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the threshold.

        Returns
        -------
        thres : float
            The threshold to be used in the Iterative Threshold algorithm.

        Notes
        -----
        The threshold is calculated using of a fixed support size i.e., by
        specifying the number of non-zero coefficients, k.

        """

        abs_coefficients = np.abs((var['alpha'] * threshold_weights).ravel())
        thres = find_k_largest(abs_coefficients, k)

        return thres

    return calculate_using_fixed


def get_function_handle(method, var):
    """
    Return a function handle to a given calculation method.

    Parameters
    ----------
    method : str
        Identifier of the calculation method to return a handle to.
    var : dict
        Local variables needed in the threshold method.

    Returns
    -------
    f_handle : function
        Handle to calculation `method` defined in this globals scope.

    """

    return globals()['wrap_calculate_using_' + method](var)
