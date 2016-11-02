"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions for calculating stop criteria used in Iterative
Threholding (IT) algorithms.

Routine listings
----------------
calculate_using_mse_convergence(var)
    Calculate stop criterion based on mse convergence.
calculate_using_residual(var)
    Calculate stop criterion based on residual.
calculate_using_residual_measurments_ratio(var)
    Calculate stop criterion based on the residual to measurements ratio.
get_function_handle(method)
    Return a function handle to a given calculation method.

"""

from __future__ import division

import numpy as np


def wrap_calculate_using_mse_convergence(var):
    """
    Arguments wrapper for `calculate_using_mse_convergence`.

    Calculate stop criterion based on mse convergence.

    """

    n = var['A'].shape[1]
    tolerance = var['tolerance']

    def calculate_using_mse_convergence(var):
        """
        Calculate stop criterion based on mse convergence.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in calculating of the stop criterion.

        Returns
        -------
        stop : bool
            The indicator of whether or not the stop criterion is satisfied.
        mse : float
            The current convergence mean squared error.

        Notes
        -----
        The IT algorithm should converge to a fixed point. This criterion
        is based on the mean squared error of the difference between the
        proposed solution in this iteration and the proposed solution in the
        previous solution.

        """

        mse = 1/n * np.linalg.norm(var['alpha_prev'] - var['alpha'])**2
        stop = mse < tolerance

        return stop, mse

    return calculate_using_mse_convergence


def wrap_calculate_using_residual(var):
    """
    Arguments wrapper for `calculate_using_residual`.

    Calculate stop criterion based on residual.

    """

    tolerance = var['tolerance']
    m = var['A'].shape[0]

    def calculate_using_residual(var):
        """
        Calculate stop criterion based on residual.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in calculating of the stop criterion.

        Returns
        -------
        stop : bool
            The indicator of whether or not the stop criterion is satisfied.
        r_mse : float
            The current mean squared error of the residual.

        Notes
        -----
        This stopping criterion is based on the mean sqaured error of the
        residual.

        """

        r_mse = 1/m * np.linalg.norm(var['r'])**2
        stop = r_mse < tolerance

        return stop, r_mse

    return calculate_using_residual


def wrap_calculate_using_residual_measurements_ratio(var):
    """
    Arguments wrapper for `calculate_using_residual_measurements_ratio`.

    Calculate stop criterion based on the residual to measurements ratio.

    """

    tolerance = var['tolerance']
    y = var['y']

    def calculate_using_residual_measurements_ratio(var):
        """
        Calculate stop criterion based on the residual to measurements ratio.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in calculating of the stop criterion.

        Returns
        -------
        stop : bool
            The indicator of whether or not the stop criterion is satisfied.
        r_norm : float
            The current 2-norm of the residual.

        Notes
        -----
        This stop criterion is based on the ratio of the 2-norm of the current
        residual to the 2-norm of the measurments.

        """

        r_norm = np.linalg.norm(var['r'])
        stop = r_norm < tolerance * np.linalg.norm(y)

        return stop, r_norm

    return calculate_using_residual_measurements_ratio


def get_function_handle(method, var):
    """
    Return a function handle to a given calculation method.

    Parameters
    ----------
    method : str
        Identifier of the calculation method to return a handle to.
    var : dict
        Local variables needed in the calculation method.

    Returns
    -------
    f_handle : function
        Handle to calculation `method` defined in this globals scope.

    """

    return globals()['wrap_calculate_using_' + method](var)
