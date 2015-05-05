"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions for calculating the starting value of sigma used in
the Smoothed l0 algorithms.

Routine listings
----------------
calculate_using_fixed(var)
    Calculate the fixed sigma value.
calculate_using_reciprocal(var)
    Calculate a sigma value in a 'reciprocal' way.
get_function_handle(method)
    Return a function handle to a given calculation method.

"""

from __future__ import division


def wrap_calculate_using_fixed(var):
    """
    Arguments wrapper for calculate_using_fixed.

    """

    convert = var['convert']
    sigma_start = convert(var['param']['sigma_start_fixed'])

    def calculate_using_fixed():
        """
        Calculate the fixed sigma value.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the sigma
            value.

        Returns
        -------
        sigma : float
            The sigma value to be used in the SL0 algorithm.

        """

        return sigma_start

    return calculate_using_fixed


def wrap_calculate_using_reciprocal(var):
    """
    Arguments wrapper for calculate_using_reciprocal.

    """

    convert = var['convert']
    delta = var['A'].shape[0] / var['A'].shape[1]
    factor = convert(var['param']['sigma_start_reciprocal'])

    def calculate_using_reciprocal():
        """
        Calculate a sigma value in a 'reciprocal' way.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the sigma value.

        Returns
        -------
        sigma : float
            The sigma value to be used in the SL0 algorithm.

        """

        return 1 / (factor * delta)

    return calculate_using_reciprocal


def get_function_handle(method, var):
    """
    Return a function handle to a given calculation method.

    Parameters
    ----------
    method : str
        Identifier of the calculation method to return a handle to.
    var : dict
        Local variables needed in the sigma method.

    Returns
    -------
    f_handle : function
        Handle to the calculation method defined in this globals scope.

    """

    return globals()['wrap_calculate_using_' + method](var)
