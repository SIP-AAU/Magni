"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions for calculating the starting value of L (the number
of gradient descent iterations for each sigma) used in the Smoothed l0
algorithms.

Routine listings
----------------
calculate_using_fixed(var)
    Calculate the fixed L value.
calculate_using_geometric(var)
    Calculate an L value in a 'geometric' way.
get_function_handle(method)
    Return a function handle to a given calculation method.

"""

from __future__ import division


def wrap_calculate_using_fixed(var):
    """
    Arguments wrapper for calculate_using_fixed.

    """

    convert = var['convert']
    L = convert(var['param']['L_fixed'])

    def calculate_using_fixed():
        """
        Calculate the fixed L value.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the L value.

        Returns
        -------
        L : float
            The L value to be used in the SL0 algorithm.

        """

        return L

    return calculate_using_fixed


def wrap_calculate_using_geometric(var):
    """
    Arguments wrapper for calculate_using_geometric.

    """

    convert = var['convert']
    L_start = convert(var['param']['L_geometric_start'])

    def calculate_using_geometric():
        """
        Calculate an L value in a 'geometric' way.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the L value.

        Returns
        -------
        L : float
            The L value to be used in the SL0 algorithm.

        """

        return L_start

    return calculate_using_geometric


def get_function_handle(method, var):
    """
    Return a function handle to a given calculation method.

    Parameters
    ----------
    method : str
        Identifier of the calculation method to return a handle to.
    var : dict
        Local variables needed in the L method.

    Returns
    -------
    f_handle : function
        Handle to the calculation method defined in this globals scope.

    """

    return globals()['wrap_calculate_using_' + method](var)
