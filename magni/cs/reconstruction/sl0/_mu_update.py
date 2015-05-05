"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions for calculating the updated value of mu (the
relative step-size used in the gradient descent iteration)  used in the
Smoothed l0 algorithms.

Routine listings
----------------
calculate_using_fixed(var)
    Calculate the updated, fixed mu value.
calculate_using_step(var)
    Calculate an updated mu value in a 'step' way.
get_function_handle(method)
    Return a function handle to a given calculation method.

"""

from __future__ import division


def wrap_calculate_using_fixed(var):
    """
    Arguments wrapper for calculate_using_fixed.

    """

    convert = var['convert']
    mu = convert(var['param']['mu_fixed'])

    def calculate_using_fixed(var):
        """
        Calculate the updated, fixed mu value.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the updated mu
            value.

        Returns
        -------
        mu : float
            The updated mu value to be used in the SL0 algorithm.

        """

        return mu

    return calculate_using_fixed


def wrap_calculate_using_step(var):
    """
    Arguments wrapper for calculate_using_step.

    """

    convert = var['convert']
    sigma_start = var['sigma_start']
    sigma_geometric = var['sigma_geometric']
    mu_start = convert(var['param']['mu_step_start'])
    mu_end = convert(var['param']['mu_step_end'])
    mu_iteration = var['param']['mu_step_iteration']

    def calculate_using_step(var):
        """
        Calculate an updated mu value in a 'step' way.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the updated mu
            value.

        Returns
        -------
        mu : float
            The updated mu value to be used in the SL0 algorithm.

        """

        i = var['i']

        if i < mu_iteration or sigma_start * sigma_geometric**i > 0.75:
            mu = mu_start
        else:
            mu = mu_end

        return mu

    return calculate_using_step


def get_function_handle(method, var):
    """
    Return a function handle to a given calculation method.

    Parameters
    ----------
    method : str
        Identifier of the calculation method to return a handle to.
    var : dict
        Local variables needed in the mu update method.

    Returns
    -------
    f_handle : function
        Handle to the calculation method defined in this globals scope.

    """

    return globals()['wrap_calculate_using_' + method](var)
