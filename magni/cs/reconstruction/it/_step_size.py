"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions for calculating the step-size (relaxation parameter)
used in the Iterative Thresholding algorithms.

Routine listings
----------------
calculate_using_adaptive(var)
    Calculate a step-size value in an 'adaptive' way.
calculate_using_fixed(var)
    Calculate the fixed step-size value.
get_function_handle(method)
    Return a function handle to a given calculation method.

"""

from __future__ import division

import numpy as np

from magni.utils.matrices import Matrix as _Matrix
from magni.utils.matrices import MatrixCollection as _MatrixC


def wrap_calculate_using_adaptive(var):
    """
    Arguments wrapper for culculate_using_adaptive.

    Calculate a step-size value in an 'adaptive' way.

    """

    convert = var['convert']
    A = var['A']

    def calculate_using_adaptive(var):
        """
        Calculate a step-size value in an 'adaptive' way.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the step-size.

        Returns
        -------
        kappa : float
            The step-size to be used in the IHT algorithm.

        Notes
        -----
        The 'adaptive' step-size selection from [1]_ is used.

        .. warning::

            This does not implement the Normalized IHT algorithm. It merely
            uses the adaptive choice of kappa for a 'correctly identified'
            support.

        References
        ----------
        .. [1] T. Blumensath and M.E. Davies, "Normalized Iterative Hard
           Thresholding: Guaranteed Stability and Performance", *IEEE Journal
           Selected Topics in Signal Processing*, vol. 4, no. 2, pp. 298-309,
           Apr. 2010.

        """

        support = var['alpha'] != 0

        if np.alltrue(~support):
            # For an empty support set, try with kappa=1
            g = np.ones(1)
            G = g

        elif isinstance(A, _Matrix) or isinstance(A, _MatrixC):
            g = var['c'].copy()
            g[~support] = 0
            G = A

        else:
            g = var['c'][support]
            G = A[:, support.ravel()]

        kappa = convert(np.linalg.norm(g)**2 / np.linalg.norm(G.dot(g))**2)

        return kappa

    return calculate_using_adaptive


def wrap_calculate_using_fixed(var):
    """
    Arguments wrapper for calculate_using_fixed.

    Calculate a fixed step-size value.

    """

    kappa = var['kappa']

    def calculate_using_fixed(var):
        """
        Calculate a fixed step-size value.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in the calculation of the step-size.

        Returns
        -------
        kappa : float
            The step-size to be used in the IHT algorithm.

        """

        return kappa

    return calculate_using_fixed


def get_function_handle(method, var):
    """
    Return a function handle to a given calculation method.

    Parameters
    ----------
    method : str
        Identifier of the calculation method to return a handle to.
    var : dict
        Local variables needed in the step-size method.

    Returns
    -------
    f_handle : function
        Handle to calculation `method` defined in this globals scope.

    """

    return globals()['wrap_calculate_using_' + method](var)
