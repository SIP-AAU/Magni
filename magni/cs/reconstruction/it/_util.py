"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing miscellaneous utility functions for the different
implementations of Iterative Thresholding (IT).

Routine listings
----------------
get_methods(module)
    Extract relevant methods from module.
def _get_operators(module)
    Extract relevant operators from `module`.

"""

from __future__ import division

from magni.cs.reconstruction.it import _step_size
from magni.cs.reconstruction.it import _threshold
from magni.cs.reconstruction.it import _threshold_operators


def _get_methods(module):
    """
    Extract relevant methods from `module`.

    Parameters
    ----------
    module : str
        The name of the module from which the methods should be extracted.

    Returns
    -------
    methods : tuple
        The names of the relevant methods from `module`.

    Notes
    -----
    Looks for functions with names starting with 'wrap_calculate_using\_' in
    the module: _"module".

    """

    method_candidates = eval('_' + module).__dict__.keys()
    methods = [candidate[21:] for candidate in method_candidates
               if candidate[:21] == 'wrap_calculate_using_']

    return tuple(methods)


def _get_operators(module):
    """
    Extract relevant operators from `module`.

    Parameters
    ----------
    module : str
        The name of the module from which the operators should be extracted.

    Returns
    -------
    methods : tuple
        The names of the relevant operators from `module`.

    Notes
    -----
    Looks for functions with names starting with 'threshold\_' in the
    module _"module".

    """

    operator_candidates = eval('_' + module).__dict__.keys()
    operators = [candidate[10:] for candidate in operator_candidates
                 if candidate[:10] == 'threshold_']

    return tuple(operators)
