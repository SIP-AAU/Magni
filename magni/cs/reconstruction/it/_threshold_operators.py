"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing thresholding operators used in Iterative Thresholding
algorithms.

Routine listings
----------------
get_function_handle(method)
    Return a function handle to a given threshold operator.
threshold_hard(var)
    The hard threshold operator.
threshold_none(var)
    The "no" threshold operator.
threshold_soft(var)
    The soft threshold operator.
threshold_weighted_hard(var)
    The weighted hard threshold operator.
threshold_weighted_soft(var)
    The weighted soft threshold operator.

"""

from __future__ import division

import numpy as np


def get_function_handle(method):
    """
    Return a function handle to a given threshold operator method.

    Parameters
    ----------
    method : str
        Identifier of the threshold operator to return a handle to.

    Returns
    -------
    f_handle : function
        Handle to threshold method defined in this globals scope.

    """

    return globals()['threshold_' + method]


def threshold_hard(var):
    """
    Threshold the entries of a vector using the hard threshold.

    Parameters
    ----------
    var : dict
        Local variables used in the threshold operation.

    Notes
    -----
    This threshold operation works "in-line" on the variables in `var`. Hence,
    this function does not return anything.

    Examples
    --------
    For example, thresholding a vector of values between -1 and 1

    >>> import copy, numpy as np, magni
    >>> from magni.cs.reconstruction.it._threshold_operators import (
    ... threshold_hard)
    >>> var = {'alpha': np.linspace(-1, 1, 10), 'threshold': 0.4}
    >>> threshold_hard(copy.copy(var))
    >>> var['alpha']
    array([-1.        , -0.77777778, -0.55555556,  0.        ,  0.        ,
            0.        ,  0.        ,  0.55555556,  0.77777778,  1.        ])

    """

    var['alpha'][np.abs(var['alpha']) < var['threshold']] = 0


def threshold_none(var):
    """
    Do not threshold the entries of a vector.

    Parameters
    ----------
    var : dict
        Local variables used in the threshold operation.

    Notes
    -----
    This is a dummy threshold operation that does nothing.

    """

    return


def threshold_soft(var):
    """
    Threshold the entries of a vector using the soft threshold.

    Parameters
    ----------
    var : dict
        Local variables used in the threshold operation.

    Notes
    -----
    This threshold operation works "in-line" on the variables in `var`. Hence,
    this function does not return anything.

    Examples
    --------
    For example, thresholding a vector of values between -1 and 1

    >>> import copy, numpy as np, magni
    >>> from magni.cs.reconstruction.it._threshold_operators import (
    ... threshold_soft)
    >>> var = {'alpha': np.linspace(-1, 1, 10), 'threshold': 0.4}
    >>> threshold_soft(copy.copy(var))
    >>> var['alpha']
    array([-0.6       , -0.37777778, -0.15555556,  0.        ,  0.        ,
            0.        ,  0.        ,  0.15555556,  0.37777778,  0.6       ])

    """

    x = var['alpha']
    thres = var['threshold']
    alpha_thres = ((x - thres) * (x > thres) + (x + thres) * (x < -thres))
    var['alpha'][:] = alpha_thres[:]


def threshold_weighted_hard(var):
    """
    Threshold the entries of a vector using a weighted hard threshold.

    Parameters
    ----------
    var : dict
        Local variables used in the threshold operation.

    Notes
    -----
    This threshold operation works "in-line" on the variables in `var`. Hence,
    this function does not return anything.

    Examples
    --------
    For example, thresholding a vector of values between -1 and 1

    >>> import copy, numpy as np, magni
    >>> from magni.cs.reconstruction.it._threshold_operators import (
    ... threshold_weighted_hard)
    >>> var = {'alpha': np.linspace(-1, 1, 10), 'threshold': 0.4,
    ... 'threshold_weights': 0.7 * np.ones(10)}
    >>> threshold_weighted_hard(copy.copy(var))
    >>> var['alpha']
    array([-1.        , -0.77777778,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.77777778,  1.        ])

    """

    mod_var = var.copy()  # Shallow copy of variable dictionary
    mod_var['alpha'] = var['alpha'].copy()  # Copy of alphas coefficents array
    mod_var['alpha'] *= var['threshold_weights']
    threshold_hard(mod_var)
    var['alpha'][mod_var['alpha'] == 0] = 0


def threshold_weighted_soft(var):
    """
    Threshold the entries of a vector using a weighted soft threshold.

    Parameters
    ----------
    var : dict
        Local variables used in the threshold operation.

    Notes
    -----
    This threshold operation works "in-line" on the variables in `var`. Hence,
    this function does not return anything.

    Examples
    --------
    For example, thresholding a vector of values between -1 and 1

    >>> import copy, numpy as np, magni
    >>> from magni.cs.reconstruction.it._threshold_operators import (
    ... threshold_weighted_soft)
    >>> var = {'alpha': np.linspace(-1, 1, 10), 'threshold': 0.4,
    ... 'threshold_weights': 0.7 * np.ones(10)}
    >>> threshold_weighted_soft(copy.copy(var))
    >>> var['alpha']
    array([-0.42857143, -0.20634921,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.20634921,  0.42857143])

    """

    mod_var = var.copy()  # Shallow copy of variable dictionary
    mod_var['alpha'] = var['alpha'].copy()  # Copy of alphas coefficents array
    mod_var['alpha'] *= var['threshold_weights']
    threshold_soft(mod_var)
    mod_var['alpha'] *= 1/var['threshold_weights']
    var['alpha'][:] = mod_var['alpha'][:]
