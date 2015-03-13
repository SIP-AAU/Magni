"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing a CS reconstruction algorithm adapted configger subclass.

Routine listings
----------------
Configger(magni.utils.config.Configger)
    Provide functionality to access a set of configuration options.

Notes
-----
This module does not itself contain any configuration options and thus has no
access to any configuration options unlike the other config modules of `magni`.

"""

import numpy as np

from magni.utils.config import Configger as _Configger
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic


class Configger(_Configger):
    """
    Provide functionality to access a set of configuration options.

    The present class redefines the methods for retrieving configuration
    parameters in order to ensure the desired precision of the floating point
    parameter values.

    Parameters
    ----------
    params : dict
        The configuration options and their default values.
    valids : dict
        The validation schemes of the configuration options.

    Attributes
    ----------
    property

    See Also
    --------
    magni.utils.config.Configger : Superclass of the present class.

    """

    def __init__(self, params, valids):
        @_decorate_validation
        def validate_input():
            _generic('params', 'mapping', has_keys=('precision_float',))

        _Configger.__init__(self, params, valids)
        validate_input()

    def __getitem__(self, name):
        """
        Get the value of a configuration parameter.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        value : None
            The value of the parameter.

        Notes
        -----
        If the value is a floating point value then that value is typecast to
        the desired precision.

        """

        value = _Configger.__getitem__(self, name)

        if isinstance(value, (float, np.floating)):
            value = self['precision_float'](value)

        return value
