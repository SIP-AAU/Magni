"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing a robust configger class.

Routine listings
----------------
Configger()
    Provide set and get functions to access a set of configuration options.

Notes
-----
This module does not itself contain any configuration options and thus has no
get or set functions unlike the other config modules of `magni`.

"""

from __future__ import division
from itertools import chain

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate as _validate


class Configger():
    """
    Provide set and get functions to access a set of configuration options.

    The set of configuration options, their default values, and their
    validation schemes are specified upon initialisation.

    Parameters
    ----------
    param : dict
        The configuration options along with their default values.
    requirements : dict
        The validation schemes of the configuration options.

    See Also
    --------
    magni.utils.validation : Validation.

    Notes
    -----
    `requirements` must contain the same keys as `param`. For each key in
    `requirements`, the value is used as validation scheme - see `set` for
    further information.

    Examples
    --------
    Instantiate Configger with the parameter 'key' with default value 'default'
    which can only assume string values.

    >>> from magni.utils.config import Configger
    >>> config = Configger({'key': 'default'}, {'key': {'type': str}})

    The parameter can either be retrieved by getting a copy of the entire
    parameter dictionary or by getting the specific key.

    >>> config.get()
    {'key': 'default'}
    >>> config.get('key')
    'default'

    Likewise the parameter can either be changed by passing a dictionary with
    the parameter or by using a keyword argument.

    >>> config.set({'key': 'value'})
    >>> config.set(key='value')
    >>> config.get('key')
    'value'

    """

    @_decorate_validation
    def _validate_init(self, param, requirements):
        """
        Validate the `__init__` function.

        See Also
        --------
        Configger.__init__ : The validated function.
        magni.utils.validation.validate : Validation.

        """

        _validate(param, 'param', {'type': dict})
        _validate(requirements, 'requirements',
                  [{'type': dict}, {'type_in': [list, tuple, dict]}])

        if not sorted(param.keys()) == sorted(requirements.keys()):
            raise KeyError('param and requirements must have the same keys.')

    def __init__(self, param, requirements):
        self._validate_init(param, requirements)

        self._param = param
        self._requirements = requirements

    @_decorate_validation
    def _validate_get(self, key):
        """
        Validate the `get` function.

        See Also
        --------
        Configger.get : The validated function.
        magni.utils.validation.validate : Validation.

        """

        _validate(key, 'key', {'val_in': list(self._param.keys())}, True)

    def get(self, key=None):
        """
        Retrieve a copy of all parameters or a specific parameter.

        Parameters
        ----------
        key : str or None, optional
            The name of the key to retrieve (the default is None, which implies
            retrieving a copy of all parameters)

        Returns
        -------
        value : dict or None
            The value of the specified key, if a key is not None. Otherwise,
            a copy of the parameter dictionary.

        """

        self._validate_get(key)

        if key is None:
            value = self._param.copy()
        else:
            value = self._param[key]

        return value

    @_decorate_validation
    def _validate_set(self, dictionary, kwargs):
        """
        Validate the `set` function.

        See Also
        --------
        Configger.set : The validated function.
        magni.utils.validation.validate : Validation.

        """

        keys = list(self._param.keys())
        _validate(dictionary, 'dictionary', {'type': dict, 'keys_in': keys})
        _validate(kwargs, 'kwargs', {'keys_in': keys})

        all_items = dict(chain(dictionary.items(), kwargs.items())).items()
        for key, value in all_items:
            _validate(value, key, self._requirements[key])

    def set(self, dictionary={}, **kwargs):
        """
        Overwrite the value of one or more parameters.

        Each value is validated according to the validation scheme of that
        parameter.

        Parameters
        ----------
        dictionary : dict, optional
            A dictionary containing the key and value pairs to update.
        kwargs : dict, optional
            Keyword arguments being the key and value pairs to update.

        See Also
        --------
        magni.utils.validation.validate : Validation.

        """

        self._validate_set(dictionary, kwargs)

        all_items = dict(chain(dictionary.items(), kwargs.items())).items()
        for key, value in all_items:
            self._param[key] = value
