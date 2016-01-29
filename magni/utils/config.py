"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing a robust configger class.

Routine listings
----------------
Configger(object)
    Provide functionality to access a set of configuration options.

Notes
-----
This module does not itself contain any configuration options and thus has no
access to any configuration options unlike the other config modules of `magni`.

"""

from __future__ import division
from itertools import chain

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


class Configger(object):
    """
    Provide functionality to access a set of configuration options.

    The set of configuration options, their default values, and their
    validation schemes are specified upon initialisation.

    Parameters
    ----------
    params : dict
        The configuration options and their default values.
    valids : dict
        The validation schemes of the configuration options.

    See Also
    --------
    magni.utils.validation : Validation.

    Notes
    -----
    `valids` must contain the same keys as `params`. For each key in 'valids',
    the first value is the validation function ('generic', 'levels', or
    'numeric'), whereas the remaining values are passed to that validation
    function.

    Examples
    --------
    Instantiate Configger with the parameter 'key' with default value 'default'
    which can only assume string values.

    >>> import magni
    >>> from magni.utils.config import Configger
    >>> valid = magni.utils.validation.validate_generic(None, 'string')
    >>> config = Configger({'key': 'default'}, {'key': valid})

    The number of parameters can be retrieved as the length:

    >>> len(config)
    1

    That parameter can be retrieved in a number of ways:

    >>> config['key']
    'default'

    >>> for key, value in config.items():
    ...     print('key: {!r}, value: {!r}'.format(key, value))
    key: 'key', value: 'default'

    >>> for key in config.keys():
    ...     print('key: {!r}'.format(key))
    key: 'key'

    >>> for value in config.values():
    ...     print('value: {!r}'.format(value))
    value: 'default'

    Likewise, the parameter can be changed in a number of ways:

    >>> config['key'] = 'value'
    >>> config['key']
    'value'

    >>> config.update({'key': 'value changed by dict'})
    >>> config['key']
    'value changed by dict'

    >>> config.update(key='value changed by keyword')
    >>> config['key']
    'value changed by keyword'

    Finally, the parameter can be reset to the default value at any point:

    >>> config.reset()
    >>> config['key']
    'default'

    """

    _funcs = {'generic': _generic, 'levels': _levels, 'numeric': _numeric}

    def __init__(self, params, valids):
        @_decorate_validation
        def validate_input():
            _generic('params', 'mapping')
            _levels('valids', (
                _generic(None, 'mapping', has_keys=tuple(params.keys())),
                _generic(None, 'explicit collection')))

            for key in valids.keys():
                _generic(('valids', key, 0), 'string',
                         value_in=tuple(self._funcs.keys()))

        validate_input()

        self._default = params.copy()
        self._params = params
        self._valids = valids

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

        """

        @_decorate_validation
        def validate_input():
            _generic('name', 'string', value_in=tuple(self._params.keys()))

        validate_input()

        return self._params[name]

    def __len__(self):
        """
        Get the number of configuration parameters.

        Returns
        -------
        length : int
            The number of parameters.

        """

        return len(self._params)

    def __setitem__(self, name, value):
        """
        Set the value of a configuration parameter.

        The value is validated according to the validation scheme of that
        parameter.

        Parameters
        ----------
        name : str
            The name of the parameter.
        value : None
            The new value of the parameter.

        """

        @_decorate_validation
        def validate_input():
            _generic('name', 'string', value_in=tuple(self._params.keys()))
            validation = self._valids[name]
            self._funcs[validation[0]]('value', *validation[1:])

        validate_input()

        self._params[name] = value

    def get(self, key=None):
        """
        Deprecated method.

        See Also
        --------
        Configger.__getitem__ : Replacing method.
        Configger.items : Replacing method.
        Configger.keys : Replacing method.
        Configger.values : Replacing method.

        """

        raise DeprecationWarning("'get' will be removed in version 1.3.0 - "
                                 "use 'var[name]', 'items', 'keys', or "
                                 "'values' instead.")

        if key is None:
            return dict(self.items())
        else:
            return self[key]

    def items(self):
        """
        Get the configuration parameters as key, value pairs.

        Returns
        -------
        items : set-like
            The list of parameters.

        """

        for key in self.keys():
            yield (key, self[key])

    def keys(self):
        """
        Get the configuration parameter keys.

        Returns
        -------
        keys : set-like
            The keys.

        """

        return self._params.keys()

    def reset(self):
        """
        Reset the parameter values to the default values.

        """

        self._params = self._default.copy()

    def set(self, dictionary={}, **kwargs):
        """
        Deprecated method.

        See Also
        --------
        Configger.__setitem__ : Replacing function.

        """

        raise DeprecationWarning("'set' will be removed in version 1.3.0 - "
                                 "use 'var[name] = value' or 'update' "
                                 "instead.")

        self.update(dictionary, **kwargs)

    def update(self, params={}, **kwargs):
        """
        Update the value of one or more configuration parameters.

        Each value is validated according to the validation scheme of that
        parameter.

        Parameters
        ----------
        params : dict, optional
            A dictionary containing the key and values to update. (the default
            value is an empty dictionary)
        kwargs : dict
            Keyword arguments being the key and values to update.

        """

        @_decorate_validation
        def validate_input():
            _generic('params', 'mapping', keys_in=tuple(self._params.keys()))

            for name, var in (('params', params), ('kwargs', kwargs)):
                for key in var.keys():
                    validation = self._valids[key]
                    self._funcs[validation[0]]((name, key), *validation[1:])

        validate_input()

        if params is not None:
            for key, value in params.items():
                self[key] = value

        if len(kwargs) > 0:
            for key, value in kwargs.items():
                self[key] = value

    def values(self):
        """
        Get the configuration parameter values.

        Returns
        -------
        values : set-like
            The values.

        """

        for key in self.keys():
            yield self[key]
