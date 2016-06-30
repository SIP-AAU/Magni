"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the `validate_generic` function.

Routine listings
----------------
validate_generic(name, type, value_in=None, len_=None, keys_in=None,
    has_keys=None, superclass=None, ignore_none=False, var=None)
    Validate non-numeric objects.

"""

from __future__ import absolute_import
from __future__ import division

import types

from magni.utils.validation._util import get_var as _get_var
from magni.utils.validation._util import report as _report


try:
    unicode = unicode
except NameError:
    _string = (str, bytes)
else:
    _string = (str, unicode)

_types = {'string': _string,
          'explicit collection': (list, tuple),
          # 'implicit collection': ...,
          # 'collection': ...,
          'mapping': dict,
          'function': types.FunctionType,
          'class': type}


def validate_generic(name, type_, value_in=None, len_=None, keys_in=None,
                     has_keys=None, superclass=None, ignore_none=False,
                     var=None):
    """
    Validate non-numeric objects.

    The present function is meant to validate the type or class of an object.
    Furthermore, if the object may only take on a limited number of values, the
    object can be validated against this list. In the case of collections (for
    example lists and tuples) and mappings (for example dictionaries), a
    specific length can be required. Furthermore, in the case of mappings, the
    keys can be validated by requiring and/or only allowing certain keys.

    If the present function is called with `name` set to None, an iterable with
    the value 'generic' followed by the remaining arguments passed in the call
    is returned. This is useful in combination with the validation function
    `magni.utils.validation.validate_levels`.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    type_ : None
        One or more references to groups of types, specific types, and/or
        specific classes.
    value_in : set-like
        The list of values accepted. (the default is None, which implies that
        all values are accepted)
    len_ : int
        The length required. (the default is None, which implies that all
        lengths are accepted)
    keys_in : set-like
        The list of accepted keys. (the default is None, which implies that all
        keys are accepted)
    has_keys : set-like
        The list of required keys. (the default is None, which implies that no
        keys are required)
    superclass : class
        The required superclass. (the default is None, which implies that no
        superclass is required)
    ignore_none : bool
        A flag indicating if the variable is allowed to be none. (the default
        is False)
    var : None
        The value of the variable to be validated.

    See Also
    --------
    magni.utils.validation.validate_levels : Validate contained objects.
    magni.utils.validation.validate_numeric : Validate numeric objects.

    Notes
    -----
    `name` must refer to a variable in the parent scope of the function or
    method decorated by `magni.utils.validation.decorate_validation` which is
    closest to the top of the call stack. If `name` is a string then there must
    be a variable of that name in that scope. If `name` is a set-like object
    then there must be a variable having the first value in that set-like
    object as name. The remaining values are used as keys/indices on the
    variable to obtain the variable to be validated. For example, the `name`
    ('name', 0, 'key') refers to the variable "name[0]['key']".

    `type_` is either a single value treated as a list with one value or a
    set-like object containing at least one value. Each value is either a
    specific type or class, or it refers to one or more types by having one of
    the string values 'string', 'explicit collection', 'implicict collection',
    'collection', 'mapping', 'function', 'class'.

    - 'string' tests if the variable is a str.
    - 'explicit collection' tests if the variable is a list or tuple.
    - 'implicit collection' tests if the variable is iterable.
    - 'collection' is a combination of the two above.
    - 'mapping' tests if the variable is a dict.
    - 'function' tests if the variable is a function.
    - 'class' tests if the variable is a type.

    `var` can be used to pass the value of the variable to be validated. This
    is useful either when the variable cannot be looked up by `name` (for
    example, if the variable is a property of the argument of a function) or to
    remove the overhead of looking up the value.

    Examples
    --------
    Every public function and method of the present package (with the exception
    of the functions of this subpackage itself) validates every argument and
    keyword argument using the functionality of this subpackage. Thus, for
    examples of how to use the present function, browse through the code.

    """

    if name is None:
        return ('generic', type_, value_in, len_, keys_in, has_keys,
                superclass, ignore_none)

    if var is None:
        var = _get_var(name)

    if var is None:
        if not isinstance(ignore_none, bool):
            _report(TypeError, 'must be {!r}.', bool, var_name='ignore_none',
                    var_value=ignore_none, expr='type({})',
                    prepend='Invalid validation call: ')

        if ignore_none:
            return
        else:
            _report(ValueError, 'must not be {!r}.', None, var_name=name)

    if isinstance(type_, (str, type)) or not hasattr(type_, '__iter__'):
        type_ = (type_,)

    _check_type(name, var, type_)
    _check_value(name, var, value_in)
    _check_len(name, var, len_)
    _check_keys(name, var, keys_in, has_keys)
    _check_inheritance(name, var, superclass)


def _check_inheritance(name, var, superclass):
    """
    Check the superclasses of a variable.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    var : None
        The value of the variable to be validated.
    superclass : class
        The required superclass.

    """

    if superclass is not None:
        if not isinstance(superclass, type):
            _report(TypeError, 'must be a type.', var_name='superclass',
                    var_value=superclass, prepend='Invalid validation call: ')

        if not issubclass(var, superclass):
            _report(TypeError, 'must be a subclass of {!r}.', superclass,
                    var_name=name)


def _check_keys(name, var, keys_in, has_keys):
    """
    Check the keys of a variable.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    var : None
        The value of the variable to be validated.
    keys_in : set-like
        The allowed keys.
    has_keys : set-like
        The required keys.

    """

    if keys_in is not None:
        if not isinstance(keys_in, (list, tuple)):
            _report(TypeError, 'must be in {!r}.', (list, tuple),
                    var_name='keys_in', var_value=keys_in,
                    prepend='Invalid validation call: ')

        for key in var.keys():
            if key not in keys_in:
                _report(KeyError, 'must be in {!r}.', keys_in, var_name=name,
                        expr='{}.keys()')

    if has_keys is not None:
        if not isinstance(has_keys, (list, tuple)):
            _report(TypeError, 'must be in {!r}.', (list, tuple),
                    var_name='has_keys', var_value=has_keys,
                    prepend='Invalid validation call: ')

        for key in has_keys:
            if key not in var.keys():
                _report(KeyError, 'must have the key {!r}.', key,
                        var_name=name)


def _check_len(name, var, len_):
    """
    Check the length of a variable.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    var : None
        The value of the variable to be validated.
    len_ : int
        The required length.

    """

    if len_ is not None:
        if not isinstance(len_, int):
            _report(TypeError, 'must be {!r}.', int, var_name='len_',
                    var_value=len_, expr='type({})',
                    prepend='Invalid validation call: ')

        if not hasattr(var, '__len__') or len(var) != len_:
            _report(ValueError, 'must be {!r}.', len_, var_name=name,
                    expr='len({})')


def _check_type(name, var, types_):
    """
    Check the type of a variable.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    var : None
        The value of the variable to be validated.
    types_ : set-like
        The allowed types.

    """

    valid = False

    for type_ in types_:
        if isinstance(type_, type):
            if isinstance(var, type_):
                valid = True
                break
        elif type_ in _types.keys():
            if isinstance(var, _types[type_]):
                valid = True
                break
        elif type_ in ('implicit collection', 'collection'):
            if hasattr(var, '__iter__'):
                valid = True
                break
        else:
            _report(ValueError, 'must be a valid type.', var_name='type_',
                    var_value=type_, prepend='Invalid validation call: ')

    if not valid:
        _report(TypeError, 'must be in {!r}.', types_, var_name=name,
                expr='type({})')


def _check_value(name, var, value_in):
    """
    Check the value of a variable.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    var : None
        The value of the variable to be validated.
    value_in : set-like
        The allowed values.

    """

    if value_in is not None:
        if not isinstance(value_in, (list, tuple)):
            _report(TypeError, 'must be in {!r}.', (list, tuple),
                    var_name='value_in', var_value=value_in,
                    prepend='Invalid validation call: ')

        if var not in value_in:
            _report(ValueError, 'must be in {!r}.', value_in, var_name=name)
