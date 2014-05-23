"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing validation capability.

The intention is to validate all public functions of the package such that
erroneous arguments in calls are reported in an informative fashion rather than
causing arbitrary exceptions or unexpected results. To avoid performance
impairments, the validation can be disabled globally.

Routine listings
----------------
decorate_validation(func)
    Decorate a validation function (see Notes).
disable_validation()
    Disable validation globally (see Notes).
validate(var, path, levels, ignore_none=False)
    Validate the value of a variable according to a validation scheme.
validate_ndarray(var, path, constraints)
    Validate a numpy.ndarray according to a validation scheme.

Notes
-----
To be able to disable validation (and to ensure consistency), for every public
function a validation function with the same name prefixed by '_validate_'
should be defined. This function should be decorated by `decorate_validation`,
be placed just above the function which it decorates, and be called as the
first thing with all arguments.

Examples
--------
If, for example, the following function is defined:

>>> def greet(person, greeting):
...     print('{}, {} {}.'.format(greeting, person['title'], person['name']))

This function probably expects its argument, 'person' to be a dictionary with
keys 'title' and 'name' and its argument, 'greeting' to be a string. If, for
example, a list is passed as the first argument, a TypeError is raised with the
description 'list indices must be integers, not str'. While obviously correct,
this message is not excessively informative to the user of the function.
Instead, this module can be used to redefine the function as follows:

>>> from magni.utils.validation import validate, decorate_validation
>>> @decorate_validation
... def _validate_greet(person, greeting):
...     validate(person, 'person', {'type': dict, 'keys': ('title', 'name')})
...     validate(greeting, 'greeting', {'type': str})
>>> def greet(person, greeting):
...     _validate_greet(person, greeting)
...     print('{}, {} {}.'.format(greeting, person['title'], person['name']))

If, again, a list is passed as the first argument, a TypeError with the
description 'type(person) must be <type 'dict'>'. Now, the user of the function
can easily identify the mistake and correct the call to read:

>>> greet({'title': 'Mr.', 'name': 'Anderson'}, 'You look surprised to see me')
You look surprised to see me, Mr. Anderson.

"""

from __future__ import division

import numpy as np


_disabled = False


def decorate_validation(func):
    """
    Decorate a validation function (see module Notes).

    Parameters
    ----------
    func : function
        The validation function which should be decorated.

    Examples
    --------
    An example of a function which accepts only an integer as argument:

    >>> from magni.utils.validation import decorate_validation
    >>> @decorate_validation
    ... def _validate_test(arg):
    ...     magni.utils.validation.validate(arg, 'arg', {'type': int})
    >>> def test(arg):
    ...     _validate_test(arg)
    ...     return

    If the function is called with anything but an integer, it fails:

    >>> try:
    ...     test('string')
    ...     print('No exception occured')
    ... except BaseException:
    ...     print('An exception occured')
    An exception occured

    """

    def wrapper(*args, **kwargs):
        """
        Wrap a validation function (see module Notes).

        Parameters
        ----------
        args : tuple
            The arguments passed to the decorated function.
        kwargs : dict
            The keyword arguments passed to the decorated function.

        """

        if not _disabled:
            func(*args, **kwargs)

    return wrapper


def disable_validation():
    """
    Disable validation globally (see module Notes).

    There is no equivalent function to enable validation since either any or
    no function calls should be validated depending on the run mode.

    Examples
    --------
    An example of a function which accepts only an integer as argument:

    >>> @magni.utils.validation.decorate_validation
    ... def _validate_test(arg):
    ...     magni.utils.validation.validate(arg, 'arg', {'type': int})
    >>> def test(arg):
    ...     _validate_test(arg)
    ...     return

    If the function is called with anything but an integer, it fails:

    >>> try:
    ...     test('string')
    ...     print('No exception occured')
    ... except BaseException:
    ...     print('An exception occured')
    An exception occured

    However, if validation is disabled, the same call does not fail:

    >>> from magni.utils.validation import disable_validation
    >>> disable_validation()
    >>> try:
    ...     test('string')
    ...     print('No exception occured')
    ... except BaseException:
    ...     print('An exception occured')
    No exception occured

    """

    global _disabled
    _disabled = True


def validate(var, path, levels, ignore_none=False):
    """
    Validate the value of a variable according to a validation scheme.

    Parameters
    ----------
    var : None
        The variable to validate which can take any type.
    path : str
        The path of the variable which is printed if the variable is invalid.
    levels : list or tuple or dict
        The validation scheme.
    ignore_none : bool, optional
        The flag indicating whether the variable is allowed to have the value
        None (the default is False).

    Notes
    -----
    The `levels` parameter is either a dict or a sequence of dicts. If it is a
    dict, it is considered a sequence of one dict. Each dict corresponds to the
    validation scheme of a 'level': the first level consists of `var` itself,
    the second level consists of any value in `var` if `var` is itself a list,
    tuple, or dict, and so on.

    A level is validated according to the keys and values of the dict
    specifying the validation scheme of that level. See the source for details
    on the usable keys and their function.

    .. note:: It is assumed that `var`, `path`, `levels`, and `ignore_none`
              are all valid arguments as these are not explicitly validated.

    Examples
    --------
    Define a function for reporting if an exception occurs when calling making
    some call:

    >>> from magni.utils.validation import validate
    >>> def report(call):
    ...     try:
    ...         call()
    ...         print('No exception occured')
    ...     except BaseException:
    ...         print('An exception occured')

    An example of how to validate an integer:

    >>> var = 1
    >>> call = lambda: validate(var, 'var', {'type': int})
    >>> report(call)
    No exception occured

    The above code fails when, e.g., a string is passed instead:

    >>> var = 'string'
    >>> report(call)
    An exception occured

    An example of how to validate a tuple expected to hold two integers:

    >>> var = (42, 1337)
    >>> call = lambda: validate(var, 'var', ({'type': tuple, 'len': 2},
    ...                                      {'type': int}))
    >>> report(call)
    No exception occured

    The above code fails when, e.g., a tuple with three integers is passed
    instead:

    >>> var = (1, 2, 3)
    >>> report(call)
    An exception occured

    """

    if ignore_none and var is None:
        return

    if isinstance(levels, dict):
        levels = [levels]

    c = levels[0]

    if 'type' in c and not type(var) is c['type']:
        _error('type', 'type({!r}) must be {!r}.', (path, c['type']))

    if 'type_in' in c and not type(var) in c['type_in']:
        _error('type', 'type({!r}) must be in {!r}.', (path, c['type_in']))

    if 'class' in c and not isinstance(var, c['class']):
        _error('type', '{!r} must be an instance of {!r}.', (path, c['class']))

    if 'val_in' in c and var not in c['val_in']:
        _error('value', '{!r} must be in {!r}.', (path, c['val_in']))

    if 'min' in c and var < c['min']:
        _error('value', '{!r} must be >= {!r}.', (path, c['min']))

    if 'max' in c and var > c['max']:
        _error('value', '{!r} must be <= {!r}.', (path, c['max']))

    if 'len' in c and len(var) != c['len']:
        _error('value', '{!r} must be of length {!r}.', (path, c['len']))

    if 'keys' in c:
        for key in c['keys']:
            if key not in var:
                _error('key', '{!r} must have the key {!r}.', (path, key))

    if 'keys_in' in c:
        for key in var:
            if key not in c['keys_in']:
                _error('key', '{!r} may only have keys in {!r}.',
                       (path, c['keys_in']))

    if len(levels) > 1:
        if not isinstance(var, dict):
            for i, item in enumerate(var):
                validate(item, '{}[{}]'.format(path, i), levels[1:])
        else:
            for key, item in var.items():
                validate(item, '{}[{}]'.format(path, key), levels[1:])


def validate_ndarray(var, path, constraints={}, ignore_none=False):
    """
    Validate a numpy.ndarray according to a validation scheme.

    Parameters
    ----------
    var : numpy.ndarray
        The variable to validate.
    path : str
        The path of the variable which is printed if the variable is invalid.
    constraints : dict
        The validation scheme.
    ignore_none : bool, optional
        Boolean indicating whether the variable is allowed to have the value
        None (the default is False).

    Notes
    -----
    The variable is validated according to the keys and values of the dict
    specifying the validation scheme. See the source for details on the usable
    keys and their function.

    .. note:: It is assumed that `var`, `path`, `levels`, and `ignore_none`
              are all valid arguments as these are not explicitly validated.

    Examples
    --------
    Define a function for reporting if an exception occurs when calling making
    some call:

    >>> from magni.utils.validation import validate_ndarray
    >>> def report(call):
    ...     try:
    ...         call()
    ...         print('No exception occured')
    ...     except BaseException:
    ...         print('An exception occured')

    An example of how to validate a float64 numpy.ndarray of shape (5,):

    >>> var = np.float64([1, 2, 3, 4, 5])
    >>> call = lambda: validate_ndarray(var, 'var',
    ...                                 {'dtype': np.float64, 'shape': (5,)})
    >>> report(call)
    No exception occured

    The above code fails when, e.g., an int64 numpy.ndarray is passed instead:

    >>> var = np.int64([1, 2, 3, 4, 5])
    >>> report(call)
    An exception occured

    """

    if ignore_none and var is None:
        return

    c = constraints

    if not isinstance(var, np.ndarray):
        _error('type', '{!r} must be an instance of np.ndarray.', (path,))

    if 'dtype' in c and var.dtype.type != c['dtype']:
        _error('type', '{!r}.dtype must be {!r}.', (path, c['dtype']))

    if 'subdtype' in c and not np.issubdtype(var.dtype, c['subdtype']):
        _error('type', '{!r}.dtype must be a supdtype of {!r}.',
               (path, c['subdtype']))

    if 'shape' in c and var.shape != c['shape']:
        _error('type', '{!r}.shape must be {!r}.', (path, c['shape']))

    if 'dim' in c and len(var.shape) != c['dim']:
        _error('type', 'The dimension of {!r} must be {!r}.', (path, c['dim']))


def _error(error, message, args):
    """
    Raise an error of a given type with a given message.

    Parameters
    ----------
    error : {'type', 'value', 'key'}
        The type of the raised error.
    message : str
        The message of the raised error after being formatted using 'format'.
    args : list or tuple
        The args passed to 'format'.

    """

    if error == 'type':
        error = TypeError
    elif error == 'value':
        error = ValueError
    elif error == 'key':
        error = KeyError

    raise error(message.format(*args))
