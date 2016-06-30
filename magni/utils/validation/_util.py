"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality for disabling validation.

The disabling functionality is provided through a decorator for validation
functions and a function for disabling validation. Furthermore, the present
module provides functionality which is internal to `magni.utils.validation`.

Routine listings
----------------
decorate_validation(func)
    Decorate a validation function to allow disabling of validation checks.
disable_validation()
    Disable validation checks in `magni`.
enable_validate_once()
    Enable validating inputs only once in certain functions in `magni`.
get_var(name)
    Retrieve the value of a variable through call stack inspection.
report(type, description, format_args=(), var_name=None, var_value=None,
    expr='{}', prepend='')
    Raise an exception.
validate_once(func)
    Decorate a function to allow for a one-time input validation only.

"""

from __future__ import division

import functools
import inspect
import os


_disabled = False
_validate_once_enabled = False


def decorate_validation(func):
    """
    Decorate a validation function to allow disabling of validation checks.

    Parameters
    ----------
    func : function
        The validation function to be decorated.

    Returns
    -------
    func : function
        The decorated validation function.

    See Also
    --------
    disable_validation : Disabling of validation checks.

    Notes
    -----
    This decorater wraps the validation function in another function which
    checks if validation has been disabled. If validation has been disabled,
    the validation function is not called. Otherwise, the validation function
    is called.

    Examples
    --------
    See `disable_validation` for an example.

    """

    @functools.wraps(func)
    def wrapper():
        if not _disabled:
            func()

    return wrapper


def disable_validation():
    """
    Disable validation checks in `magni`.

    See Also
    --------
    decorate_validation : Decoration of validation functions.

    Notes
    -----
    This function merely sets a global flag and relies on `decorate_validation`
    to perform the actual disabling.

    Examples
    --------
    An example of a function which accepts only an integer as argument:

    >>> import magni
    >>> def test(arg):
    ...     @magni.utils.validation.decorate_validation
    ...     def validate_input():
    ...         magni.utils.validation.validate_numeric('arg', 'integer')
    ...     validate_input()

    If the function is called with anything but an integer, it fails:

    >>> try:
    ...     test('string')
    ... except BaseException:
    ...     print('An exception occured')
    ... else:
    ...     print('No exception occured')
    An exception occured

    However, if validation is disabled, the same call does not fail:

    >>> from magni.utils.validation import disable_validation
    >>> disable_validation()
    >>> try:
    ...     test('string')
    ... except BaseException:
    ...     print('An exception occured')
    ... else:
    ...     print('No exception occured')
    No exception occured

    Although strongly discouraged, validation can be re-enabled after being
    disabled in the following way:

    >>> import magni.utils.validation._util
    >>> magni.utils.validation._util._disabled = False

    As this interface is not public it may be subject to changes in future
    releases without further notice.

    """

    global _disabled
    _disabled = True


def enable_validate_once():
    """
    Enable validating inputs only once in certain functions in `magni`.

    See Also
    --------
    validate_once : Decoration of functions.

    Notes
    -----
    This function merely sets a global flag and relies on `validate_once`
    to implement the actual "validate once" functionality.

    Examples
    --------
    An example of a function which accepts only an integer as argument:

    >>> import magni
    >>> @magni.utils.validation.validate_once
    ... def test(arg):
    ...     @magni.utils.validation.decorate_validation
    ...     def validate_input():
    ...         magni.utils.validation.validate_numeric('arg', 'integer')
    ...     validate_input()

    If "validate once" is enabled, the function validation is only called on
    its first run:

    >>> magni.utils.validation.enable_validate_once()
    >>> try:
    ...     test('string')
    ... except BaseException:
    ...     print('An exception occured')
    ... else:
    ...     print('No exception occured')
    An exception occured

    >>> try:
    ...     test('string')
    ... except BaseException:
    ...     print('An exception occured')
    ... else:
    ...     print('No exception occured')
    No exception occured

    """

    global _validate_once_enabled
    _validate_once_enabled = True


def get_var(name):
    """
    Retrieve the value of a variable through call stack inspection.

    `name` must refer to a variable in the parent scope of the function or
    method decorated by `magni.utils.validation.decorate_validation` which is
    closest to the top of the call stack. If `name` is a string then there must
    be a variable of that name in that scope. If `name` is a set-like object
    then there must be a variable having the first value in that set-like
    object as name. The remaining values are used as keys/indices on the
    variable to obtain the variable to be validated. For example, the `name`
    ('name', 0, 'key') refers to the variable "name[0]['key']".

    Parameters
    ----------
    name : None
        The name of the variable to be retrieved.

    Returns
    -------
    var : None
        The value of the retrieved variable.

    Notes
    -----
    The present function searches the call stack from top to bottom until it
    finds a function named 'wrapper' defined in this file. That is, until it
    finds a decorated validation function. The present function then looks up
    the variable indicated by `name` in the parent scope of that decorated
    validation function.

    """

    frame = inspect.currentframe()
    path = frame.f_code.co_filename
    index = path.rindex(os.path.sep)
    path = path[:index]
    frame = frame.f_back.f_back
    code = frame.f_code

    try:
        while code.co_firstlineno != 73 or code.co_filename[:index] != path:
            frame = frame.f_back
            code = frame.f_code
    except AttributeError:
        report(RuntimeError, 'Validation functions must be decorated.',
               prepend='Invalid validation call: ')

    if not isinstance(name, (list, tuple)):
        name = (name,)

    lookups = name[1:]
    name = name[0]

    try:
        var = frame.f_back.f_locals[name]
    except KeyError:
        report(NameError, 'must refer to an argument of {!r}.',
               frame.f_back.f_code.co_name, var_name='name', var_value=name,
               prepend='Invalid validation call: ')

    try:
        for i, lookup in enumerate(lookups):
            var = var[lookup]
    except Exception:
        report(LookupError, 'must have the key or index, {!r}.', lookup,
               var_name=[name] + list(lookups[:i]))

    return var


def report(type_, description, format_args=(), var_name=None, var_value=None,
           expr='{}', prepend=''):
    """
    Raise an exception.

    The type of the exception is given by `type_`, and the message can take on
    many forms. This ranges from a single description to a description
    formatted using a number of arguments, prepended by an expression using a
    variable name followed by the variable value, prepended by another
    description.

    Parameters
    ----------
    type_ : type
        The exception type.
    description : str
        The core description.
    format_args : list or tuple
        The arguments which `description` is formatted with. (the default is
        (), which implies no arguments)
    var_name : None
        The name of the variable which the description concerns. (the default
        is None, which implies that no variable name and value is prepended to
        the description)
    var_value : None
        The value of the variable which the description concerns. (the default
        is None, which implies that the value is looked up from the `var_name`
        in the call stack if `var_name` is not None)
    expr : str
        The expression to evaluate with the variable name. (the default is
        '{}', which implies that the variable value is used directly)
    prepend : str
        The text to prepend to the message generated by the previous arguments.
        (the default is '')

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

    """

    try:
        descr = description.format(format_args)
    except IndexError:
        descr = description.format(*format_args)

    if var_name is not None:
        value = get_var(var_name) if var_value is None else var_value
        value = eval(expr.replace('{}', 'value'))
        descr = ', {!r}, '.format(value) + descr

        if not isinstance(var_name, (list, tuple)):
            var_name = (var_name,)

        path = ['[{!r}]'.format(part) for part in var_name[1:]]
        path = var_name[0] + ''.join(path)
        descr = ('>>' + expr + '<<').format(path) + descr

        descr = 'The value(s) of ' + descr

    descr = prepend + descr
    raise type_(descr)


def validate_once(func):
    """
    Decorate a function to allow for a one-time input validation only.

    Parameters
    ----------
    func : function
        The validation function to be decorated.

    Returns
    -------
    func : function
        The decorated validation function.

    See Also
    --------
    enable_validate_once : Enabling allow validate once functionality.

    Notes
    -----
    This decorater wraps any function. It checks if the "validate once" has
    been enabled. If it has been enabled, the validation function is only
    called on the first run. Otherwise, the validation function is always
    called if not otherwise disabled.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _disabled

        if _disabled or not _validate_once_enabled:
            return func(*args, **kwargs)
        elif not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)
        else:
            try:
                _disabled = True
                return func(*args, **kwargs)
            finally:
                _disabled = False

    wrapper.has_run = False

    return wrapper
