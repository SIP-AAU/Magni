"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the `validate_numeric` function.

Routine listings
----------------
validate_numeric(name, type, range_='[-inf;inf]', shape=(), precision=None,
    ignore_none=False, var=None)
    Validate numeric objects.

"""

from __future__ import division

import numpy as np

from magni.utils.validation.types import MatrixBase as _MatrixBase
from magni.utils.validation._util import get_var as _get_var
from magni.utils.validation._util import report as _report


try:
    long = long
except NameError:
    long = int

_types = {
    'boolean': {
        None: (bool, np.bool_),
        8: np.bool8 if 'bool8' in dir(np) else np.bool_},
    'integer': {
        None: (int, np.signedinteger, long),
        8: getattr(np, 'int8', np.int_),
        16: getattr(np, 'int16', np.int_),
        32: getattr(np, 'int32', np.int_),
        64: getattr(np, 'int64', np.int_)},
    'floating': {
        None: (float, np.floating),
        16: getattr(np, 'float16', np.float_),
        32: getattr(np, 'float32', np.float_),
        64: getattr(np, 'float64', np.float_),
        128: getattr(np, 'float128', np.float_)},
    'complex': {
        None: (complex, np.complexfloating),
        32: getattr(np, 'complex64', np.complex_),
        64: getattr(np, 'complex128', np.complex_),
        128: getattr(np, 'complex256', np.complex_)}}


def validate_numeric(name, type_, range_='[-inf;inf]', shape=(),
                     precision=None, ignore_none=False, var=None):
    """
    Validate numeric objects.

    The present function is meant to valdiate the type or class of an object.
    Furthermore, if the object may only take on a connected range of values,
    the object can be validated against this range. Also, the shape of the
    object can be validated. Finally, the precision used to represent the
    object can be validated.

    If the present function is called with `name` set to None, an iterable with
    the value 'numeric' followed by the remaining arguments passed in the call
    is returned. This is useful in combination with the validation function
    `magni.utils.validation.validate_levels`.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    type_ : None
        One or more references to groups of types.
    range_ : None
        The range of accepted values. (the default is '[-inf;inf]', which
        implies that all values are accepted)
    shape : list or tuple
        The accepted shape. (the default is (), which implies that only scalar
        values are accepted)
    precision : None
        One or more precisions.
    ignore_none : bool
        A flag indicating if the variable is allowed to be none. (the default
        is False)
    var : None
        The value of the variable to be validated.

    See Also
    --------
    magni.utils.validation.validate_generic : Validate non-numeric objects.
    magni.utils.validation.validate_levels : Validate contained objects.

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
    set-like object containing at least one value. Each value refers to a
    number of data types depending if the string value is 'boolean', 'integer',
    'floating', or 'complex'.

    - 'boolean' tests if the variable is a bool or has the data type
      `numpy.bool8`.
    - 'integer' tests if the variable is an int or has the data type
      `numpy.int8`, `numpy.int16`, `numpy.int32`, or `numpy.int64`.
    - 'floating' tests if the variable is a float or has the data type
      `numpy.float16`, `numpy.float32`, `numpy.float64`, or `numpy.float128`.
    - 'complex' tests if the variable is a complex or has the data type
      `numpy.complex32`, `numpy.complex64`, or `numpy.complex128`.

    `range_` is either a list with two strings or a single string. In the
    latter case, the default value of the argument is used as the second
    string. The first value represents the accepted range of real values
    whereas the second value represents the accepted range of imaginary values.
    Each string consists of the following parts:

    - One of the following delimiters: '[', '(', ']'.
    - A numeric value (or '-inf').
    - A semi-colon.
    - A numeric value (or 'inf').
    - One of the following delimiters: ']', ')', '['.

    `shape` is either None meaning that any shape is accepted or a list of
    integers. In the latter case, the integer -1 may be used to indicate that
    the given axis may have any length.

    `precision` is either an integer treated as a list with one value or a
    set-like object containing at least one integer. Each value refers to an
    accepted number of bits used to store each value of the variable.

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
        return ('numeric', type_, range_, shape, precision, ignore_none)

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

    dtype, bounds, dshape = _examine_var(name, var)

    if isinstance(type_, str) or not hasattr(type_, '__iter__'):
        type_ = (type_,)

    _check_type(name, dtype, type_)
    _check_range(name, bounds, range_)
    _check_shape(name, dshape, shape)
    _check_precision(name, dtype, type_, precision)


def _check_precision(name, dtype, types_, precision):
    """
    Check the precision of a variable.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    dtype : type
        The data type of the variable to be validated.
    types_ : set-like
        The list of accepted types.
    precision : None
        The accepted precision(s).

    """

    if not isinstance(precision, str) and hasattr(precision, '__iter__'):
        precisions = precision
    else:
        precisions = (precision,)

    if None not in precisions:
        dtypes = []

        for type_ in types_:
            for precison in precisions:
                if precision not in _types[type_]:
                    _report(ValueError, 'must be in {!r}.',
                            _types[type_].keys(), var_name='precision',
                            var_value=precisions,
                            prepend='Invalid validation call: ')

                dtypes.append(_types[type_][precision])

        if not isinstance(dtype(), tuple(dtypes)):
            _report(TypeError, '>>{}.dtype<<, {!r}, must be in {!r}.',
                    (name, dtype, dtypes), prepend='The value(s) of ')


def _check_range(name, bounds, range_):
    """
    Check the range of a variable.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    bounds : list or tuple
        The bounds of the variable to be validated.
    range_ : None
        The accepted range(s).

    """

    if not isinstance(range_, (list, tuple)):
        range_ = (range_, '[-inf;inf]')

    try:
        tests = [req.split(';') for req in range_]

        for i in (0, 1):
            tests[i][0] = [bounds[i][0], tests[i][0][0], tests[i][0][1:]]
            tests[i][0][1] = {']': '>', '(': '>', '[': '>='}[tests[i][0][1]]
            tests[i][1] = [bounds[i][1], tests[i][1][-1], tests[i][1][:-1]]
            tests[i][1][1] = {'[': '<', ')': '<', ']': '<='}[tests[i][1][1]]

        for expr, tests_ in zip(('real({})', 'imag({})'), tests):
            for func, test in zip(('min', 'max'), tests_):
                bound, op, value = test
                test = '{} {} {}'.format(bound, op, value)

                if not eval(test.replace('inf', 'np.inf')):
                    _report(ValueError, '>>{}({})<<, {!r}, must be {} {!r}.',
                            (func, expr.format(name), bound, op, eval(value)),
                            prepend='The value(s) of ')
    except (AttributeError, KeyError, SyntaxError):
        _report(ValueError, 'must be a valid range.', var_name='range_',
                var_value=range_, prepend='Invalid validation call: ')


def _check_shape(name, dshape, shape):
    """
    Check the shape of a variable.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    dshape : list or tuple
        The shape of the variable to be validated.
    shape : list or tuple
        The accepted shape.

    """

    if shape is not None:
        if not isinstance(shape, (list, tuple)):
            _report(TypeError, 'must be in {!r}.', (list, tuple),
                    var_name='shape', var_value=shape, expr='type({})',
                    prepend='Invalid validation call: ')

        if len(dshape) != len(shape):
            _report(ValueError, '>>len({}.shape)<<, {!r}, must be {!r}.',
                    (name, len(dshape), len(shape)),
                    prepend='The value(s) of ')

        for i, value in enumerate(shape):
            if not isinstance(value, (int, long)):
                _report(TypeError, 'must be {!r}.', (int, long),
                        var_name=('shape', i), var_value=value,
                        expr='type({})', prepend='Invalid validation call: ')

            if value > -1 and dshape[i] != value:
                _report(ValueError, '>>{}.shape[{}]<<, {!r}, must be {!r}.',
                        (name, i, dshape[i], value), prepend='The value(s) of')


def _check_type(name, dtype, types_):
    """
    Check the type of a variable.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    dtype : type
        The data type of the variable to be validated.
    types_ : set-like
        The accepted types.

    """

    valid = False

    for type_ in types_:
        if type_ not in _types.keys():
            _report(ValueError, 'must be in {!r}.', _types.keys(),
                    var_name='type_', var_value=type_,
                    prepend='Invalid validation call: ')

        if isinstance(dtype(), _types[type_][None]):
            valid = True
            break

    if not valid:
        _report(TypeError, '>>{}.dtype<<, {!r}, must be in {!r}.',
                (name, dtype, types_), prepend='The value(s) of ')


def _examine_var(name, var):
    """
    Examine a variable.

    The present function examines the data type, the bounds, and the shape of a
    variable. The variable can be of a built-in type, a numpy type, or a
    subclass of the `magni.utils.validation.types.MatrixBase` class.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    var : None
        The value of the variable to be examined.

    Returns
    -------
    dtype : type
        The data type of the examined variable.
    bounds : tuple
        The bounds of the examined variable.
    dshape : tuple
        The shape of the examined variable.

    """

    if isinstance(var, (np.generic, np.ndarray)):
        dtype = var.dtype.type

        if len(var.shape) > 0 and max(var.shape) == 0:
            bounds = ((np.inf, -np.inf), (np.inf, -np.inf))
        elif isinstance(dtype(), _types['complex'][None]):
            bounds = ((np.nanmin(var.real), np.nanmax(var.real)),
                      (np.nanmin(var.imag), np.nanmax(var.imag)))
        else:
            bounds = ((np.nanmin(var), np.nanmax(var)), (0, 0))

        dshape = var.shape
    elif isinstance(var, _MatrixBase):
        dtype = var.dtype
        bounds = var.bounds
        dshape = var.shape
    else:
        dtype = type(var)

        if isinstance(var, _types['complex'][None]):
            bounds = ((var.real, var.real), (var.imag, var.imag))
        else:
            bounds = ((var, var), (0, 0))

        dshape = ()

    if np.any(np.isnan(bounds[0])):
        bounds = ((np.inf, -np.inf), bounds[1])

    if np.any(np.isnan(bounds[1])):
        bounds = (bounds[0], (np.inf, -np.inf))

    return dtype, bounds, dshape
