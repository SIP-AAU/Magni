"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the `validate_levels` function.

Routine listings
----------------
validate_levels(name, levels)
    Validate containers and mappings as well as contained objects.

"""

from __future__ import division

from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.validation._util import get_var as _get_var
from magni.utils.validation._util import report as _report


def validate_levels(name, levels):
    """
    Validate containers and mappings as well as contained objects

    The present function is meant to valdiate the 'levels' of a variable. That
    is, the value of the variable itself, the values of the second level (in
    case the value is a list, tuple, or dict), the values of the third level,
    and so on.

    Parameters
    ----------
    name : None
        The name of the variable to be validated.
    levels : list or tuple
        The list of levels.

    See Also
    --------
    magni.utils.validation.validate_generic : Validate non-numeric objects.
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

    `levels` is a list containing the levels. The value of the variable is
    validated against the first level. In case the value is a list, tuple, or
    dict, the values contained in this are validated against the second level
    and so on. Each level is itself a list with the first value being either
    'generic' or 'numeric' followed by the arguments that should be passed to
    the respective function (with the exception of `name` which is
    automatically prepended by the present function).

    Examples
    --------
    Every public function and method of the present package (with the exception
    of the functions of this subpackage itself) validates every argument and
    keyword argument using the functionality of this subpackage. Thus, for
    examples of how to use the present function, browse through the code.

    """

    if name is None:
        return ('levels', levels)

    if isinstance(name, (list, tuple)):
        name = list(name)
    elif isinstance(name, str):
        name = [name]
    elif hasattr(name, '__iter__'):
        name = [value for value in name]
    else:
        name = [name]

    if not isinstance(levels, (list, tuple)):
        _report(TypeError, 'must be in {!r}.', (list, tuple),
                var_name='levels', var_value=levels, expr='type({})',
                prepend='Invalid validation call: ')

    _validate_level(name, _get_var(name), levels)


def _validate_level(name, var, levels, index=0):
    """
    Validate a level.

    Parameters
    ----------
    name : None
        The name of the variable.
    var : None
        The value of the variable.
    levels : set-like
        The levels.
    index : int
        The index of the current level. (the default is 0)

    """

    if not isinstance(levels[index], (list, tuple)):
        _report(TypeError, 'must be in {!r}.', (list, tuple),
                var_name=('levels', index), var_value=levels[index],
                expr='type({})', prepend='Invalid validation call: ')

    if levels[index][0] not in ('generic', 'numeric'):
        _report(ValueError, 'must be in {!r}.', ('generic', 'numeric'),
                var_name=('levels', index, 0), var_value=levels[index][0],
                prepend='Invalid validation call: ')

    func = {'generic': _generic, 'numeric': _numeric}[levels[index][0]]
    args = (name,) + tuple(levels[index][1:]) + (var,)
    func(*args)

    if index + 1 < len(levels):
        if isinstance(var, (list, tuple)):
            for i, value in enumerate(var):
                _validate_level(name + [i], value, levels, index + 1)
        elif isinstance(var, dict):
            for key, value in var.items():
                _validate_level(name + [key], value, levels, index + 1)
