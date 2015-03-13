"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the deprecated validation functionality.

See Also
--------
magni.utils.validation.validate_generic : Replacing function.
magni.utils.validation.validate_levels : Replacing function.
magni.utils.validation.validate_numeric : Replacing function.

"""

from __future__ import division

import numpy as np


def validate(var, path, levels, ignore_none=False):
    """
    Deprecated function.

    See Also
    --------
    magni.utils.validation.validate_generic : Replacing function.
    magni.utils.validation.validate_levels : Replacing function.
    magni.utils.validation.validate_numeric : Replacing function.

    """

    raise DeprecationWarning("'validate' will be removed in version 1.3.0 - "
                             "use 'validate_generic', 'validate_levels', or "
                             "'validate_numeric' instead.")

    if ignore_none and var is None:
        return

    if isinstance(levels, dict):
        levels = [levels]

    c = levels[0]

    if 'type' in c and not type(var) is c['type']:
        _raise('type', 'type({!r}) must be {!r}.', (path, c['type']))

    if 'type_in' in c and not type(var) in c['type_in']:
        _raise('type', 'type({!r}) must be in {!r}.', (path, c['type_in']))

    if 'class' in c and not isinstance(var, c['class']):
        _raise('type', '{!r} must be an instance of {!r}.', (path, c['class']))

    if 'val_in' in c and var not in c['val_in']:
        _raise('value', '{!r} must be in {!r}.', (path, c['val_in']))

    if 'min' in c and var < c['min']:
        _raise('value', '{!r} must be >= {!r}.', (path, c['min']))

    if 'max' in c and var > c['max']:
        _raise('value', '{!r} must be <= {!r}.', (path, c['max']))

    if 'len' in c and len(var) != c['len']:
        _raise('value', '{!r} must be of length {!r}.', (path, c['len']))

    if 'keys' in c:
        for key in c['keys']:
            if key not in var:
                _raise('key', '{!r} must have the key {!r}.', (path, key))

    if 'keys_in' in c:
        for key in var:
            if key not in c['keys_in']:
                _raise('key', '{!r} may only have keys in {!r}.',
                       (path, c['keys_in']))

    if len(levels) > 1:
        if not isinstance(var, dict):
            for i, item in enumerate(var):
                validate(item, '{}[{}]'.format(path, i), levels[1:],
                         ignore_none=ignore_none)
        else:
            for key, item in var.items():
                validate(item, '{}[{}]'.format(path, key), levels[1:],
                         ignore_none=ignore_none)


def validate_ndarray(var, path, constraints={}, ignore_none=False):
    """
    Deprecated function.

    See Also
    --------
    magni.utils.validation.validate_generic : Replacing function.
    magni.utils.validation.validate_levels : Replacing function.
    magni.utils.validation.validate_numeric : Replacing function.

    """

    raise DeprecationWarning("'validate_ndarray' will be removed in version "
                             "1.3.0 - use 'validate_numeric' instead.")

    if ignore_none and var is None:
        return

    c = constraints

    if not isinstance(var, np.ndarray):
        _raise('type', '{!r} must be an instance of np.ndarray.', (path,))

    if 'dtype' in c and var.dtype.type != c['dtype']:
        _raise('type', '{!r}.dtype must be {!r}.', (path, c['dtype']))

    if 'subdtype' in c and not np.issubdtype(var.dtype, c['subdtype']):
        _raise('type', '{!r}.dtype must be a supdtype of {!r}.',
               (path, c['subdtype']))

    if 'shape' in c and var.shape != c['shape']:
        _raise('type', '{!r}.shape must be {!r}.', (path, c['shape']))

    if 'dim' in c and len(var.shape) != c['dim']:
        _raise('type', 'The dimension of {!r} must be {!r}.', (path, c['dim']))


def _raise(error, message, args):
    """
    Deprecated function.

    """

    if error == 'type':
        error = TypeError
    elif error == 'value':
        error = ValueError
    elif error == 'key':
        error = KeyError

    raise error(message.format(*args))
