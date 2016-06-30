"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing validation capability.

The intention is to validate all public functions of the package such that
erroneous arguments in calls are reported in an informative fashion rather than
causing arbitrary exceptions or unexpected results. To avoid performance
impairments, the validation can be disabled globally.

Routine listings
----------------
types
    Module providing abstract superclasses for validation.
decorate_validation(func)
    Decorate a validation function (see Notes).
disable_validation()
    Disable validation globally (see Notes).
enable_validate_once()
    Enable validating inputs only once (see Notes).
validate_generic(name, type, value_in=None, len_=None, keys_in=None,
    has_keys=None, superclass=None, ignore_none=False, var=None)
    Validate non-numeric objects.
validate_levels(name, levels)
    Validate containers and mappings as well as contained objects.
validate_numeric(name, type, range_='[-inf;inf]', shape=(), precision=None,
    ignore_none=False, var=None)
    Validate numeric objects.
validate_once(func)
    Decorate a function to allow for a one-time input validation (see Notes).

Notes
-----
To be able to disable validation (and to ensure consistency), every public
function or method should define a nested validation function with the name
'validate_input' which takes no arguments. This function should be decorated by
`decorate_validation`, be placed in the beginning of the parent function or
method, and be called as the first thing after its definition.

Functions in magni may be decorated by `validate_once`. If the validate once
functionality is enabled, these functions only validate their input arguments
on the first call to the function.

Examples
--------
If, for example, the following function is defined:

>>> def greet(person, greeting):
...     print('{}, {} {}.'.format(greeting, person['title'], person['name']))

This function expects its argument, 'person' to be a dictionary with keys
'title' and 'name' and its argument, 'greeting' to be a string. If, for
example, a list is passed as the first argument, a TypeError is raised with the
description 'list indices must be integers, not str'. While obviously correct,
this message is not excessively informative to the user of the function.
Instead, this module can be used to redefine the function as follows:

>>> from magni.utils.validation import decorate_validation, validate_generic
>>> def greet(person, greeting):
...     @decorate_validation
...     def validate_input():
...         validate_generic('person', 'mapping', has_keys=('title', 'name'))
...         validate_generic('greeting', 'string')
...     validate_input()
...     print('{}, {} {}.'.format(greeting, person['title'], person['name']))

If, again, a list is passed as the first argument, a TypeError with the
description "The value(s) of >>type(person)<<, <type 'list'>, must be in
('mapping',)." is raised. Now, the user of the function can easily identify the
mistake and correct the call to read:

>>> greet({'title': 'Mr.', 'name': 'Anderson'}, 'You look surprised to see me')
You look surprised to see me, Mr. Anderson.

"""

from magni.utils.validation import types
from magni.utils.validation._generic import validate_generic
from magni.utils.validation._numeric import validate_numeric
from magni.utils.validation._levels import validate_levels
from magni.utils.validation._util import decorate_validation
from magni.utils.validation._util import enable_validate_once
from magni.utils.validation._util import disable_validation
from magni.utils.validation._util import validate_once
