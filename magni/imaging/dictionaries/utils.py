"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing support functionality for the dictionaries subpackage.

Routine listings
----------------
get_function_handle(type\_, transform)
    Function to get a function handle to a transform method.
get_transform_names()
    Function to get a tuple of names of the available transforms.

"""

from __future__ import division
import types

from magni.imaging.dictionaries import _matrices
from magni.imaging.dictionaries import _visualisations
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic


def get_function_handle(type_, transform):
    """
    Return a function handle to a given transform method.

    Parameters
    ----------
    type_ : {'matrix', 'visualisation'}
        Identifier of the type of method to return a handle to.
    transform : str
        Identifier of the transform method to return a handle to.

    Returns
    -------
    f_handle : function
        Handle to `transform`.

    Examples
    --------
    For example, return a handle to the matrix method for a DCT:

    >>> from magni.imaging.dictionaries.utils import get_function_handle
    >>> get_function_handle('matrix', 'DCT').__name__
    'get_DCT'

    or return a handle to the visualisation method for a DFT:

    >>> get_function_handle('visualisation', 'DFT').__name__
    'visualise_DFT'

    """

    @_decorate_validation
    def validate_input():
        _generic('type_', 'string', value_in=('matrix', 'visualisation'))
        _generic('transform', 'string')

    @_decorate_validation
    def validate_output():
        _generic('f_handle', 'function')

    validate_input()

    modules = {'matrix': _matrices, 'visualisation': _visualisations}

    if type_ == 'matrix':
        f_handle = vars(modules[type_])['get_' + transform]
    elif type_ == 'visualisation':
        f_handle = vars(modules[type_])['visualise_' + transform]

    validate_output()

    return f_handle


def get_transform_names():
    """
    Return a tuple of names of the available transforms.

    Returns
    -------
    names : tuple
        The tuple of names of available transforms.

    Examples
    --------
    For example, get transform names and extract 'DCT' and 'DFT'

    >>> from magni.imaging.dictionaries.utils import get_transform_names
    >>> names = get_transform_names()
    >>> tuple(sorted(name for name in names if name in ('DCT', 'DFT')))
    ('DCT', 'DFT')

    and a handle to corresponding visualisation method for the DCT

    >>> from magni.imaging.dictionaries.utils import get_function_handle
    >>> f_handles = tuple(get_function_handle('visualisation', name)
    ... for name in names)
    >>> tuple(f_h.__name__ for f_h in f_handles if 'DCT' in f_h.__name__)
    ('visualise_DCT',)

    """

    name_candidates = _matrices.__dict__.keys()
    names = [candidate[4:] for candidate in name_candidates
             if candidate[:4] == 'get_']

    return tuple(names)
