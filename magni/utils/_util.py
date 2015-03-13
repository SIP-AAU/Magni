"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the public function of the magni.utils subpackage.

"""

from __future__ import division

import os

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic


def split_path(path):
    """
    Split a path into folder path, file name, and file extension.

    The returned folder path ends with a folder separation character while the
    returned file extension starts with an extension separation character. The
    function is independent of the operating system and thus of the use of
    folder separation character and extension separation character.

    Parameters
    ----------
    path : str
        The path of the file either absolute or relative to the current working
        directory.

    Returns
    -------
    path : str
        The path of the containing folder of the input path.
    name : str
        The name of the object which the input path points to.
    ext : str
        The extension of the object which the input path points to (if any).

    Examples
    --------
    Concatenate a dummy path and split it using the present function:

    >>> import os
    >>> from magni.utils._util import split_path
    >>> path = 'folder' + os.sep + 'file' + os.path.extsep + 'extension'
    >>> parts = split_path(path)
    >>> print(tuple((parts[0][-7:-1], parts[1], parts[2][1:])))
    ('folder', 'file', 'extension')

    """

    @_decorate_validation
    def validate_input():
        _generic('path', 'string')

    validate_input()

    path = os.path.realpath(str(path))
    pos = str.rfind(path, os.path.sep) + 1
    path, name = path[:pos], path[pos:]

    if os.path.extsep in name:
        pos = str.rfind(name, os.path.extsep)
        name, ext = name[:pos], name[pos:]
    else:
        ext = ''

    return (path, name, ext)
