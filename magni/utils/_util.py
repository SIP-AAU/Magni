"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the public function of the magni.utils subpackage.

"""

from __future__ import division

import os

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate as _validate


@_decorate_validation
def _validate_split_path(path):
    """
    Validate the `split_path` function.

    See Also
    --------
    split_path : The validated function.
    magni.utils.validation.validate : Validation.

    """

    _validate(path, 'path', {'type': str})


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

    >>> from magni.utils._util import split_path
    >>> path = 'folder' + os.sep + 'file' + os.path.extsep + 'extension'
    >>> parts = split_path(path)
    >>> print(tuple((parts[0][-7:-1], parts[1], parts[2][1:])))
    ('folder', 'file', 'extension')

    """

    _validate_split_path(path)

    path = os.path.realpath(path)
    pos = str.rfind(path, os.path.sep) + 1
    path, name = path[:pos], path[pos:]

    if os.path.extsep in name:
        pos = str.rfind(name, os.path.extsep)
        name, ext = name[:pos], name[pos:]
    else:
        ext = ''

    return (path, name, ext)
