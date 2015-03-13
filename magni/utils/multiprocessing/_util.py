"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the public class of the magni.utils.multiprocessing
subpackage.

"""

from __future__ import division
import multiprocessing as mp

import tables

from magni.utils.validation import decorate_validation as _decorate_validation


_lock = mp.Lock()


class File():
    """
    Control pytables access to hdf5 files when using multiprocessing.

    `File` retains the interface of `tables.open_file` and should only be used
    in 'with' statements (see Examples).

    Parameters
    ----------
    args : tuple
        The arguments that are passed to 'tables.open_file'.
    kwargs : dict
        The keyword arguments that are passed to 'tables.open_file'.

    See Also
    --------
    tables.open_file : The wrapped function.

    Notes
    -----
    Internally the module uses a global lock which is shared amongst all files.
    This solution is simple and does not entail significant overhead. However,
    the wait time introduced when using multiple files at the same time can be
    significant.

    Examples
    --------
    The class is used in the following way:

    >>> from magni.utils.multiprocessing._util import File
    >>> with File('database.hdf5', 'a') as f:
    ...     pass # execute something involving the opened file

    """

    def __init__(self, *args, **kwargs):
        @_decorate_validation
        def validate_input():
            if len(args) == 0 and 'filename' not in kwargs:
                raise ValueError('File must be called with a filename '
                                 'argument.')

        validate_input()

        self._args = args
        self._kwargs = kwargs

        if len(args) > 0:
            self._filename = args[0]
        else:
            self._filename = kwargs['filename']

    def __enter__(self):
        """
        Acquire the global lock before opening and returning the file.

        Returns
        -------
        file : tables.File
            The file specified in the call to `__init__`.

        """

        _lock.acquire()
        self._file = tables.open_file(*self._args, **self._kwargs)
        return self._file

    def __exit__(self, type, value, traceback):
        """
        Release the global lock after closing the file.

        Parameters
        ----------
        type : type
            The type of the exception raised, if any.
        value : Exception
            The exception rasied, if any.
        traceback : traceback
            The traceback of the exception raised, if any.

        """

        self._file.close()
        _lock.release()
