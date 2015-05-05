"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions that may be used to chase data.

Routine listings
----------------
get_main_file_name()
    Function that returns the name of the main file/script
get_main_file_source()
    Function that returns the source code of the main file/script
get_main_source_code()
    Function that returns the 'local' source code of the main file/script
get_stack_trace()
    Function that returns the complete stack trace

"""

from __future__ import absolute_import  # stdlib io is shadowed by . io
from __future__ import division
import contextlib
import inspect
import traceback

try:
    from StringIO import StringIO  # Python 2 byte str (Python 2 only)
except ImportError:
    from io import StringIO  # Python 3 unicode str (both Py2 and Py3)


def get_main_file_name():
    """
    Return the name of the main file/script which called this function.

    This will (if possible) return the name of the main file/script invoked by
    the Python interpreter i.e., the returned name is the name of the file, in
    which the bottom call on the stack is defined. Thus, the main file is still
    returned if the call to this function is buried deep in the code.

    Returns
    -------
    source_file : str
        The name of the main file/script which called this function.

    Notes
    -----
    It may not be possible to associate the call to this function with a main
    file. This is for instance the case if the call is handled by an IPython
    kernel. If no file is found, the returned string will indicate so.

    """

    stack_bottom = inspect.stack()[-1][0]

    try:
        source_file = inspect.getsourcefile(stack_bottom)
    except TypeError as e:
        source_file = 'Failed with TypeError: {!r}'.format(e.args[0])

    if source_file is None:
        source_file = 'Failed to find main file.'

    return source_file


def get_main_file_source():
    """
    Return the source code of the main file/script which called this function.

    This will (if possible) return the source code of the main file/script
    invoked by the Python interpreter i.e., the returned source code is the
    source code of the file, in which the bottom call on the stack is defined.
    Thus, the source code is still returned if the call to this function is
    buried deep in the code.

    Returns
    -------
    source : str
       The source code of the main file/script which called this function.

    Notes
    -----
    It may not be possible to associate the call to this function with a main
    file. This is for instance the case if the call is handled by an IPython
    kernel. If no file is found, the returned string will indicate so.

    """

    stack_bottom = inspect.stack()[-1][0]

    try:
        source_file = inspect.getsourcefile(stack_bottom)
    except TypeError as e:
        source_file = 'Failed with TypeError: {!r}'.format(e.args[0])

    if source_file is not None:
        try:
            with open(source_file) as sourcefile:
                source = sourcefile.read()
        except (OSError, IOError) as e:
            source = 'Failed with (OS/IO)Error: {!r}'.format(e.args[0])
    else:
        source = 'Failed to find main file.'

    return source


def get_main_source():
    """
    Return the local source code of the main file which called this function.

    This will (if possible) return the part of the source code of the main
    file/script surrounding (local to) the call to this function. The returned
    source code is a part of the source code of the file, in which the bottom
    call on the stack is defined. Thus, the source code is still returned if
    the call to this function is buried deep in the code.

    Returns
    -------
    source : str
        The local source code of the main file/script.

    Notes
    -----
    It may not be possible to associate the call to this function with a main
    file. This is for instance the case if the call is handled by an IPython
    kernel. If no file is found, the returned string will indicate so.

    """

    stack_bottom = inspect.stack()[-1][0]

    try:
        source = inspect.getsource(stack_bottom)
    except (OSError, IOError) as e:
        source = 'Failed with (OS/IO)Error: {!r}'.format(e.args[0])

    if source is None:
        source = 'Failed to find main file.'

    return source


def get_stack_trace():
    """
    Return the complete stack trace that led to the call to this function.

    A (pretty) print version of the stack trace is returned.

    Returns
    -------
    printed_stack : str
       The stack trace that led to the call to this function.

    """

    with contextlib.closing(StringIO()) as str_file:
        traceback.print_stack(file=str_file)  # Needs str (not unicode) on Py2
        printed_stack = str_file.getvalue()

    return printed_stack
