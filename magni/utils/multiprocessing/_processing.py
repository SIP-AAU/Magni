"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the process function.

Routine listings
----------------
process(func, namespace={}, args_list=None, kwargs_list=None, maxtasks=None)
    Map multiple function calls to multiple processors.

See Also
--------
magni.utils.multiprocessing.config : Configuration options.

"""

from __future__ import division
import multiprocessing as mp
import os
import sys
import traceback
import types

# disable mkl multiprocessing to avoid conflicts with manual multiprocessing
try:
    import mkl
    _get_num_threads = mkl.get_max_threads
    _set_num_threads = mkl.set_num_threads
except ImportError:
    def _get_num_threads():
        return 0

    def _set_num_threads(n):
        pass

from magni.utils.multiprocessing import config as _config
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


def process(func, namespace={}, args_list=None, kwargs_list=None,
            maxtasks=None):
    """
    Map multiple function calls to multiple processors.

    For each entry in args_list and kwargs_list, a task is formed which is used
    for a function call of the type `func(*args, **kwargs)`.

    Parameters
    ----------
    func : function
        A function handle to the function which the calls should be mapped to.
    namespace : dict, optional
        A dict whose keys and values should be globally available in func (the
        default is an empty dict).
    args_list : list or tuple, optional
        A sequence of argument lists for the function calls (the default is
        None, which implies that no arguments are used in the calls).
    kwargs_list : list or tuple, optional
        A sequence of keyword argument dicts for the function calls (the
        default is None, which implies that no keyword arguments are used in
        the calls).
    maxtasks : int, optional
        The maximum number of tasks of a process before it is replaced by a new
        process (the default is None, which implies that processes are not
        replaced).

    Returns
    -------
    results : list
        A list with the results from the function calls.

    See Also
    --------
    magni.utils.multiprocessing.config : Configuration options.

    Notes
    -----
    If the `workers` configuration option is equal to 0, map is used.
    Otherwise, the map functionality of a multiprocessing worker pool is used.

    Reasons for using this function over map or standard multiprocessing:

    - Simplicity of the code over standard multiprocessing.
    - Simplicity in switching between single- and multiprocessing.
    - The use of both arguments and keyword arguments in the function calls.
    - The reporting of exceptions before termination.
    - The possibility of terminating multiprocessing with a single interrupt.

    Examples
    --------
    An example of how to use args_list, and kwargs_list:

    >>> from magni.utils.multiprocessing._processing import process
    >>> def calculate(a, b, op='+'):
    ...     if op == '+':
    ...         return a + b
    ...     elif op == '-':
    ...         return a - b
    ...
    >>> args_list = [[5, 7], [9, 3]]
    >>> kwargs_list = [{'op': '+'}, {'op': '-'}]
    >>> process(calculate, args_list=args_list, kwargs_list=kwargs_list)
    [12, 6]

    """

    @_decorate_validation
    def validate_input():
        _generic('func', 'function')
        _generic('namespace', 'mapping')
        _levels('args_list', (_generic(None, 'collection', ignore_none=True),
                              _generic(None, 'explicit collection')))
        _levels('kwargs_list', (_generic(None, 'collection', ignore_none=True),
                                _generic(None, 'mapping')))

        if args_list is None and kwargs_list is None:
            msg = ('The value of >>args_list<<, {!r}, and/or the value of '
                   '>>kwargs_list<<, {!r}, must be different from {!r}.')
            raise ValueError(msg.format(args_list, kwargs_list, None))
        elif args_list is not None and kwargs_list is not None:
            if len(args_list) != len(kwargs_list):
                msg = ('The value of >>len(args_list)<<, {!r}, must be equal '
                       'to the value of >>len(kwargs_list)<<, {!r}.')
                raise ValueError(msg.format(len(args_list), len(kwargs_list)))

        _numeric('maxtasks', 'integer', range_='(0;inf)', ignore_none=True)

    validate_input()

    if args_list is None:
        args_list = [[] for dct in kwargs_list]

    if kwargs_list is None:
        kwargs_list = [{} for lst in args_list]

    tasks = [func for lst in args_list]
    tasks = list(zip(tasks, args_list, kwargs_list))

    if _config['workers'] == 0:
        _process_init(func, namespace)
        results = list(map(_process_worker, tasks))
    else:
        if os.name == 'nt' and sys.version_info.major == 2:
            raise NotImplementedError('This function is not available under '
                                      'Windows with Python 2.')

        try:
            num_threads = _get_num_threads()
            _set_num_threads(1)
            workers = mp.Pool(_config['workers'], _process_init,
                              (func, namespace), maxtasks)
            results = workers.map(_process_worker, tasks, chunksize=1)
        finally:
            workers.close()
            workers.join()
            _set_num_threads(num_threads)

    return results


def _process_init(func, namespace):
    """
    Initialise the process by making global variables available to it.

    Parameters
    ----------
    func : function
        A function handle to the function which the calls should be mapped to.
    namespace : dict
        A dict whose keys and values should be globally available in func.

    """

    func.__globals__.update(namespace)


def _process_worker(fak_tuple):
    """
    Unpack and map a task to the function.

    Parameters
    ----------
    fak_tuple : tuple
        A tuple (func, args, kwargs) containing the parameters listed below.
    func : function
        A function handle to the function which the calls should be mapped to.
    args : list or tuple
        The sequence of arguments that should be unpacked and passed.
    kwargs : list or tuple
        The sequence of keyword arguments that should be unpacked and passed.

    Notes
    -----
    If an exception is raised in `func`, the stacktrace of that exception is
    printed since the exception is otherwise silenced until every task has been
    executed when using multiple workers.

    Also, a workaround has been implemented to allow KeyboardInterrupts to
    interrupt the current tasks and all remaining tasks. This is done by
    setting a global variable, when catching a KeyboardInterrupt, which is
    checked for every call.

    """

    func, args, kwargs = fak_tuple

    if 'interrupted' not in _process_worker.__globals__:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            _process_worker.__globals__['interrupted'] = True
        except BaseException as e:
            print(traceback.format_exc())

            if _config['silence_exceptions']:
                return e
            else:
                raise RuntimeError('An exception has occured.')
