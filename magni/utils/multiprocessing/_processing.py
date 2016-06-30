"""
..
    Copyright (c) 2014-2016, Magni developers.
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
import warnings

try:
    # The concurrent.futures was first added to Python in version 3.2
    # A backport for Python 2 is available at https://pythonhosted.org/futures/
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    _futures_available = True
except ImportError:
    _futures_available = False

try:
    # disable mkl multiprocessing to avoid conflicts with manual
    # multiprocessing
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
    Map multiple function calls to multiple processes.

    For each entry in args_list and kwargs_list, a task is formed which is used
    for a function call of the type `func(*args, **kwargs)`. Each task is
    executed in a seperate process using the concept of a processing pool.

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

    Raises
    ------
    BrokenPoolError
        If using the `concurrent.futures` module and one or more workers
        terminate abrubtly with the automatic broken pool restart funtionality
        disabled.

    See Also
    --------
    magni.utils.multiprocessing.config : Configuration options.

    Notes
    -----
    If the `workers` configuration option is equal to 0, map is used.
    Otherwise, the map functionality of a processing pool is used.

    Reasons for using this function over map or standard multiprocessing:

    - Simplicity of the code over standard multiprocessing.
    - Simplicity in switching between single- and multiprocessing.
    - The use of both arguments and keyword arguments in the function calls.
    - The reporting of exceptions before termination.
    - The possibility of terminating multiprocessing with a single interrupt.
    - The option of automatically restarting a broken process pool.

    As of Python 3.2, two different, though quite similar, modules exist in the
    standard library for managing processing pools: `multiprocessing` and
    `concurrent.futures`. According to Python core contributor Jesse Noller,
    the plan is to eventually make concurrent.futures the only interface to the
    high level processing pools (futures), whereas multiprocessing is supposed
    to serve more low level needs for individually handling processes, locks,
    queues, etc. (see https://bugs.python.org/issue9205#msg132661). As of
    Python 3.5, both the `multiprocessing.Pool` and
    `concurrent.futures.ProcessPoolExecutor` serve almost the same purpose and
    provide very similar interfaces. The main differences between the two are:

    - The option of using a worker initialiser is only available in
      `multiprocessing`.
    - The option of specifing a maximum number of tasks for a worker to execute
      before being replaced to free up ressources (the maxtasksperchild option)
      is only available in `multiprocessing`.
    - The option of specifying a context is only available in `multiprocessing`
    - "Reasonable" handling of abrubt worker termination and exceptions is only
       available in `concurrent.futures`.

    Particularly, the "reasonable" handling of a broken process pool may be a
    strong argument to prefer `concurrent.futures` over `multiprocessing`. The
    matter of handling a broken process pool has been extensively discussed in
    https://bugs.python.org/issue9205 which led to the fix for
    `concurrent.futures`. A similar fix for `multiprocessing` has been proposed
    in https://bugs.python.org/issue22393.

    Both the `multiprocessing` and `concurrent.futures` interfaces are
    available for use with this function. If the configuration parameter
    `prefer_futures` is set to True and the `concurrent.futures` module is
    available, this is used. Otherwise, the `multiprocessing` module is used. A
    Python 2 backport of `concurrent.futures` is available at
    https://pythonhosted.org/futures/.

    When using `concurrent.futures`, the `maxtasks`, `namespace`, and
    `init_args` are ignored since these are not supported by that module. It
    seems that `concurrent.futures` works as if maxtasks==1, however this is
    based purely on emprical observations. The `init_args` functionality may be
    added later on if an initialiser is added to `concurrent.futures` - see
    http://bugs.python.org/issue21423. If the `max_broken_pool_restarts`
    configuration parameter is set to a value different from 0, the Pool is
    automatically restarted and the tasks are re-run should a broken pool be
    encountered. If `max_broken_pool_restarts` is set to 0, a BrokenPoolError
    is raised should a broken pool be encountered.

    When using `multiprocessing`, the `max_broken_pool_restarts` is ignored
    since the BrokenPoolError handling has not yet been implemented for the
    `multiprocessing.Pool` - see https://bugs.python.org/issue22393 as well as
    https://bugs.python.org/issue9205.

    Examples
    --------
    An example of how to use args_list, and kwargs_list:

    >>> import magni
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

    or the same example preferring `concurrent.futures` over `multiprocessing`:

    >>> magni.utils.multiprocessing.config['prefer_futures'] = True
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

    tasks = _get_tasks(func, args_list, kwargs_list)

    if _config['workers'] == 0:
        _process_init(func, namespace)
        results = list(map(_process_worker, tasks))
    else:
        if os.name == 'nt':
            raise NotImplementedError('This function is not available under '
                                      'Windows.')

        if _futures_available and _config['prefer_futures']:
            map_ = _map_using_futures
        else:
            map_ = _map_using_mppool

        try:
            num_threads = _get_num_threads()
            _set_num_threads(1)
            results = map_(_process_worker, tasks, (func, namespace), maxtasks,
                           _config['max_broken_pool_restarts'])
        finally:
            _set_num_threads(num_threads)

    return results


def _get_tasks(func, args_list, kwargs_list):
    """
    Prepare a list of tasks.

    Parameters
    ----------
    func : function
        A function handle to the function which the calls should be mapped to.
    args_list : list or tuple, optional
        A sequence of argument lists for the function calls (the default is
        None, which implies that no arguments are used in the calls).
    kwargs_list : list or tuple, optional
        A sequence of keyword argument dicts for the function calls (the
        default is None, which implies that no keyword arguments are used in
        the calls).

    Returns
    -------
    tasks : list
        The list of tasks.

    """

    if args_list is None:
        args_list = [() for dct in kwargs_list]

    if kwargs_list is None:
        kwargs_list = [{} for lst in args_list]

    tasks = [func for lst in args_list]
    tasks = list(zip(tasks, args_list, kwargs_list))

    return tasks


def _map_using_futures(func, tasks, init_args, maxtasks,
                       max_broken_pool_restarts):
    """
    Map a set of `tasks` to `func` and run them in parallel using a futures.

    If `max_broken_pool_restarts` is different from 0, the tasks must be an
    explicit collection, e.g. a list or tuple, for the restart to work. If an
    exception occurs in one of the function calls, the process pool terminates
    ASAP and re-raises the first exception that occurred. All exceptions that
    may have occurred in the workers are available as the last element in the
    exceptions args.

    Parameters
    ----------
    func : function
        A function handle to the function which the calls should be mapped to.
    tasks : iterable
        The list of tasks to use as arguments in the function calls.
    maxtasks : int
        The maximum number of tasks of a process before it is replaced by a new
        process. If set to None, the process is never replaced.
    init_args : tuple
        The (func, namespace) tuple for the _process_init initialisation
        function.
    max_broken_pool_restarts : int or None
        The maximum number of attempts at restarting the process pool upon a
        BrokenPoolError. If set to None, the process pool may restart
        indefinitely.

    Returns
    -------
    results : list
        The list of results from the map operation.

    Notes
    -----
    The `maxtasks`, `namespace`, and `init_args` are ignored since these are
    not supported by `concurrent.futures`. It seems that `concurrent.futures`
    works as if maxtasks==1, however this is unclear. The `init_args`
    functionality may be added if an initialiser is added to
    `concurrent.futures` - see http://bugs.python.org/issue21423.

    """

    try:
        workers = ProcessPoolExecutor(max_workers=_config['workers'])
        futures = [workers.submit(func, task) for task in tasks]
        concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_EXCEPTION)
        return [future.result() for future in futures]
    except BrokenProcessPool as e:
        workers.shutdown()
        base_msg = 'A BrokenProcessPool was encountered. '

        if max_broken_pool_restarts is None:
            msg = base_msg + 'Restarting the process pool.'
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return _map_using_futures(func, tasks, init_args, maxtasks, None)
        elif max_broken_pool_restarts > 0:
            new_restart_count = max_broken_pool_restarts - 1
            msg = (base_msg + 'Restarting the process pool with ' +
                   'max_broken_pool_restarts={}.').format(new_restart_count)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return _map_using_futures(
                func, tasks, init_args, maxtasks, new_restart_count)
        else:
            msg = (base_msg + 'Giving up on restarting the process pool ' +
                   'and re-raising.')
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            raise
    except BaseException as e:
        worker_exceptions = [future.exception() for future in futures]
        msg = ('An exception occurred in one or more workers. ' +
               'Re-raising with all exceptions appended to the current ' +
               'exceptions arguments.')
        warnings.warn(msg, RuntimeWarning)
        e.args = e.args + tuple(worker_exceptions)
        raise
    finally:
        workers.shutdown()


def _map_using_mppool(func, tasks, init_args, maxtasks,
                      max_broken_pool_restarts):
    """
    Map a set of `tasks` to `func` and run them in parallel via multiprocessing

    Parameters
    ----------
    func : function
        A function handle to the function which the calls should be mapped to.
    tasks : iterable
        The list of tasks to use as arguments in the function calls.
    maxtasks : int
        The maximum number of tasks of a process before it is replaced by a new
        process. If set to None, the process is never replaced.
    init_args : tuple
        The (func, namespace) tuple for the _process_init initialisation
        function.
    max_broken_pool_restarts : int or None
        The maximum number of attempts at restarting the process pool upon a
        BrokenPoolError. If set to None, the process pool may restart
        indefinitely.

    Returns
    -------
    results : list
        The list of results from the map operation.

    Notes
    -----
    The `max_broken_pool_restarts` is ignored since the BrokenPoolError
    handling has not yet been implemented in the multiprocessing.Pool - see
    https://bugs.python.org/issue22393 and https://bugs.python.org/issue9205.

    """

    try:
        workers = mp.Pool(
            _config['workers'], _process_init, init_args, maxtasks)
        return workers.map(func, tasks, chunksize=1)
    finally:
        workers.close()
        workers.join()


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
            elif _config['re_raise_exceptions']:
                raise
            else:
                raise RuntimeError('An exception has occured.')
