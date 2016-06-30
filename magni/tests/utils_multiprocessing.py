"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests for `magni.utils.multiprocessing`.

**Testing Strategy**
All elements of the multiprocessing subpackage are tested individually.
Furthermore, the problem identified in https://bugs.python.org/issue25906 is
tested.

Routine Listings
----------------
TestProcessingAuxiliaries(unittest.TestCase)
    Test of auxiliary functions in the processing module.
TestProcessingProcess(unittest.TestCase)
    Test of the process functions in the processing module.
TestUtils(unittest.TestCase)
    Test of the utils module.

"""

from __future__ import division
from contextlib import contextmanager
import os
import signal
import sys
import time
import unittest
import warnings

import magni


class TestProcessingAuxiliaries(unittest.TestCase):
    """
    Test of auxiliary functions in the processing module.

    The following tests are implemented:

    - *test_get_tasks_with_args_and_kwargs*
    - *test_get_tasks_with_args_only*
    - *test_get_tasks_with_kwargs_only*
    - *test_process_init*
    - *test_process_worker_normal*
    - *test_process_worker_except*
    - *test_process_worker_except_re_raise*
    - *test_process_worker_except_silently*

    """

    def setUp(self):
        self.func = _func
        self.except_func = _except_func
        self.args = ['test', 1]
        self.kwargs = [{'mp_test': 1}, {'foo': 'bar'}]

    def tearDown(self):
        magni.utils.multiprocessing.config.reset()

    def test_get_tasks_with_args_and_kwargs(self):
        tasks = magni.utils.multiprocessing._processing._get_tasks(
            self.func, self.args, self.kwargs)

        self.assertEqual(tasks, [(self.func, self.args[0], self.kwargs[0]),
                                 (self.func, self.args[1], self.kwargs[1])])

    def test_get_tasks_with_args_only(self):
        tasks = magni.utils.multiprocessing._processing._get_tasks(
            self.func, self.args, None)

        self.assertEqual(tasks, [(self.func, self.args[0], {}),
                                 (self.func, self.args[1], {})])

    def test_get_tasks_with_kwargs_only(self):
        tasks = magni.utils.multiprocessing._processing._get_tasks(
            self.func, None, self.kwargs)

        self.assertEqual(tasks, [(self.func, (), self.kwargs[0]),
                                 (self.func, (), self.kwargs[1])])

    def test_process_init(self):
        self.assertFalse(
            list(self.kwargs[0].keys())[0] in self.func.__globals__)
        magni.utils.multiprocessing._processing._process_init(
            self.func, self.kwargs[0])
        self.assertTrue(
            list(self.kwargs[0].keys())[0] in self.func.__globals__)

    def test_process_worker_normal(self):
        fak_tuple = (self.func, self.args, self.kwargs[0])
        result = magni.utils.multiprocessing._processing._process_worker(
            fak_tuple)

        self.assertEqual(result, (('test', 1), {'mp_test': 1}))

    def test_process_worker_except(self):
        fak_tuple = (self.except_func, self.args, self.kwargs[0])
        with self.assertRaises(RuntimeError):
            magni.utils.multiprocessing._processing._process_worker(fak_tuple)

    def test_process_worker_except_re_raise(self):
        magni.utils.multiprocessing.config['re_raise_exceptions'] = True
        fak_tuple = (self.except_func, self.args, self.kwargs[0])
        with self.assertRaises(ValueError):
            magni.utils.multiprocessing._processing._process_worker(fak_tuple)

    def test_process_worker_except_silently(self):
        magni.utils.multiprocessing.config['silence_exceptions'] = True
        fak_tuple = (self.except_func, self.args, self.kwargs[0])
        result = magni.utils.multiprocessing._processing._process_worker(
            fak_tuple)

        self.assertIsInstance(result, ValueError)


class TestProcessingProcess(unittest.TestCase):
    """
    Test of the process functions in the processing module.

    The following tests are implemented:

    - *test_special_validation*
    - *test_no_multiprocessing*
    - *test_one_worker_multiprocessing*
    - *test_two_workers_multiprocessing*
    - *test_multiprocessing_multiprocessing*
    - *test_futures_multiprocessing*
    - *test_map_using_mppool*
    - *test_map_using_futures*
    - *test_map_using_futures_exception*
    - *test_map_using_futures_give_up_on_broken_pool*
    - *test_map_using_futures_double_restart_broken_pool*
    - *test_map_using_futures_infinitely_restart_broken_pool*
    - *test_issue_25906*

    """

    def setUp(self):
        self.func = _func
        self.except_func = _except_func
        self.kill_self_func = _kill_self_func
        self.magni_test_var_in_globals_func = _magni_test_var_in_globals_func
        self.work_25906 = _work_25906
        self.args_list = [['test'], [1]]
        self.kwargs_list = [{'mp_test': 1}, {'foo': 'bar'}]
        self.func_true_results = [(('test',), {'mp_test': 1}),
                                  ((1,), {'foo': 'bar'})]
        self.tasks = magni.utils.multiprocessing._processing._get_tasks(
            self.func, self.args_list, self.kwargs_list)
        self.except_tasks = magni.utils.multiprocessing._processing._get_tasks(
            self.except_func, self.args_list, self.kwargs_list)

        self.assertFalse(os.path.exists('join_crash_test.hdf'))

    def tearDown(self):
        magni.utils.multiprocessing.config.reset()
        if os.path.exists('join_crash_test.hdf'):
            os.remove('join_crash_test.hdf')

    def test_special_validation(self):
        with self.assertRaises(ValueError):
            magni.utils.multiprocessing.process(
                self.func, args_list=None, kwargs_list=None)

        with self.assertRaises(ValueError):
            magni.utils.multiprocessing.process(
                self.func, args_list=[self.args_list[0]],
                kwargs_list=self.kwargs_list)

    def test_no_multiprocessing(self):
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 0)
        results = magni.utils.multiprocessing.process(
            self.func, args_list=self.args_list, kwargs_list=self.kwargs_list)

        self.assertEqual(results, self.func_true_results)

    def test_one_worker_multiprocessing(self):
        magni.utils.multiprocessing.config['workers'] = 1
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 1)

        with self._wrap_win_error_handler():
            results = magni.utils.multiprocessing.process(
                self.func, args_list=self.args_list,
                kwargs_list=self.kwargs_list)

            self.assertEqual(results, self.func_true_results)

    def test_two_workers_multiprocessing(self):
        magni.utils.multiprocessing.config['workers'] = 2
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 2)

        with self._wrap_win_error_handler():
            results = magni.utils.multiprocessing.process(
                self.func, args_list=self.args_list,
                kwargs_list=self.kwargs_list)

            self.assertEqual(results, self.func_true_results)

    def test_multiprocessing_multiprocessing(self):
        magni.utils.multiprocessing.config['workers'] = 2
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 2)
        # If multiprocessing is used, the namespace should be honored.

        with self._wrap_win_error_handler():
            results = magni.utils.multiprocessing.process(
                self.magni_test_var_in_globals_func, args_list=self.args_list,
                kwargs_list=self.kwargs_list,
                namespace={'magni_test_var': 'some_value'})

            self.assertEqual(results, [True, True])

    def test_futures_multiprocessing(self):
        magni.utils.multiprocessing.config.update(
            {'workers': 2, 'prefer_futures': True})
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 2)
        self.assertEqual(magni.utils.multiprocessing.config['prefer_futures'],
                         True)

        with self._wrap_win_error_handler():
            try:
                from concurrent.futures.process import BrokenProcessPool
                # If futures is used, the namespace should be ignored.
                results = magni.utils.multiprocessing.process(
                    self.magni_test_var_in_globals_func,
                    args_list=self.args_list, kwargs_list=self.kwargs_list,
                    namespace={'magni_test_var': 'some_value'})

                self.assertEqual(results, [False, False])
            except ImportError:
                # If futures not available, this test is essentially a dupe of
                # test_multiprocessing_multiprocessing
                results = magni.utils.multiprocessing.process(
                    self.magni_test_var_in_globals_func,
                    args_list=self.args_list, kwargs_list=self.kwargs_list,
                    namespace={'magni_test_var': 'some_value'})

                self.assertEqual(results, [True, True])

    def test_map_using_mppool(self):
        magni.utils.multiprocessing.config['workers'] = 2
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 2)

        if os.name != 'nt':
            result = magni.utils.multiprocessing._processing._map_using_mppool(
                magni.utils.multiprocessing._processing._process_worker,
                self.tasks, (self.func, {'foo_var': 'bar'}), 1, 0)

            self.assertEqual(result, self.func_true_results)

    def test_map_using_futures(self):
        magni.utils.multiprocessing.config.update(
            {'workers': 2, 'prefer_futures': True})
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 2)
        self.assertEqual(magni.utils.multiprocessing.config['prefer_futures'],
                         True)

        if os.name != 'nt':
            try:
                from concurrent.futures.process import BrokenProcessPool
                r = magni.utils.multiprocessing._processing._map_using_futures(
                    magni.utils.multiprocessing._processing._process_worker,
                    self.tasks, (self.func, {'foo_var': 'bar'}), 1, 0)

                self.assertEqual(r, self.func_true_results)
            except ImportError:
                pass

    def test_map_using_futures_exception(self):
        magni.utils.multiprocessing.config.update(
            {'workers': 2, 'prefer_futures': True,
             're_raise_exceptions': True})
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 2)
        self.assertEqual(magni.utils.multiprocessing.config['prefer_futures'],
                         True)
        self.assertTrue(
            magni.utils.multiprocessing.config['re_raise_exceptions'])
        mumppw = magni.utils.multiprocessing._processing._process_worker

        if os.name != 'nt':
            try:
                from concurrent.futures.process import BrokenProcessPool
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always')
                    magni.utils.multiprocessing._processing._map_using_futures(
                        mumppw, self.except_tasks,
                        (self.except_func, {'foo_var': 'bar'}), 1, 0)
            except ImportError:
                pass
            except ValueError as e:
                self.assertIsInstance(e.args[1], ValueError)
                self.assertIsInstance(e.args[2], ValueError)
                self.assertEqual(len(ws), 1)
                self.assertIsInstance(ws[0].message, RuntimeWarning)
                self.assertEqual(
                    ws[0].message.args[0],
                    'An exception occurred in one or more workers. ' +
                    'Re-raising with all exceptions appended to ' +
                    'the current exceptions arguments.')

    def test_map_using_futures_give_up_on_broken_pool(self):
        magni.utils.multiprocessing.config.update(
            {'workers': 2, 'prefer_futures': True})
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 2)
        self.assertEqual(magni.utils.multiprocessing.config['prefer_futures'],
                         True)

        if os.name != 'nt':
            try:
                from concurrent.futures.process import BrokenProcessPool
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always')
                    with self.assertRaises(BrokenProcessPool):
                        magni.utils.multiprocessing.process(
                            self.kill_self_func,
                            args_list=[[time.time() + 10]] * 2)
                self.assertEqual(len(ws), 1)
                self.assertIsInstance(ws[0].message, RuntimeWarning)
                self.assertEqual(
                    ws[0].message.args[0],
                    'A BrokenProcessPool was encountered. ' +
                    'Giving up on restarting the process pool and re-raising.')
            except ImportError:
                pass

    def test_map_using_futures_double_restart_broken_pool(self):
        magni.utils.multiprocessing.config.update(
            {'workers': 2, 'prefer_futures': True,
             'max_broken_pool_restarts': 2})
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 2)
        self.assertEqual(magni.utils.multiprocessing.config['prefer_futures'],
                         True)
        self.assertEqual(
            magni.utils.multiprocessing.config['max_broken_pool_restarts'], 2)

        base_msg = 'A BrokenProcessPool was encountered. '
        warn_msgs = [base_msg + restart_msg for restart_msg in [
            'Restarting the process pool with max_broken_pool_restarts=1.',
            'Restarting the process pool with max_broken_pool_restarts=0.',
            'Giving up on restarting the process pool and re-raising.']]

        if os.name != 'nt':
            try:
                from concurrent.futures.process import BrokenProcessPool
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always')
                    with self.assertRaises(BrokenProcessPool):
                        magni.utils.multiprocessing.process(
                            self.kill_self_func,
                            args_list=[[time.time() + 10]] * 2)
                self.assertEqual(len(ws), 3)
                for w, w_msg in zip(ws, warn_msgs):
                    self.assertIsInstance(w.message, RuntimeWarning)
                    self.assertEqual(w.message.args[0], w_msg)
            except ImportError:
                pass

    def test_map_using_futures_infinitely_restart_broken_pool(self):
        magni.utils.multiprocessing.config.update(
            {'workers': 2, 'prefer_futures': True,
             'max_broken_pool_restarts': None})
        self.assertEqual(magni.utils.multiprocessing.config['workers'], 2)
        self.assertEqual(magni.utils.multiprocessing.config['prefer_futures'],
                         True)
        self.assertEqual(
            magni.utils.multiprocessing.config['max_broken_pool_restarts'],
            None)

        if os.name != 'nt':
            try:
                from concurrent.futures.process import BrokenProcessPool
                tasks = [[time.time() + offset] for offset in [1, 2]] * 2
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always')
                    r = magni.utils.multiprocessing.process(
                        self.kill_self_func, args_list=tasks)
                self.assertEqual(r, [task[0] for task in tasks])
                for w in ws:
                    self.assertIsInstance(w.message, RuntimeWarning)
                    self.assertEqual(
                        w.message.args[0],
                        'A BrokenProcessPool was encountered. ' +
                        'Restarting the process pool.')
            except ImportError:
                pass

    def test_issue_25906(self):
        num_workers = 24
        magni.utils.multiprocessing.config.update(
            {'workers': num_workers, 'prefer_futures': True})
        self.assertEqual(
            magni.utils.multiprocessing.config['workers'], num_workers)
        self.assertEqual(magni.utils.multiprocessing.config['prefer_futures'],
                         True)

        if os.name != 'nt':
            try:
                from concurrent.futures.process import BrokenProcessPool
                for iteration in range(10):
                    print('Now processing iteration: {}'.format(iteration))
                    tasks = list(zip(range(num_workers),
                                     num_workers * [iteration]))
                    result = magni.utils.multiprocessing.process(
                        self.work_25906, args_list=tasks, maxtasks=1)

                    self.assertEqual(result, list(tasks))
            except ImportError:
                pass

    @contextmanager
    def _wrap_win_error_handler(self):
        if os.name != 'nt':
            yield
        else:
            with self.assertRaises(NotImplementedError):
                yield


class TestUtils(unittest.TestCase):
    """
    Test of the utils module.

    The following tests are implemented:

    - *test_File_open_normally*
    - *test_File_open_file_using_kwargs*
    - *test_File_missing_filename_error*

    """

    def setUp(self):
        self.filename = 'magni_mp_test_file.hdf'
        self.assertFalse(os.path.exists(self.filename))

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_File_open_normally(self):
        with magni.utils.multiprocessing.File(self.filename, 'w') as h5_file:
            pass

        self.assertTrue(os.path.exists(self.filename))

    def test_File_open_using_kwargs(self):
        with magni.utils.multiprocessing.File(
                filename=self.filename, mode='w') as h5_file:
            pass

        self.assertTrue(os.path.exists(self.filename))

    def test_File_missing_filename_error(self):
        with self.assertRaises(ValueError):
            with magni.utils.multiprocessing.File(mode='w') as h5_file:
                pass


def _except_func(*args, **kwargs):
    """Test function used in some tests."""
    raise ValueError('Some ValueError')


def _func(*args, **kwargs):
    """Test function used in some tests."""
    return args, kwargs


def _kill_self_func(*args, **kwargs):
    """Test function used in some tests."""
    time.sleep(0.5)
    if args[0] <= time.time():
        return args[0]
    else:
        os.kill(os.getpid(), signal.SIGKILL)


def _magni_test_var_in_globals_func(*args, **kwargs):
    """Test function used in some tests."""
    return 'magni_test_var' in globals()


def _work_25906(worker_num, iteration):
    """Test function from https://bugs.python.org/issue25906."""

    with magni.utils.multiprocessing.File(
            'join_crash_test.hdf', mode='a') as h5_file:
        h5_file.create_array('/', 'a{}_{}'.format(worker_num, iteration),
                             obj=(worker_num, iteration))
    print('Worker {} finished writing to HDF table at iteration {}'.format(
        worker_num, iteration))

    return (worker_num, iteration)
