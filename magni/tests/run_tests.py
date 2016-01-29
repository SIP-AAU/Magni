#!/usr/bin/env python

"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Script for running all Magni tests using nosetests

This script identifies TestCases in modules in magni/magni/tests. It then runs
these TestCases in a temporary folder such that auxillary files created by the
tests are automatically removed after their have been run.

usage: run_tests.py [-h] [--keep-output] [--skip [SKIP [SKIP ...]]]

optional arguments:
  -h, --help                   show this help message and exit
  --keep-output                keep test output in the .magnitest directory
  --no-coverage                do not print a coverage report
  --skip [SKIP [SKIP ...]]     name(s) of test module(s) to skip

Additional arguments are passed on to the nose test runner.

"""

from __future__ import division, print_function
import argparse
import nose
import os
import shutil
import sys

try:
    _disp = os.environ['DISPLAY']
except KeyError:
    print('Warning: DISPLAY environment variable not set. ' +
          'Falling back to "Agg" backend for matplotlib.')
    import matplotlib as mpl
    mpl.use('Agg')


def run_tests(test_modules, args, nose_argv):
    """
    Run the tests using nosetests.

    Parameters
    ----------
    test_modules : list or tuple
        The modules (as strings) containing tests to run.
    args : list or tuple
        The no_coverage and keep_output argument flags.
    nose_argv : list or tuple
        The arguments to pass to nosetests.

    """

    # Create a temporary folder to run tests in
    tmp_test_dir = path + '.magnitests'
    print(('Creating temporary directory for test output: {}... ')
          .format(tmp_test_dir), end='')
    cur_dir = os.getcwd()
    try:
        os.mkdir(tmp_test_dir)
    except OSError:
        pass

    print('done')
    os.chdir(tmp_test_dir)

    # Run the tests according to specification
    print('Running tests defined in:')
    for module in test_modules:
        print('  - {}'.format(module))

    fixed_nose_argv = ['--no-byte-compile', '--nologcapture']

    if not args.no_coverage:
        fixed_nose_argv.extend(
            ['--with-coverage', '--cover-html', '--cover-package=magni',
             '--cover-branches'])

    nose_test_paths = [path + test_module for test_module in test_modules]

    success = nose.run(defaultTest=nose_test_paths,
                       argv=fixed_nose_argv + nose_argv)
    os.chdir(cur_dir)

    # Handle test output
    if args.keep_output:
        print('Test output may be found in: {}'.format(tmp_test_dir))
    else:
        print('Deleting temporary test output... ', end='')
        shutil.rmtree(tmp_test_dir)
        print('done')

    # Exit 0 on success as required by Travis-CI
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    # Arguments parsing
    arg_parser = argparse.ArgumentParser(description='Magni test suite runner')
    arg_parser.add_argument('--keep-output', action='store_true',
                            help=('Keep test output in the' +
                                  '.magnitest directory'))
    arg_parser.add_argument('--no-coverage', action='store_true',
                            help=('Do not show coverage report'))
    arg_parser.add_argument('--skip', type=str, nargs='*', default=[],
                            help='Name(s) of test module(s) to skip')
    args, nose_argv = arg_parser.parse_known_args()

    # Identify modules containing tests
    excludes = ['run_tests.py', 'dep_check.py', '__init__.py'] + args.skip
    real_path = os.path.realpath(str(__file__))
    path_pos = str.rfind(real_path, os.path.sep) + 1
    path, name = real_path[:path_pos], real_path[path_pos:]
    test_modules = sorted([module for module in os.listdir(path)
                           if module[-3:] == '.py' and module not in excludes])

    run_tests(test_modules, args, nose_argv)
