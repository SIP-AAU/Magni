"""
..
    Copyright (c) 2014-2017, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Script for running Magni test suite when build a conda package.

"""

import os
import subprocess

import magni

path, name, ext = magni.utils.split_path(magni.__file__)
test_script = path + 'tests' + os.sep + 'run_tests.py'

subprocess.call(['python', test_script, '--no-coverage', '--skip',
                 'ipynb_examples.py', 'build_docs.py', 'style_checks.py'])
