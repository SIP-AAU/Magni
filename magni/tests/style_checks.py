"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module for wrapping style checks of the Magni source code.

"""

from __future__ import division
import contextlib
import os
import re
import unittest
try:
    from StringIO import StringIO  # Python 2 byte str (Python 2 only)
except ImportError:
    from io import StringIO  # Python 3 unicode str (both Py2 and Py3)

import pep8
import pyflakes.api
import radon.cli
import radon.complexity

import magni


class TestStyleConformance(unittest.TestCase):
    """
    Test of code conformance to style guides.

    """

    def setUp(self):
        """
        Identify magni source files.

        """

        path, name, ext = magni.utils.split_path(__file__)
        self.magni_base_path = path.rsplit(os.sep, 2)[0]
        self.exclude_patterns = ['__pycache__', 'tests']

        self.source_files = [os.path.join(dirpath, filename)
                             for dirpath, dirnames, filenames in
                             os.walk(self.magni_base_path)
                             for filename in filenames
                             if filename[-3:] == '.py' and
                             not any(
                                 [exclude_pattern in dirpath
                                  for exclude_pattern in
                                  self.exclude_patterns])]

    def test_pep8_conformance(self):
        """
        Tests for PEP8 conformance.

        Based on the example from the official documentation
        (http://pep8.readthedocs.org/en/latest/advanced.html)

        """

        style = pep8.StyleGuide()
        result = style.check_files(self.source_files)
        self.assertEqual(result.total_errors, 0)

    def test_pyflakes_conformance(self):
        """
        Tests for Python source code errors using Pyflakes

        """

        pyflakes_ignores = ['unable to detect undefined names',
                            'imported but unused',
                            'is assigned to but nevenr used',
                            'redefinition of unused']
        ignore_re = re.compile('|'.join(pyflakes_ignores))

        with contextlib.closing(StringIO()) as out:
            with contextlib.closing(StringIO()) as err:

                reporter = pyflakes.reporter.Reporter(out, err)
                for file in self.source_files:
                    pyflakes.api.checkPath(file, reporter=reporter)

                out_value = out.getvalue()
                err_value = err.getvalue()

        out_filtered = [line for line in out_value.splitlines()
                        if not ignore_re.search(line)]

        self.assertEqual(len(err_value), 0,
                         msg='Pyflakes errors:\n {!s}'.format(err_value))

        self.assertEqual(len(out_filtered), 0,
                         msg='Pyflake warnings:\n {!s}'.format(out_filtered))

    def test_CC_conformance(self):
        """
        Tests for Cyclomatic Complexity levels below 10.

        *Corresponds to running from the magni folder*

        radon cc --total-average --no-assert --show-complexity --order SCORE
            --exclude "magni/utils/validation.py" --ignore "magni/tests*" magni

        """

        cc_threshold = 10
        exclude_modules = [os.path.join(
            'utils', 'validation', '_deprecated.py')]
        excludes = ','.join([os.path.join(self.magni_base_path, exclude)
                             for exclude in exclude_modules])

        config = radon.cli.Config(
            min='A',
            max='F',
            exclude=excludes,
            ignore='tests',
            show_complexity=True,
            average=False,
            total_average=True,
            order=radon.complexity.LINES,
            no_assert=True,
            show_closures=False)

        cc_harvester = radon.cli.CCHarvester([self.magni_base_path], config)

        cc_problems = dict()
        for key, val in cc_harvester._to_dicts().items():
            for k, element in enumerate(val):
                if element['complexity'] >= cc_threshold:
                    cc_problems['{} -- {}'.format(key, k)] = element

        cc_problems_print = [mod + ' :\n' + '\n'.join(
            [' : '.join([key, str(val)])
             for key, val in sorted(cc_problems[mod].items())])
            for mod in cc_problems]

        radon.cli.log_result(cc_harvester)
        self.assertFalse(
            cc_problems, msg=('Cyclomatic complexity exceeds {}:\n\n'
                              .format(cc_threshold) +
                              '\n\n'.join(cc_problems_print)))
