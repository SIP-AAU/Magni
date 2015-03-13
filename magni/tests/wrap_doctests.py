"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module for wrapping the doctests embedded in the Magni source code docstrings.

"""

from __future__ import division
import contextlib
import doctest
import os
import unittest
try:
    from StringIO import StringIO  # Python 2 byte str (Python 2 only)
except ImportError:
    from io import StringIO  # Python 3 unicode str (both Py2 and Py3)

import numpy as np

import magni


class TestDoctests(unittest.TestCase):
    """
    Test doctests in Magni source code docstrings.

    This TestCase in itself runs a TestSuite of doctests. As such all doctests
    are considered a single TestCase.

    """

    def setUp(self):
        """
        Create a TestSuite of doctests

        """

        self.doctest_suite = unittest.TestSuite()
        path, name, ext = magni.utils.split_path(__file__)
        magni_base_path = path.rsplit(os.sep, 2)[0]
        exclude_patterns = ['__pycache__', 'tests']

        for dirpath, dirnames, filenames in os.walk(magni_base_path):
            for filename in filenames:
                if filename[-3:] == '.py' and not any(
                        [exclude_pattern in dirpath
                         for exclude_pattern in exclude_patterns]):
                    self.doctest_suite.addTest(
                        doctest.DocFileSuite(
                            os.path.join(dirpath, filename),
                            module_relative=False))

    def test_all_doctests(self):
        """
        Run doctests TestSuite

        """

        with contextlib.closing(StringIO()) as stream:
            testRunner = unittest.TextTestRunner(
                stream=stream, descriptions=False, verbosity=1)
            doctest_result = testRunner.run(self.doctest_suite)
            error_msg = '{} failures in doctests.\n {!s}'.format(
                len(doctest_result.failures), stream.getvalue())
            self.assertTrue(doctest_result.wasSuccessful(), msg=error_msg)
