"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module for testing the Magni documentation build process.

"""

from __future__ import division, print_function
import os
import subprocess
import unittest

import magni


class TestDocBuild(unittest.TestCase):
    """
    Test of correct build of magni documentation using Sphinx

    """

    def setUp(self):
        """
        Change directory to magni/doc and run docapi.

        """

        self.stdout = 'Capture of stdout:\n'
        self.stderr = 'Capture of stderr:\n'
        self.cwd = os.getcwd()

        path, name, ext = magni.utils.split_path(__file__)
        magni_base_path = path.rsplit(os.sep, 3)[0]
        os.chdir(os.path.join(magni_base_path, 'doc'))
        try:
            p_sourceclean = subprocess.Popen(['make', 'sourceclean'],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
            p_sourceclean.wait()
            self.stdout += str(p_sourceclean.stdout.read())
            self.stderr += str(p_sourceclean.stderr.read())

            p_docapi = subprocess.Popen(['make', 'docapi'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            p_docapi.wait()
            self.stdout += str(p_docapi.stdout.read())
            self.stderr += str(p_docapi.stderr.read())

        except OSError as e:
            os.chdir(self.cwd)
            raise e

    def tearDown(self):
        """
        Run sourceclean and change directory back to tests directory.

        """
        try:
            p_sourceclean = subprocess.Popen(['make', 'sourceclean'],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
            p_sourceclean.wait()
            self.stdout += str(p_sourceclean.stdout.read())
            self.stderr += str(p_sourceclean.stderr.read())

        finally:
            os.chdir(self.cwd)

    def test_html_build(self):
        """
        Test the invocation of "make html" to build Sphinx html documentation.

        """

        p_html = subprocess.Popen(['make', 'SPHINXOPTS=-W', 'html'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        returncode = p_html.wait()
        self.stdout += str(p_html.stdout.read())
        self.stderr += str(p_html.stderr.read())

        error_msg = ('Magni HTML doc build failed.\n\n' +
                     'Stdout:\n{!s}\n\nStderr:\n{!s}'.format(
                         self.stdout, self.stderr))
        self.assertEqual(returncode, 0, msg=error_msg)
