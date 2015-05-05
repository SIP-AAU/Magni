"""
..
    Copyright (c) 2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests for `magni.reproducibility`.

**Testing Strategy**

Each annotation is a dict which must have specific keys. It is tested that
these keys are as expected.

Each chase is a string. It is tested that this string has certain properties.

"""

from __future__ import division
import inspect
import os
import subprocess
import unittest

from magni.reproducibility import _annotation, _chase


class TestAnnotations(unittest.TestCase):
    """
    Test of annotations.

    """

    def setUp(self):
        self.ref_conda_info_keys = ['channels', 'conda_version', 'config_file',
                                    'default_prefix', 'envs_dirs',
                                    'is_foreign_system', 'linked_modules',
                                    'modules_info', 'package_cache',
                                    'platform', 'root_prefix', 'status']
        self.ref_datetime_keys = ['pretty_utc', 'status', 'today', 'utcnow']
        self.ref_git_revision_ok_keys = ['branch', 'status', 'tag']
        self.ref_git_revision_nok_keys = ['output', 'returncode', 'status']
        self.ref_magni_config_keys = ['magni.afm.config',
                                      'magni.cs.phase_transition.config',
                                      'magni.cs.reconstruction.iht.config',
                                      'magni.cs.reconstruction.it.config',
                                      'magni.cs.reconstruction.sl0.config',
                                      'magni.utils.multiprocessing.config',
                                      'status']
        self.ref_magni_info_keys = ['help_magni']
        self.ref_platform_info_keys = ['libc', 'linux', 'mac_os', 'machine',
                                       'node', 'processor', 'python',
                                       'release', 'status', 'system',
                                       'version', 'win32']

    def test_get_conda_info(self):
        conda_info = _annotation.get_conda_info()
        conda_info_keys = sorted(list(conda_info.keys()))

        self.assertTrue('status' in conda_info_keys)

        if conda_info['status'] != 'Failed':
            self.assertListEqual(self.ref_conda_info_keys, conda_info_keys)
            self.assertTrue(conda_info['status'] == 'Succeeded')

    def test_datetime(self):
        datetime_ = _annotation.get_datetime()
        datetime_keys = sorted(list(datetime_.keys()))

        self.assertListEqual(self.ref_datetime_keys, datetime_keys)
        self.assertTrue(datetime_['status'] == 'Succeeded')

    def test_get_git_revision(self):
        try:
            subprocess.check_output(['git', '--version'])
        except subprocess.CalledProcessError as e:
            pass
        except OSError as e:
            pass
        else:
            git_revision = _annotation.get_git_revision()
            git_revision_keys = sorted(list(git_revision.keys()))

            self.assertTrue('status' in git_revision_keys)

            if git_revision['status'] == 'Succeeded':
                self.assertListEqual(self.ref_git_revision_ok_keys,
                                     git_revision_keys)
            else:
                self.assertListEqual(self.ref_git_revision_nok_keys,
                                     git_revision_keys)

    def test_get_magni_config(self):
        magni_config = _annotation.get_magni_config()
        magni_config_keys = sorted(list(magni_config.keys()))

        self.assertTrue('status' in magni_config_keys)
        self.assertTrue(magni_config['status'] == 'Succeeded')
        self.assertListEqual(self.ref_magni_config_keys, magni_config_keys)

    def test_get_magni_info(self):
        magni_info = _annotation.get_magni_info()
        magni_info_keys = sorted(list(magni_info.keys()))

        self.assertListEqual(self.ref_magni_info_keys, magni_info_keys)

    def test_get_platform_info(self):
        platform_info = _annotation.get_platform_info()
        platform_info_keys = sorted(list(platform_info.keys()))

        self.assertFalse('failed' in platform_info['status'])
        self.assertListEqual(self.ref_platform_info_keys, platform_info_keys)


class TestChases(unittest.TestCase):
    """
    Test of chases.

    """

    def setUp(self):
        self.cur_dir = os.getcwd()
        try:
            # chdir in run_tests.py workaround
            os.chdir(inspect.stack()[-2][0].f_locals['cur_dir'])
        except KeyError:
            pass

    def tearDown(self):
        os.chdir(self.cur_dir)

    def test_get_main_file_name(self):
        main_file_name = _chase.get_main_file_name()
        self.assertIsInstance(main_file_name, str)
        self.assertNotEqual(main_file_name, '')
        self.assertNotIn('Failed', main_file_name)

    def test_get_main_file_source(self):
        main_file_source = _chase.get_main_file_source()
        self.assertIsInstance(main_file_source, str)
        self.assertNotEqual(main_file_source, '')
        self.assertNotIn('Failed', main_file_source)

    def test_get_main_source(self):
        main_source = _chase.get_main_source()
        self.assertIsInstance(main_source, str)
        self.assertNotEqual(main_source, '')
        self.assertNotIn('Failed', main_source)

    def test_get_stack_trace(self):
        stack_trace = _chase.get_stack_trace()
        self.assertIsInstance(stack_trace, str)
        self.assertNotEqual(stack_trace, '')
        self.assertNotIn('Failed', stack_trace)
