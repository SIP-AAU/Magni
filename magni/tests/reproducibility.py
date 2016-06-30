"""
..
    Copyright (c) 2015-2016, Magni developers.
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
import json
import os
import subprocess
import unittest

import tables as tb

from magni.reproducibility import _annotation, _chase, io


class TestAnnotations(unittest.TestCase):
    """

    Test of annotations.

    """

    def setUp(self):
        self.ref_conda_info_keys = ['channels', 'conda_version', 'config_file',
                                    'default_prefix', 'env_export',
                                    'envs_dirs', 'is_foreign_system',
                                    'linked_modules', 'modules_info',
                                    'package_cache', 'platform', 'root_prefix',
                                    'status']
        self.ref_datetime_keys = ['pretty_utc', 'status', 'today', 'utcnow']
        self.ref_git_revision_ok_keys = ['branch', 'remote', 'status', 'tag']
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
        self.file_ = 'file_hashes_test.txt'
        self.true_file_hashes = {
            'md5sum': '7af13f5773a4f06d1307cb2aeb9e08b8',
            'sha256sum': ('011e092f282c1dbba5fc3d58cff32e20' +
                          '753f6749f24bc58bd39da24984ac3391')}
        self.assertFalse(os.path.exists(self.file_))
        with open(self.file_, mode='w') as file_:
            file_.write('Some dummy text')

    def tearDown(self):
        if os.path.exists(self.file_):
            os.remove(self.file_)

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

    def test_file_hashes(self):
        file_hashes = _annotation.get_file_hashes(self.file_, blocksize=1)
        self.assertEqual(self.true_file_hashes, file_hashes)

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
            git_revision_here = _annotation.get_git_revision('.')
            git_revision_here_keys = sorted(list(git_revision_here.keys()))

            self.assertTrue('status' in git_revision_keys)
            self.assertTrue('status' in git_revision_here_keys)

            if git_revision['status'] == 'Succeeded':
                self.assertListEqual(self.ref_git_revision_ok_keys,
                                     git_revision_keys)
            else:
                self.assertListEqual(self.ref_git_revision_nok_keys,
                                     git_revision_keys)

        with self.assertRaises(OSError):
            _annotation.get_git_revision('this/should/not/be/a/directory')

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


class TestIO(unittest.TestCase):
    """
    Tests of IO functions.

    """

    def setUp(self):
        self.h5_name = 'db.hdf5'
        self.assertFalse(os.path.exists(self.h5_name))
        self.ref_annotation_keys = ['conda_info', 'datetime', 'git_revision',
                                    'magni_config', 'magni_info',
                                    'platform_info']
        self.ref_chase_keys = ['main_file_name', 'main_file_source',
                               'main_source', 'stack_trace']

    def tearDown(self):
        if os.path.exists(self.h5_name):
            os.remove(self.h5_name)

    def test_annotations_roundtrip(self):
        with tb.File(self.h5_name, mode='a') as h5_file:
            io.annotate_database(h5_file)
            annotation_keys = sorted(io.read_annotations(h5_file).keys())
            self.assertEqual(annotation_keys, self.ref_annotation_keys)
            io.remove_annotations(h5_file)
            with self.assertRaises(tb.NoSuchNodeError):
                io.read_annotations(h5_file)

    def test_annotations_exceptions(self):
        with tb.File(self.h5_name, mode='a') as h5_file:
            # No annotations
            io.remove_annotations(h5_file)  # Pass if no annotations
            with self.assertRaises(tb.NoSuchNodeError):
                io.read_annotations(h5_file)

            # Non-compliant annotation
            annotations_group = h5_file.create_group('/', 'annotations')
            h5_file.create_array(annotations_group, 'fail_ann',
                                 obj='[fail)'.encode())
            h5_file.flush()
            with self.assertRaises(ValueError):
                io.read_annotations(h5_file)

            # Already annotated database
            with self.assertRaises(tb.NodeError):
                io.annotate_database(h5_file)

    def test_chases_roundtrip(self):
        with tb.File(self.h5_name, mode='a') as h5_file:
            io.chase_database(h5_file)
            chase_keys = sorted(io.read_chases(h5_file).keys())
            self.assertEqual(chase_keys, self.ref_chase_keys)
            io.remove_chases(h5_file)
            with self.assertRaises(tb.NoSuchNodeError):
                io.read_chases(h5_file)

    def test_chases_exceptions(self):
        with tb.File(self.h5_name, mode='a') as h5_file:
            # No chases
            io.remove_chases(h5_file)  # Pass if no chases
            with self.assertRaises(tb.NoSuchNodeError):
                io.read_chases(h5_file)

            # Non-complient chase
            chases_group = h5_file.create_group('/', 'chases')
            h5_file.create_array(chases_group, 'fail_chase',
                                 obj='[fail)'.encode())
            h5_file.flush()
            with self.assertRaises(ValueError):
                io.read_chases(h5_file)

            # Already chased database
            with self.assertRaises(tb.NodeError):
                io.chase_database(h5_file)

    def test_create_database(self):
        io.create_database(self.h5_name)
        with tb.File(self.h5_name, mode='a') as h5_file:
            annotation_keys = sorted(io.read_annotations(h5_file).keys())
            self.assertEqual(annotation_keys, self.ref_annotation_keys)
            chase_keys = sorted(io.read_chases(h5_file).keys())
            self.assertEqual(chase_keys, self.ref_chase_keys)

        # Overwrite test
        io.create_database(self.h5_name)
        with self.assertRaises(IOError):
            io.create_database(self.h5_name, overwrite=False)

    def test_write_custom_annotation(self):
        with tb.File(self.h5_name, mode='a') as h5_file:
            # Roundtrip
            io.write_custom_annotation(h5_file, 'test_ann', 'a test')

            self.assertEqual(
                'a test',
                json.loads(h5_file.root.annotations.test_ann.read().decode()))

            io.remove_annotations(h5_file)

            with self.assertRaises(tb.NoSuchNodeError):
                h5_file.root.annotations.test_ann.read()

            # Invalid annotation
            with self.assertRaises(TypeError):
                io.write_custom_annotation(h5_file, 'fail_ann', lambda x: x)

            # Annotations subgroup
            io.write_custom_annotation(h5_file, 'test_ann', 'a test',
                                       annotations_sub_group='test_group')

            self.assertEqual(
                'a test',
                json.loads(
                    h5_file.root.annotations.test_group.test_ann.read(
                        ).decode()
                ))

            io.remove_annotations(h5_file)

            with self.assertRaises(tb.NoSuchNodeError):
                h5_file.root.annotations.test_group.test_ann.read()

            # Writing to existing annotation
            io.write_custom_annotation(h5_file, 'test_ann', 'a test')
            with self.assertRaises(tb.NodeError):
                io.write_custom_annotation(h5_file, 'test_ann', 'a test')

    def test_recursive_annotation_read(self):
        with tb.File(self.h5_name, mode='a') as h5_file:
            io.annotate_database(h5_file)
            io.write_custom_annotation(
                h5_file, 'test_ann_1', 'a test 1',
                annotations_sub_group='zzz_test_group_1')
            io.write_custom_annotation(
                h5_file, 'test_ann_2', 'a test 2',
                annotations_sub_group='zzz_test_group_2')
            io.write_custom_annotation(
                h5_file, 'test_ann_11', 'a test 11',
                annotations_sub_group='zzz_test_group_1/zzz_test_group_11')

            annotations = io.read_annotations(h5_file)
            annotation_lvl_0_keys = sorted(annotations.keys())
            self.assertEqual(
                annotation_lvl_0_keys,
                self.ref_annotation_keys +
                ['zzz_test_group_1', 'zzz_test_group_2'])
            annotation_lvl_01_keys = sorted(annotations['zzz_test_group_1'])
            self.assertEqual(
                annotation_lvl_01_keys,
                ['test_ann_1', 'zzz_test_group_11'])
            annotation_lvl_02_keys = sorted(annotations['zzz_test_group_2'])
            self.assertEqual(annotation_lvl_02_keys, ['test_ann_2'])
            annotation_lvl_11_keys = sorted(
                annotations['zzz_test_group_1']['zzz_test_group_11'])
            self.assertEqual(annotation_lvl_11_keys, ['test_ann_11'])
