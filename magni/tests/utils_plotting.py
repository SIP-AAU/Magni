"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests for `magni.utils.plotting`.

Routine Listings
----------------
TestColourHandling(unittest.TestCase)
    Test of colour handling functionality.
class TestMatplotlibConfiguration(unittest.TestCase)
    Test of matplotlib configuration.

"""

from __future__ import division
import copy
import unittest
import warnings

try:
    from cycler import Cycler
except ImportError:
    Cycler = list
import matplotlib as mpl
import matplotlib.pyplot as plt

import magni


class TestColourHandling(unittest.TestCase):
    """
    Test of colour handling functionality.

    """

    def testColourCollection(self):
        colour_collection = magni.utils.plotting._ColourCollection(
            {'cc1': ((1, 2, 3), (4, 5, 6), (7, 8, 9)),
             'cc2': ((10, 10, 10), (100, 100, 100))})

        true_cc1 = ((0.0039, 0.0078, 0.0118), (0.0157, 0.0196, 0.0235),
                    (0.0275, 0.0314, 0.0353))
        true_cc2 = ((0.0392, 0.0392, 0.0392), (0.3922, 0.3922, 0.3922))
        self.assertEqual(colour_collection['cc1'], true_cc1)
        self.assertEqual(colour_collection['cc2'], true_cc2)


class TestMatplotlibConfiguration(unittest.TestCase):
    """
    Test of matplotlib configuration.

    """

    def setUp(self):
        self.settings_defaults = copy.deepcopy(magni.utils.plotting._settings)
        self.cmap_default = magni.utils.plotting._cmap

    def tearDown(self):
        magni.utils.plotting.setup_matplotlib(
            settings=self.settings_defaults, cmap=self.cmap_default)
        magni.utils.plotting._settings = self.settings_defaults
        magni.utils.plotting._cmap = self.cmap_default
        for name1, settings in self.settings_defaults.items():
            for name2, setting in settings.items():
                self.assertEqual(
                    self._normalise_rcParam(
                        mpl.rcParams['.'.join([name1, name2])]),
                    self._normalise_rcParam(setting))
        self.assertEqual(plt.get_cmap().name, self.cmap_default)

    def test_default_settings(self):
        magni.utils.plotting.setup_matplotlib()
        for name1, settings in self.settings_defaults.items():
            for name2, setting in settings.items():
                self.assertEqual(
                    self._normalise_rcParam(
                        mpl.rcParams['.'.join([name1, name2])]),
                    self._normalise_rcParam(setting))
        self.assertEqual(plt.get_cmap().name, self.cmap_default)

    def test_builtin_cmap(self):
        magni.utils.plotting.setup_matplotlib(cmap='jet')
        self.assertEqual(plt.get_cmap().name, 'jet')

    def test_custom_cmap(self):
        cmap_clone = plt.get_cmap()
        magni.utils.plotting.setup_matplotlib(cmap='jet')
        magni.utils.plotting.setup_matplotlib(
            cmap=('cmap_clone', cmap_clone))
        self.assertEqual(plt.get_cmap(), cmap_clone)

    def test_double_settings_change(self):
        magni.utils.plotting.setup_matplotlib({'font': {'size': 14}})
        magni.utils.plotting.setup_matplotlib({'figure': {'figsize': (1, 1)}})
        self.assertEqual(mpl.rcParams['font.size'], 14)
        self.assertEqual(mpl.rcParams['figure.figsize'], [1, 1])

    def test_invalid_settings(self):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            magni.utils.plotting.setup_matplotlib({'invalid': {'setting': 0}})
        self.assertEqual(len(ws), 1)
        self.assertIsInstance(ws[0].message, UserWarning)
        self.assertEqual(ws[0].message.args[0], "Setting 'invalid' ignored.")

    def _normalise_rcParam(self, rcParam):
        if isinstance(rcParam, (list, Cycler)):
            return tuple(rcParam)
        else:
            return rcParam
