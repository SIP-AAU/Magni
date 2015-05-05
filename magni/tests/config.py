"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests of the config modules.

Routine listings
----------------
UpdateTest(unittest.TestCase):
    Test if each config module can be updated with its defaults.

"""

from __future__ import division
import unittest

import magni


class UpdateTest(unittest.TestCase):
    """
    Test if each config module can be updated with its defaults.

    """

    def test_afm_config(self):
        config = magni.afm.config
        config.update(dict(config.items()))

    def test_cs_phase_transition_config(self):
        config = magni.cs.phase_transition.config
        config.update(dict(config.items()))

    def test_cs_reconstruction_iht_config(self):
        config = magni.cs.reconstruction.iht.config
        config.update(dict(config.items()))

    def test_cs_reconstruction_it_config(self):
        config = magni.cs.reconstruction.it.config
        config.update(dict(config.items()))

    def test_cs_reconstruction_sl0_config(self):
        config = magni.cs.reconstruction.sl0.config
        config.update(dict(config.items()))

    def test_utils_multiprocessing_config(self):
        config = magni.utils.multiprocessing.config
        config.update(dict(config.items()))
