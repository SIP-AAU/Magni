"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing functionality for image dictionary manipulations.

Routine listings
----------------
get_DCT(shape)
    Get the DCT fast operation dictionary for the given image shape.
get_DFT(shape)
    Get the DFT fast operation dictionary for the given image shape.
analysis
    Module providing functionality to analyse dictionaries.
utils
    Module providing utility functions for the dictionaries subpackage.

"""

from magni.imaging.dictionaries._matrices import get_DCT, get_DFT
from magni.imaging.dictionaries import analysis
from magni.imaging.dictionaries import utils
