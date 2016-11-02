"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing functionality for image dictionary manipulations.

Routine listings
----------------
get_DCT(shape, overcomplete_shape=None)
    Get the DCT fast operation dictionary for the given image shape.
get_DFT(shape, overcomplete_shape=None)
    Get the DFT fast operation dictionary for the given image shape.
def get_DFT_transform_matrix(N)
    Return the normalised N-by-N discrete fourier transform (DFT) matrix.
analysis
    Module providing functionality to analyse dictionaries.
utils
    Module providing utility functions for the dictionaries subpackage.

"""

from magni.imaging.dictionaries._matrices import get_DCT, get_DFT
from magni.imaging.dictionaries._mtx1D import (
    get_DCT_transform_matrix, get_DFT_transform_matrix)
from magni.imaging.dictionaries import analysis
from magni.imaging.dictionaries import utils
