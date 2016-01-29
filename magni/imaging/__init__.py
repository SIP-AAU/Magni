"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing functionality for image manipulation.

Routine listings
----------------
dictionaries
    Module providing fast linear operations wrapped in matrix emulators.
domains
    Module providing a multi domain image class.
evaluation
    Module providing functions for evaluation of image reconstruction quality.
measurements
    Module providing functions for constructing scan patterns for measurements.
preprocessing
    Module providing functionality to remove tilt in images.
visualisation
    Module providing functionality for visualising images.
mat2vec(x)
    Function to reshape a matrix into vector by stacking columns.
vec2mat(x, mn_tuple)
    Function to reshape a vector into a matrix.
double_mirror(img, fftstyle=True)
    Function to mirror image in both the vertical and horisontal axes.
get_inscribed_masks(img, as_vec=False)
    Function to get inscribed masks covering the image.

Notes
-----
See `_util` for documentation of `mat2vec`, `vec2mat`, `double_mirror`, and
`get_inscribed_masks`.

"""

# the mat2vec and vec2mat functions need to be imported first to avoid
# recursive imports
from magni.imaging._util import mat2vec
from magni.imaging._util import vec2mat
from magni.imaging._util import double_mirror
from magni.imaging._util import get_inscribed_masks

from magni.imaging import dictionaries
from magni.imaging import domains
from magni.imaging import evaluation
from magni.imaging import measurements
from magni.imaging import preprocessing
from magni.imaging import visualisation
