"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing atomic force miscroscopy specific functionality.

The present subpackage includes functionality for handling AFM files and data
and functionality for utilizing the other subpackages for such AFM data.

Routine listings
----------------
config
    Module providing configuration options for this subpackage.
io
    Module providing input/output functionality for MI files.
reconstruction
    Module providing reconstruction and analysis of reconstructed images.
types
    Module providing data container classes for MI files.

"""

from magni.afm import config
from magni.afm import io
from magni.afm import reconstruction
from magni.afm import types
