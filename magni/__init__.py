"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Package providing a toolbox for compressed sensing for atomic force microscopy.

Routine listings
----------------
afm
    Subpackage providing atomic force miscroscopy specific functionality.
cs
    Subpackage providing generic compressed sensing functionality.
imaging
    Subpackage providing generic imaging functionality.
tests
    Subpackage providing unittesting of the other subpackages.
utils
    Subpackage providing support functionality for the other subpackages.

Notes
-----
See the README file for additional information.

"""

__version__ = '1.0.0'

from magni import afm
from magni import cs
from magni import imaging
from magni import reproducibility
from magni import utils
