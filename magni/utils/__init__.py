"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing support functionality for the other subpackages.

Routine listings
----------------
multiprocessing
    Subpackage providing intuitive and extensive multiprocessing functionality.
config
    Module providing a robust configger class.
matrices
    Module providing matrix emulators.
plotting
    Module providing utilities for control of plotting using `matplotlib`.
validation
    Subpackage providing validation capability.
types
    Module providing custom data types.
split_path(path)
    Split a path into folder path, file name, and file extension.

Notes
-----
See `_util` for documentation of `split_path`.

"""

# the validation and config modules need to be imported first to avoid
# recursive imports
from magni.utils import validation
from magni.utils import config

from magni.utils import matrices
from magni.utils import multiprocessing
from magni.utils import plotting
from magni.utils import types
from magni.utils._util import split_path
