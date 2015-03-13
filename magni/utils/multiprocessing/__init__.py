"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing intuitive and extensive multiprocessing functionality.

Routine listings
----------------
config
    Configger providing configuration options for this subpackage.
File()
    Control pytables access to hdf5 files when using multiprocessing.
process(func, namespace={}, args_list=None, kwargs_list=None, maxtasks=None)
    Map multiple function calls to multiple processors.

Notes
-----
See `_util` for documentation of `File`.
See `_processing` for documentation of `process`.

"""

from magni.utils.multiprocessing._config import configger as config
from magni.utils.multiprocessing._util import File
from magni.utils.multiprocessing._processing import process
