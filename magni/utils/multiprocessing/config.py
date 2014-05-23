"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the multiprocessing subpackage.

Routine listings
----------------
get(key=None)
    Get the value of one or more configuration options.
set(dictionary={}, \*\*kwargs)
    Set the value of one or more configuration options.

See Also
--------
magni.utils.config.Configger : The Configger class used.

Notes
-----
This module instantiates the `Configger` class provided by `magni.utils.config`
and assigns handles for the `get` and `set` methods of that class instance. The
configuration options are the following:

workers : int
    The number of workers to use for multiprocessing (the default is 0, which
    implies no multiprocessing).

"""

from __future__ import division

from magni.utils.config import Configger as _Configger


_configger = _Configger({'workers': 0}, {'workers': {'type': int, 'min': 0}})
get = _configger.get
set = _configger.set
