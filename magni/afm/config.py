"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the `magni.afm` subpackage.

Routine listings
----------------
get(key=None)
    Get the value of one or more configuration options.
set(dictionary={}, \*\*kwargs)
    Set the value of one or more configuration options.

See also
--------
magni.utils.config : The Configger class used

Notes
-----
This module instantiates the `Configger` class provided by `magni.utils.config`
and assigns handles for the `get` and `set` methods of that class instance. The
configuration options are the following:

algorithm : {'iht', 'sl0'}
    The compressed sensing reconstruction algorithm to use (the default is
    'iht').

"""

from __future__ import division

from magni.utils.config import Configger as _Configger


_configger = _Configger({'algorithm': 'iht'},
                        {'algorithm': {'type': str,
                                       'value_in': ['iht', 'sl0']}})

get = _configger.get
set = _configger.set
