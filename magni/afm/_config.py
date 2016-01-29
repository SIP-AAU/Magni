"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the `magni.afm` subpackage.

See also
--------
magni.utils.config.Configger : The Configger class used

Notes
-----
This module instantiates the `Configger` class provided by
`magni.utils.config`. The configuration options are the following:

algorithm : {'it', 'iht', 'sl0'}
    The compressed sensing reconstruction algorithm subpackage to use (the
    default is 'it').

"""

from __future__ import division

from magni.utils.config import Configger as _Configger
from magni.utils.validation import validate_generic as _generic


configger = _Configger(
    {'algorithm': 'it'},
    {'algorithm': _generic(None, 'string', value_in=('iht', 'it', 'sl0'))})
