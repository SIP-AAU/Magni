"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing configuration options for the multiprocessing subpackage.

See also
--------
magni.utils.config.Configger : The Configger class used

Notes
-----
This module instantiates the `Configger` class provided by
`magni.utils.config`. The configuration options are the following:

silence_exceptions : bool
    A flag indicating if exceptions should be silenced (the default is False,
    which implies that exceptions are raised).
workers : int
    The number of workers to use for multiprocessing (the default is 0, which
    implies no multiprocessing).

"""

from __future__ import division

from magni.utils.config import Configger as _Configger
from magni.utils.validation import validate_numeric as _numeric


configger = _Configger(
    {'silence_exceptions': False, 'workers': 0},
    {'silence_exceptions': _numeric(None, 'boolean'),
     'workers': _numeric(None, 'integer', range_='[0;inf)')})
