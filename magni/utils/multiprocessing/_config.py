"""
..
    Copyright (c) 2014-2016, Magni developers.
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

max_broken_pool_restarts : int or None
    The maximum number of attempts at restarting the process pool upon a
    BrokenPoolError (the default is 0, which implies that the process pool
    may not be restarted). If set to None, the process pool may restart
    indefinitely.
prefer_futures : bool
    The indicator of whether or not to prefer the concurrent.futures module
    over the multiprocessing module (the default is False, which implies that
    the multiprocessing module is used).
re_raise_exceptions : bool
    A flag indicating if exceptions should be re-raised (the default is False,
    which implies that the exception are not re-raised). It is useful to set
    this to True if the processing pool supports proper exception handling as
    e.g. when using "futures".
silence_exceptions : bool
    A flag indicating if exceptions should be silenced (the default is False,
    which implies that exceptions are raised).
workers : int
    The number of workers to use for multiprocessing (the default is 0, which
    implies no multiprocessing).

See the notes for the `magni.utils.multiprocessing._processing.process`
function for more details about the `prefer_futures` and
`max_broken_pool_restarts` configuration parameters.

"""

from __future__ import division

from magni.utils.config import Configger as _Configger
from magni.utils.validation import validate_numeric as _numeric


configger = _Configger(
    {'max_broken_pool_restarts': 0,
     'prefer_futures': False,
     're_raise_exceptions': False,
     'silence_exceptions': False,
     'workers': 0},
    {'max_broken_pool_restarts': _numeric(None, 'integer', range_='[0;inf)',
                                          ignore_none=True),
     'prefer_futures': _numeric(None, 'boolean'),
     're_raise_exceptions': _numeric(None, 'boolean'),
     'silence_exceptions': _numeric(None, 'boolean'),
     'workers': _numeric(None, 'integer', range_='[0;inf)')})
