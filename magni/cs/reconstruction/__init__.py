"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing implementations of generic reconstruction algorithms.

Each subpackage provides a family of generic reconstruction algorithms. Thus
each subpackage has a config module and a run function which provide the
interface of the given family of reconstruction algorithms.

Routine listings
----------------
it
    Subpackage providing implementations of Iterative Thresholding (IT).
iht
    Subpackage providing implementations of Iterative Hard Thresholding (IHT).
    (Deprecated)
sl0
    Subpackage providing implementations of Smoothed l0 Norm (SL0).

"""

from magni.cs.reconstruction import it
from magni.cs.reconstruction import iht
from magni.cs.reconstruction import sl0
