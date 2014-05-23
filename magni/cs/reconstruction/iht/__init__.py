"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing an implementation of Iterative Hard Thresholding (IHT).

Routine listings
----------------
config
    Module providing configuration options for this subpackage.
run(y, A)
    Run the IHT reconstruction algorithm.

Notes
-----
See `_original` for documentation of `run`.

The IHT reconstruction algorithm is described in [1]_.

References
----------
.. [1] A. Maleki and D.L. Donoho, "Optimally Tuned Iterative Reconstruction
   Algorithms for Compressed Sensing", *IEEE Journal Selected Topics in Signal
   Processing*, vol. 3, no. 2, pp. 330-341, Apr. 2010.

"""

from magni.cs.reconstruction.iht import config
from magni.cs.reconstruction.iht._original import run
