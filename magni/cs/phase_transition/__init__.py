"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing phase transition determination functionality.

Routine listings
----------------
config
    Configger providing configuration options for this subpackage.
io
    Module providing input/output functionality for stored phase transitions.
plotting
    Module providing plotting for this subpackage.
determine(algorithm, path, label='default', overwrite=False)
    Determine the phase transition of a reconstruction algorithm.

Notes
-----
See `_util` for documentation of `determine`.

The phase transition of a reconstruction algorithm describes the reconstruction
capabilities of that reconstruction algorithm. For a description of the concept
of phase transition, see [1]_.

References
----------
.. [1] C. S. Oxvig, P. S. Pedersen, T. Arildsen, and T. Larsen, "Surpassing the
   Theoretical 1-norm Phase Transition in Compressive Sensing by Tuning the
   Smoothed l0 Algorithm", *in IEEE International Conference on Acoustics,
   Speech and Signal Processing (ICASSP)*, Vancouver, Canada, May 26-31, 2013,
   pp. 6019-6023.

"""

from magni.cs.phase_transition._config import configger as config
from magni.cs.phase_transition import io
from magni.cs.phase_transition import plotting
from magni.cs.phase_transition._util import determine
