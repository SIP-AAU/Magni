"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing implementations of Approximate Message Passing (AMP).

Routine listings
----------------
config
    Configger providing configuration options for this subpackage.
threshold_operator
    Module providing AMP threshold operators.
run(y, A)
    Run the AMP reconstruction algorithm.
stop_criterion
    Module providing AMP stop criteria.
util
    Module providing AMP utilities.


Notes
-----
An implementation of the Donoho, Maleki, Montanari (DMM) AMP using soft
thresholding from [1]_ is provided with threshold options as specified in [2]_.

The implementation allows for using custom thresholding functions and stop
criteria.

References
----------
.. [1] D. L. Donoho, A. Maleki, and A. Montanari, "Message-passing algorithms
   for compressed sensing", *Proceedings of the National Academy of Sciences of
   the United States of America*, vol. 106, no. 45, p. 18914-18919, Nov. 2009.
.. [2] A. Montanari, "Graphical models concepts in compressed sensing" *in
   Compressed Sensing: Theory and Applications*, Y. C. Eldar and G. Kutyniok
   (Ed.), Cambridge University Press, ch. 9, pp. 394-438, 2012.

"""

from magni.cs.reconstruction.amp import (threshold_operator, stop_criterion)
from magni.cs.reconstruction.amp._config import configger as config
from magni.cs.reconstruction.amp._algorithm import run
import magni.cs.reconstruction.amp.util
