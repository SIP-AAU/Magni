"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing implementations of Smoothed l0 Norm (SL0).

The implementations provided are the original SL0 reconstruction algorithm and
a modified SL0 reconstruction algorithm. The algorithm used depends on the
'algorithm' configuration option: 'std' refers to the original algorithm while
'mod' (default) refers to the modified algorithm.

Routine listings
----------------
config
    Configger providing configuration options for this subpackage.
run(y, A)
    Run the specified SL0 reconstruction algorithm.

Notes
-----
See `_algorithm` for documentation of `run`.

Implementations of the original SL0 reconstruction algorithm [1]_ and a
modified Sl0 reconstruction algorithm [3]_ are available. It is also possible
to configure the subpackage to provide customised versions of the SL0
reconstruction algorithm. The projection algorithm [1]_ is used for small delta
(< 0.55) whereas the contraint elimination algorithm [2]_ is used for large
delta (>= 0.55) which merely affects the computation time.

References
----------
.. [1] H. Mohimani, M. Babaie-Zadeh, and C. Jutten, "A Fast Approach for
   Overcomplete Sparse Decomposition Based on Smoothed l0 Norm", *IEEE
   Transactions on Signal Processing*, vol. 57, no. 1, pp. 289-301, Jan. 2009.
.. [2] Z. Cui, H. Zhang, and W. Lu, "An Improved Smoothed l0-norm Algorithm
   Based on Multiparameter Approximation Function", *in 12th IEEE International
   Conference on Communication Technology (ICCT)*, Nanjing, China, Nov. 11-14,
   2011, pp. 942-945.
.. [3] C. S. Oxvig, P. S. Pedersen, T. Arildsen, and T. Larsen, "Surpassing the
   Theoretical 1-norm Phase Transition in Compressive Sensing by Tuning the
   Smoothed l0 Algorithm", *in IEEE International Conference on Acoustics,
   Speech and Signal Processing (ICASSP)*, Vancouver, Canada, May 26-31, 2013,
   pp. 6019-6023.

"""

from magni.cs.reconstruction.sl0._config import configger as config
from magni.cs.reconstruction.sl0._algorithm import run
