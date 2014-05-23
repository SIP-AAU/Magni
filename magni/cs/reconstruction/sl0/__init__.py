"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing implementations of Smoothed l0 Norm SL0).

The implementations provided are the original SL0 reconstruction algorithm and
a modified SL0 reconstruction algorithm. The algorithm used depends on the
'algorithm' configuration option: 'std' refers to the original algorithm while
'mod' (default) refers to the modified algorithm.

Routine listings
----------------
config
    Module providing configuration options for this subpackage.
run(y, A)
    Run the specified SL0 reconstruction algorithm.

Notes
-----
See `_util` for documentation of `run`.

The original SL0 reconstruction algorithm by Mohimani et. al is described in
[1]_ whereas the constraint elimiation intepretation of the original SL0
algorithm by Cui et. al. is described in [2]_. The modified SL0 reconstruction
algorithm by Oxvig et. al. is described in [3]_. Specifically, the provided
sequential implementations are:

| std : The standard SL0 algorithm
|     For delta < 0.55: Standard projection algorithm by Mohimani et. al.
|     For delta >= 0.55: Standard constraint elimination algorithm by Cui et.
      al.
| mod : The modified SL0 algorithm (the default)
|     For delta < 0.55: Modified projection algorithm
|     For delta >= 0.55: Modified constraint elimination algorithm

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

from magni.cs.reconstruction.sl0 import config
from magni.cs.reconstruction.sl0._util import run
