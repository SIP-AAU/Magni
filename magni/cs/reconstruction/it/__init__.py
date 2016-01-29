"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing implementations of Iterative Thresholding (IT).

Routine listings
----------------
config
    Configger providing configuration options for this subpackage.
run(y, A)
    Run the IT reconstruction algorithm.

Notes
-----
Implementations of Iterative Hard Thresholding (IHT) [1]_, [2]_ as well as
implementations of Iterative Soft Thresholding (IST) [3]_, [4]_ are available.
It is also possible to configure the subpackage to use a model based approach
as described in [5]_.

References
----------
.. [1] T. Blumensath and M.E. Davies, "Iterative Thresholding for Sparse
   Approximations", *Journal of Fourier Analysis and Applications*, vol. 14,
   pp. 629-654, Sep. 2008.
.. [2] T. Blumensath and M.E. Davies, "Normalized Iterative Hard Thresholding:
   Guaranteed Stability and Performance", *IEEE Journal Selected Topics in
   Signal Processing*, vol. 4, no. 2, pp. 298-309, Apr. 2010.
.. [3] I. Daubechies, M. Defrise, and C. D. Mol, "An Iterative Thresholding
   Algorithm for Linear Inverse Problems with a Sparsity Constraint",
   *Communications on Pure and Applied Mathematics*, vol. 57, no. 11,
   pp. 1413-1457, Nov. 2004.
.. [4] A. Maleki and D.L. Donoho, "Optimally Tuned Iterative Reconstruction
   Algorithms for Compressed Sensing", *IEEE Journal Selected Topics in Signal
   Processing*, vol. 3, no. 2, pp. 330-341, Apr. 2010.
.. [5] R.G. Baraniuk, V. Cevher, M.F. Duarte, and C. Hedge, "Model-Based
   Compressive Sensing", *IEEE Transactions on Information Theory*, vol. 56,
   no. 4, pp. 1982-2001, Apr. 2010.

"""

from magni.cs.reconstruction.it._config import configger as config
from magni.cs.reconstruction.it._algorithm import run
