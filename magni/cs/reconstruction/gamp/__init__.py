"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing implementations of Generalised Approximate Message Passing
(GAMP).

Routine listings
----------------
channel_initialisation
    Module providing functionality for initialisations of GAMP channels.
config
    Configger providing configuration options for this subpackage.
input_channel
    Module providing GAMP input channels.
output_channel
    Module providing GAMP output channels.
run(y, A, A_asq=None)
    Run the GAMP reconstruction algorithm.
stop_criterion
    Module providing GAMP stop criteria.


Notes
-----
Implementations of Mimimum Mean Squared Error (MMSE) Generalised Approximate
Message Passing (GAMP) from [1]_, [2]_ based on description of it in [3]_ are
available. The GAMP is a generalisation of the Approximate Message Passing
(AMP) algorithm derived independelty by Donoho et al. [4]_ and Krzakala et al.
[5]_, [6]_.

This implementation allows custom input- and output channels as well as the use
of sum approximations of the squared system matrix as detailed in [2]_, [5]_.
Furthermore, a simple damping option is available based on the description in
[7]_ (see also [8]_ for more details on damping in GAMP).

References
----------
.. [1] S. Rangan, "Generalized Approximate Message Passing for Estimation with
   Random Linear Mixing", *in IEEE International Symposium on Information
   Theory (ISIT)*, St. Petersburg, Russia, Jul. 31 - Aug. 5, 2011,
   pp. 2168-2172.
.. [2] S. Rangan, "Generalized Approximate Message Passing for Estimation
   with Random Linear Mixing", arXiv:1010.5141v2, pp. 1-22, Aug. 2012.
.. [3] J. T. Parker, "Approximate Message Passing Algorithms for Generalized
   Bilinear Inference", PhD Thesis, Graduate School of The Ohio State
   University, 2014
.. [4] D.L. Donoho, A. Maleki, and A. Montanari, "Message-passing algorithms
   for compressed sensing", *Proceedings of the National Academy of Sciences of
   the United States of America*, vol. 106, no. 45, pp. 18914-18919, Nov. 2009.
.. [5] F. Krzakala, M. Mezard, F. Sausset, Y. Sun, and L. Zdeborova,
   "Probabilistic reconstruction in compressed sensing: algorithms, phase
   diagrams, and threshold achieving matrices", *Journal of Statistical
   Mechanics: Theory and Experiment*, vol. P08009, pp. 1-57, Aug. 2012.
.. [6] F. Krzakala, M. Mezard, F. Sausset, Y. Sun, and L. Zdeborova,
   "Statistical-Physics-Based Reconstruction in Compressed Sensing", *Physics
   Review X*, vol. 2, no. 2, pp. (021005-1)-(021005-18), May 2012.
.. [7] S. Rangan, P. Schniter, and A. Fletcher. "On the Convergence of
   Approximate Message Passing with Arbitrary Matrices", *in IEEE International
   Symposium on Information Theory (ISIT)*, pp. 236-240, Honolulu, Hawaii, USA,
   Jun. 29 - Jul. 4, 2014.
.. [8] J. Vila, P. Schniter, S. Rangan, F. Krzakala, L. Zdeborova, "Adaptive
   Damping and Mean Removal for the Generalized Approximate Message Passing
   Algorithm", *in IEEE International Conference on Acoustics, Speech, and
   Signal Processing (ICASSP)*, South Brisbane, Queensland, Australia, Apr.
   19-24, 2015, pp. 2021-2025.

"""

from magni.cs.reconstruction.gamp import (
    channel_initialisation, input_channel, output_channel, stop_criterion)
from magni.cs.reconstruction.gamp._config import configger as config
from magni.cs.reconstruction.gamp._algorithm import run
