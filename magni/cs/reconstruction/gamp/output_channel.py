"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing output channel functions for the Generalised Approximate
Message Passing (GAMP) algorithm.

Routine listings
----------------
ValidatedMMSEOutputChannel(magni.utils.validation.types.MMSEOutputChannel)
    A base class for validated `magni.cs.reconstruction.gamp` output channels.
AWGN(ValidatedMMSEOutputChannel)
    An Additive White Gaussian Noise (AWGN) MMSE output channel.

"""

from __future__ import division

import numpy as np
import scipy.stats
try:
    import bottleneck as bn
    calculate_median = bn.median
except ImportError:
    calculate_median = np.median

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.validation import validate_once as _validate_once
from magni.utils.validation.types import (
    MMSEOutputChannel as _MMSEOutputChannel)


class ValidatedMMSEOutputChannel(_MMSEOutputChannel):
    """
    A base class for validated `magni.cs.reconstruction.gamp` output channels.

    Parameters
    ----------
    var : dict
        The output channel state variables.

    """

    def __init__(self, var):
        super(ValidatedMMSEOutputChannel, self).__init__(var)

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()

    @_validate_once
    def compute(self, var):
        """
        Compute the output channel value.

        Parameters
        ----------
        var : dict
            The variables used in computing of the output channel value.

        Returns
        -------
        mean : ndarray
            The computed output channel mean.
        variance : ndarray
            The computed output channel variance.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()


class AWGN(ValidatedMMSEOutputChannel):
    """
    An Additive White Gaussian Noise (AWGN) MMSE output channel.

    This channel is based on equations (41), (42), and (43) in [1]_ and allows
    for using Expectation Maximization (EM) for learning the channel parameter
    as detailed in equation (77) in [2]_ (see also [2]_ for an introduction to
    EM for GAMP).

    Parameters
    ----------
    sigma_sq : float or int
        The noise level variance (initial noise level when the noise level is
        estimated).
    noise_level_estimation : str
        The method used for estimating (learning) the noise level in each
        iteration.

    Notes
    -----
    The above Parameters are the output channel parameters that must be passed
    in a `var` dict to the channel constructor.

    Possible values for `noise_level_estimation` are:

    * 'sample_variance' - Estimate noise level using the sample variance.
    * 'median' - Estimate noise level from the median.
    * 'em' - Estimate noise using Expectation Maximization (EM).
    * 'fixed' - Use a fixed noise level in all iterations.

    References
    ----------
    .. [1] S. Rangan, "Generalized Approximate Message Passing for Estimation
       with Random Linear Mixing", arXiv:1010.5141v2, pp. 1-22, Aug. 2012.
    .. [2] F. Krzakala, M. Mezard, F. Sausset, Y. Sun, and L. Zdeborova,
       "Probabilistic reconstruction in compressed sensing: algorithms, phase
       diagrams, and threshold achieving matrices", *Journal of Statistical
       Mechanics: Theory and Experiment*, vol. P08009, pp. 1-57, Aug. 2012.
    .. [3] J. P. Vila and P. Schniter, "Expectation-Maximization
       Gaussian-Mixture Approximate Message Passing", *IEEE Transactions on
       Signal Processing*, 2013, vol. 61, no. 19, pp. 4658-4672, Oct. 2013.

    """

    def __init__(self, var):
        super(AWGN, self).__init__(var)

        @_decorate_validation
        def validate_channel_parameters():
            _generic(('var', 'output_channel_parameters'), 'mapping')
            _numeric(('var', 'output_channel_parameters', 'sigma_sq'),
                     ('integer', 'floating'), range_='[0;inf)')
            _generic(('var', 'output_channel_parameters',
                      'noise_level_estimation'),
                     'string',
                     value_in=['sample_variance', 'median', 'em', 'fixed'])

        validate_channel_parameters()

        c_params = var['output_channel_parameters']

        self.m = var['y'].shape[0]
        self.noise_level_estimation = c_params['noise_level_estimation']
        self.stdQ1 = var['convert'](scipy.stats.norm.ppf(1 - 0.25))
        self.sigma_sq = var['convert'](c_params['sigma_sq'])

    def compute(self, var):
        """
        Compute the AWGN output channel value.

        Parameters
        ----------
        var : dict
            The variables used in computing of the output channel value.

        Returns
        -------
        mean : ndarray
            The computed output channel mean.
        variance : ndarray
            The computed output channel variance.

        """

        super(AWGN, self).compute(var)

        y = var['y']

        if var['it'] != 0:  # Allow for fixed initialisation of sigma_sq
            # Noise level estimation for current iteration
            if self.noise_level_estimation == 'sample_variance':
                self.sigma_sq = 1.0/self.m * np.sum(
                    (y - var['A_dot_alpha_bar'])**2)
            elif self.noise_level_estimation == 'median':
                # Estimate variance based on median
                # std(x) ~= median(|x|) / stdQ1 for x~N(0, std**2)
                self.sigma_sq = (
                    calculate_median(np.abs(y - var['A_dot_alpha_bar'])) /
                    self.stdQ1) ** 2

        sigma_sq_plus_v = self.sigma_sq + var['v']

        mean = (var['v'] * y + self.sigma_sq * var['o']) / sigma_sq_plus_v
        variance = (self.sigma_sq * var['v']) / sigma_sq_plus_v

        if self.noise_level_estimation == 'em':
            # EM noise level recursion
            # See Eq. (77) in [1]
            self.sigma_sq = self.sigma_sq * np.sum(
                (np.abs(mean - var['o']) / var['v'])**2) / np.sum(
                    1.0 / sigma_sq_plus_v)

        if len(variance.shape) != 2:
            # When using a sum approximation of A_asq, the variance becomes
            # scalar. However, our interface dictates that it must be a column
            # vector for the sum approximation to be correct. Also, the
            # denominator sum of the sigma_sq EM update must be scaled
            # correctly as if sigma_sq_plus_v had been a vector.

            variance = np.ones(mean.shape, dtype=var['convert']) * variance
            self.sigma_sq = self.sigma_sq / len(y)

        return mean, variance
