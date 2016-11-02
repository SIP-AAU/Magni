"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing threshold functions for the Approximate Message Passing (AMP)
algorithm.

Routine listings
----------------
ValidatedThresholdOperator(magni.utils.validation.types.ThresholdOperator)
    A base class for validated `magni.cs.reconstruction.amp` threshold operator
SoftThreshold(ValidatedThresholdOperator)
    A soft threshold operator.

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
    ThresholdOperator as _ThresholdOperator)


class ValidatedThresholdOperator(_ThresholdOperator):
    """
    A base class for validated `magni.cs.reconstruction.amp` threshold operator

    Parameters
    ----------
    var : dict
        The threshold operator state variables.

    """

    def __init__(self, var):
        super(ValidatedThresholdOperator, self).__init__(var)

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()

    @_validate_once
    def compute_deriv_threshold(self, var):
        """
        Compute the entrywise derivative threshold.

        Parameters
        ----------
        var : dict
            The variables used in computing the derivative threshold.

        Returns
        -------
        eta_deriv : ndarray
            The computed entrywise derivative threshold.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()

    @_validate_once
    def compute_threshold(self, var):
        """
        Compute the entrywise threshold.

        Parameters
        ----------
        var : dict
            The variables used in computing the threshold.

        Returns
        -------
        eta : ndarray
            The computed entrywise threshold.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()

    @_validate_once
    def update_threshold_level(self, var):
        """
        Update the threshold level state.

        Parameters
        ----------
        var : dict
            The variables used in computing the threshold level update.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()


class SoftThreshold(ValidatedThresholdOperator):
    """
    A soft threshold operator.

    This soft threshold operator is based on the description of it and its use
    in AMP as given in [1]_ with corrections from [2]_.

    Parameters
    ----------
    threshold_level_update_method : {'residual', 'median'}
        The method to use for updating the threshold level.
    theta : float
        The tunable regularisation parameter in the threshold level.
    tau_hat_sq : float
        The mean squared error of the (approximated) un-thresholded
        estimate used to determine the threshold level.

    Notes
    -----
    The above Parameters are the threshold parameters that must be passed
    in a `var` dict to the threshold constructor.

    References
    ----------
    .. [1] A. Montanari, "Graphical models concepts in compressed sensing" *in
       Compressed Sensing: Theory and Applications*, Y. C. Eldar and
       G. Kutyniok (Ed.), Cambridge University Press, ch. 9, pp. 394-438, 2012.
    .. [2] J. T. Parker, "Approximate Message Passing Algorithms for
       Generalized Bilinear Inference", PhD Thesis, Graduate School of The Ohio
       State University, 2014

    """

    def __init__(self, var):
        super(SoftThreshold, self).__init__(var)

        @_decorate_validation
        def validate_threshold_parameters():
            _generic(('var', 'threshold_parameters'), 'mapping')
            _generic(('var', 'threshold_parameters',
                      'threshold_level_update_method'), 'string',
                     value_in=['residual', 'median'])
            _numeric(('var', 'threshold_parameters', 'theta'),
                     ('integer', 'floating'), range_='(0;inf)')
            _numeric(('var', 'threshold_parameters', 'tau_hat_sq'),
                     ('integer', 'floating'), range_='[0;inf)')

        validate_threshold_parameters()

        t_params = var['threshold_parameters']

        self.theta = t_params['theta']  # alpha in Eq. (9.44) in [1].
        self.tau_hat_sq = var['convert'](t_params['tau_hat_sq'])
        self.update_method = t_params['threshold_level_update_method']
        self.m = var['y'].shape[0]
        self.stdQ1 = var['convert'](scipy.stats.norm.ppf(1 - 0.25))

    def compute_deriv_threshold(self, var):
        """
        Compute the entrywise derivative soft threshold.

        Parameters
        ----------
        var : dict
            The variables used in computing the derivative threshold.

        Returns
        -------
        eta_deriv : ndarray
            The computed entrywise derivative threshold.

        """

        super(SoftThreshold, self).compute_deriv_threshold(var)

        op = var['alpha_bar_prev'] + var['AH_dot_chi']
        thres = self.theta * np.sqrt(self.tau_hat_sq)
        eta_deriv = var['convert']((op > thres) + (op < -thres))

        return eta_deriv

    def compute_threshold(self, var):
        """
        Compute the entrywise soft threshold.

        Parameters
        ----------
        var : dict
            The variables used in computing the threshold.

        Returns
        -------
        eta : ndarray
            The computed entrywise threshold.

        """

        super(SoftThreshold, self).compute_threshold(var)

        op = var['alpha_bar'] + var['AH_dot_chi']
        thres = self.theta * np.sqrt(self.tau_hat_sq)
        eta = var['convert'](
            ((op - thres) * (op > thres) + (op + thres) * (op < -thres)))

        return eta

    def update_threshold_level(self, var):
        """
        Update the threshold level state.

        Parameters
        ----------
        var : dict
            The variables used in computing the threshold level update.

        """

        super(SoftThreshold, self).update_threshold_level(var)

        chi = var['chi']

        if self.update_method == 'residual':
            # Eq. (9.44) in [1]
            self.tau_hat_sq = 1.0 / self.m * np.linalg.norm(chi)**2

        elif self.update_method == 'median':
            # Eq. (9.45) in [1] corrected according to [2]
            self.tau_hat_sq = (
                1.0 / self.stdQ1 * calculate_median(np.abs(chi)))**2
