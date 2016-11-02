"""
..
    Copyright (c) 2015-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing input channel functions for the Generalised Approximate
Message Passing (GAMP) algorithm.

Routine listings
----------------
ValidatedMMSEInputChannel(magni.utils.validation.types.MMSEInputChannel)
    A base class for validated `magni.cs.reconstruction.gamp` input channels.
IIDBG(ValidatedMMSEInputChannel)
    An i.i.d. Bernoulli Gauss MMSE input channel.
IIDsGB(ValidatedMMSEInputChannel)
    An i.i.d. sparse Gauss Bernoulli MMSE input channel.

"""

from __future__ import division

import numpy as np

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.validation import validate_once as _validate_once
from magni.utils.validation.types import MMSEInputChannel as _MMSEInputChannel


class ValidatedMMSEInputChannel(_MMSEInputChannel):
    """
    A base class for validated `magni.cs.reconstruction.gamp` input channels.

    Parameters
    ----------
    var : dict
        The input channel state variables.

    """

    def __init__(self, var):
        super(ValidatedMMSEInputChannel, self).__init__(var)

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()

    @_validate_once
    def compute(self, var):
        """
        Compute the input channel value.

        Parameters
        ----------
        var : dict
            The variables used in computing of the input channel value.

        Returns
        -------
        mean : ndarray
            The computed input channel mean.
        variance : ndarray
            The computed input channel variance.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()


class IIDBG(ValidatedMMSEInputChannel):
    """
    An i.i.d. Bernoulli Gauss MMSE input channel.

    This channel is based on equations (6), (7) in [3]_ and allows for using
    Expectation Maximization (EM) for learning the channel parameters as
    detailed in equations (19), (25), and (32) in [3]_ (see also [4]_ for an
    introduction to EM for GAMP).

    Parameters
    ----------
    tau : float or int
        The prior signal "density" (fraction of Gauss to Bernouilli).
    theta_bar : float or int
        The prior Gaussian mean.
    theta_tilde : float or int
        The prior Gaussian variance.
    use_em : bool
        The indicator of whether or not to use Expectation Maximazion (EM) to
        learn the prior parameters.

    Notes
    -----
    The above Parameters are the input channel parameters that must be passed
    in a `var` dict to the channel constructor.

    If `use_em` is True, the values given for `tau`, `theta_bar`, and
    `theta_tilde` are used for initialialisation.

    This channel is theoretically equivalent to the `IIDsGB` channel. However,
    due to numerical subtleties, it may give different results.

    References
    ----------
    .. [3] J. Vila, J. and P. Schniter, "Expectation-Maximization
       Bernoulli-Gaussian Approximate Message Passing", *in Forty Fifth
       Asilomar Conference on Signals, Systems and Computers (ASILOMAR)*,
       pp. 799-803, Pacific Grove, California, USA, Nov. 6-9, 2011
    .. [4] J. P. Vila and P. Schniter, "Expectation-Maximization
       Gaussian-Mixture Approximate Message Passing", *IEEE Transactions on
       Signal Processing*, 2013, vol. 61, no. 19, pp. 4658-4672, Oct. 2013.

    """

    def __init__(self, var):
        super(IIDBG, self).__init__(var)

        @_decorate_validation
        def validate_channel_parameters():
            _generic(('var', 'input_channel_parameters'), 'mapping')
            _numeric(('var', 'input_channel_parameters', 'tau'),
                     ('integer', 'floating'), range_='[0;1]')
            _numeric(('var', 'input_channel_parameters', 'theta_bar'),
                     ('integer', 'floating'))
            _numeric(('var', 'input_channel_parameters', 'theta_tilde'),
                     ('integer', 'floating'), range_='[0;inf)')
            _numeric(('var', 'input_channel_parameters', 'use_em'),
                     ('boolean'))

        validate_channel_parameters()

        c_params = var['input_channel_parameters']

        self.use_em = c_params['use_em']  # Whether or not to use EM learning
        self.tau = var['convert'](c_params['tau'])
        self.theta_bar = var['convert'](c_params['theta_bar'])
        self.theta_tilde = var['convert'](c_params['theta_tilde'])
        self.n = var['convert'](var['A'].shape[1])

    def compute(self, var):
        """
        Compute the IIDBG input channel value.

        Parameters
        ----------
        var : dict
            The variables used in computing of the input channel value.

        Returns
        -------
        mean : ndarray
            The computed input channel mean.
        variance : ndarray
            The computed input channel variance.

        """

        super(IIDBG, self).compute(var)

        s = var['s']
        r = var['r']
        tau = self.tau
        theta_bar = self.theta_bar
        theta_tilde = self.theta_tilde

        h_pzc = (theta_tilde * s) / (s + theta_tilde)
        g_pzc = (theta_bar / theta_tilde + r / s) * h_pzc
        k_pzc = 1 + (1 - tau) / tau * np.sqrt(theta_tilde / h_pzc) * np.exp(
            1/2 * ((r - theta_bar)**2 / (s + theta_tilde) - r**2 / s))
        l_pzc = 1.0 / k_pzc

        # New values of alpha_bar (mean) and alpha_tilde (variance)
        mean = l_pzc * g_pzc
        variance = l_pzc * (h_pzc + g_pzc**2) - l_pzc**2 * g_pzc**2

        if self.use_em:
            # EM update of tau
            # See Eq. (19) in [3]
            tau = 1.0 / self.n * np.sum(l_pzc)
            self.tau = var['convert'](tau)

            # EM update of theta_bar
            # See Eq. (25) in [3]
            theta_bar = 1.0 / (self.n * tau) * np.sum(mean)
            self.theta_bar = var['convert'](theta_bar)

            # EM update of theta_tilde
            # See Eq. (32) in [3]
            theta_tilde = 1.0 / (self.n * tau) * np.sum(
                l_pzc * ((theta_bar - g_pzc)**2 + h_pzc))
            self.theta_tilde = var['convert'](theta_tilde)

        return mean, variance


class IIDsGB(ValidatedMMSEInputChannel):
    """
    An i.i.d. sparse Gauss Bernoulli MMSE input channel.

    This channel is based on equations (68), (69) in [1]_ and allows for using
    Expectation Maximization (EM) for learning the channel parameters as
    detailed in equations (74), (78), and (79) in [1]_ (see also [2]_ for an
    introduction to EM for GAMP).

    Parameters
    ----------
    tau : float or int
        The prior signal "density" (fraction of Gauss to Bernouilli).
    theta_bar : float or int
        The prior Gaussian mean.
    theta_tilde : float or int
        The prior Gaussian variance.
    use_em : bool
        The indicator of whether or not to use Expectation Maximazion (EM) to
        learn the prior parameters.
    em_damping : float or int
        The damping of the EM updates (if using EM).

    Notes
    -----
    The above Parameters are the input channel parameters that must be passed
    in a `var` dict to the channel constructor.

    If `use_em` is True, the values given for `tau`, `theta_bar`, and
    `theta_tilde` are used for initialialisation. The `em_damping` must be in
    [0, 1) with 0 being no damping.

    This channel is theoretically equivalent to the `IIDBG` channel. However,
    due to numerical subtleties, it may give different results.

    References
    ----------
    .. [1] F. Krzakala, M. Mezard, F. Sausset, Y. Sun, and L. Zdeborova,
       "Probabilistic reconstruction in compressed sensing: algorithms, phase
       diagrams, and threshold achieving matrices", *Journal of Statistical
       Mechanics: Theory and Experiment*, vol. P08009, pp. 1-57, Aug. 2012.
    .. [2] J. P. Vila and P. Schniter, "Expectation-Maximization
       Gaussian-Mixture Approximate Message Passing", *IEEE Transactions on
       Signal Processing*, 2013, vol. 61, no. 19, pp. 4658-4672, Oct. 2013.

    """

    def __init__(self, var):
        super(IIDsGB, self).__init__(var)

        @_decorate_validation
        def validate_channel_parameters():
            _generic(('var', 'input_channel_parameters'), 'mapping')
            _numeric(('var', 'input_channel_parameters', 'tau'),
                     ('integer', 'floating'), range_='[0;1]')
            _numeric(('var', 'input_channel_parameters', 'theta_bar'),
                     ('integer', 'floating'))
            _numeric(('var', 'input_channel_parameters', 'theta_tilde'),
                     ('integer', 'floating'), range_='[0;inf)')
            _numeric(('var', 'input_channel_parameters', 'use_em'),
                     ('boolean'))

            if var['input_channel_parameters']['use_em']:
                _numeric(('var', 'input_channel_parameters', 'em_damping'),
                         ('integer', 'floating'), range_='[0;1)')

        validate_channel_parameters()

        c_params = var['input_channel_parameters']
        m = var['convert'](var['A'].shape[0])

        self.use_em = c_params['use_em']  # Whether or not to use EM learning
        if self.use_em:
            # EM damping level
            self.em_damp = var['convert'](c_params['em_damping'])
        self.tau = var['convert'](c_params['tau'])
        self.theta_bar = var['convert'](c_params['theta_bar'])
        self.theta_tilde = var['convert'](c_params['theta_tilde'])
        self.n = var['convert'](var['A'].shape[1])
        self.delta = var['convert'](m / self.n)

    def compute(self, var):
        """
        Compute the IIDsGB input channel value.

        Parameters
        ----------
        var : dict
            The variables used in computing of the input channel value.

        Returns
        -------
        mean : ndarray
            The computed input channel mean.
        variance : ndarray
            The computed input channel variance.

        """

        super(IIDsGB, self).compute(var)

        s = var['s']
        r = var['r']
        tau = self.tau
        theta_bar = self.theta_bar
        theta_tilde = self.theta_tilde

        q_denom = s + theta_tilde
        q_bar = (s * theta_bar + r * theta_tilde) / q_denom
        q_tilde = s * theta_tilde / q_denom

        fct1 = np.sqrt(1 + theta_tilde / s)
        arg1 = ((-r**2 * theta_tilde + s * theta_bar * (theta_bar - 2 * r)) /
                (2 * s * (s + theta_tilde)))
        fct2 = tau + (1 - tau) * fct1 * np.exp(arg1)

        # New values of alpha_bar (mean) and alpha_tilde (variance)
        mean = q_bar * tau / fct2
        variance = tau * (q_tilde + q_bar**2) / fct2 - mean**2

        if self.use_em:
            # EM update of tau
            # See Eq. (74) in [2]
            qem1 = 1/theta_tilde + 1/s
            qem2 = r / s + theta_bar / theta_tilde
            qem_exp = np.exp(
                1/2 * (qem2**2 / qem1 - theta_bar**2 / theta_tilde))
            qem_num = np.sum(qem1 / qem2 * mean)
            qem_denom = np.sum(
                1 / (1 - tau + tau * np.sqrt(qem1 / theta_tilde) * qem_exp))
            tau_f = qem_num / qem_denom
            if tau_f > self.delta:
                tau_f = self.delta
            tau = (1 - self.em_damp) * tau_f + self.em_damp * tau
            self.tau = var['convert'](tau)

            # EM update of theta_bar
            # See Eq. (78) in [2]
            theta_bar_f = np.sum(mean) / (tau * self.n)
            theta_bar = ((1 - self.em_damp) * theta_bar_f +
                         self.em_damp * theta_bar)
            self.theta_bar = var['convert'](theta_bar)

            # EM update of theta_tilde
            # See Eq. (79) in [2]
            theta_tilde_f = np.sum(
                variance + mean**2) / (tau * self.n) - theta_bar**2
            if theta_tilde_f < 0:
                theta_tilde_f = var['convert'](0)
            theta_tilde = ((1 - self.em_damp) * theta_tilde_f +
                           self.em_damp * theta_tilde)
            self.theta_tilde = var['convert'](theta_tilde)

        return mean, variance
