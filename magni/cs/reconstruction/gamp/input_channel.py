"""
..
    Copyright (c) 2015-2017, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing input channel functions for the Generalised Approximate
Message Passing (GAMP) algorithm.

Routine listings
----------------
ValidatedMMSEInputChannel(magni.utils.validation.types.MMSEInputChannel)
    A base class for validated `magni.cs.reconstruction.gamp` input channels.
ValidatedBasicMMSEInputChannel(ValidatedMMSEInputChannel)
    A base class for validated basic input channels.
GWS(ValidatedMMSEInputChannel)
    A General Weighted Sparse MMSE input channel.
IIDG(ValidatedBasicMMSEInputChannel)
    An i.i.d. Gaussian MMSE input channel.
IIDL(ValidatedBasicMMSEInputChannel)
    An i.i.d. Laplace MMSE input channel.
IIDBG(ValidatedMMSEInputChannel)
    An i.i.d. Bernoulli Gauss MMSE input channel.
IIDsGB(ValidatedMMSEInputChannel)
    An i.i.d. sparse Gauss Bernoulli MMSE input channel.

"""

from __future__ import division
import copy

import numpy as np
from scipy import special, stats

from magni.utils.config import Configger as _Configger
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

        Subclasses of this class are expected to override this method and then
        call it using `super` since it only implements the necessary input
        validation.

        """

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()


class ValidatedBasicMMSEInputChannel(ValidatedMMSEInputChannel):
    """
    A base class for validated basic input channels.

    The term "basic" refers to channels that may be used in combination with
    the General Weighted Sparse (`GWS`) channel framework.

    Parameters
    ----------
    var : dict
        The input channel state variables.

    """

    def __init__(self, var):
        super(ValidatedBasicMMSEInputChannel, self).__init__(var)

    @_validate_once
    def compute_Z(self, var):
        """
        Compute the input channel normalisation constant.

        Parameters
        ----------
        var : dict
            The variables used in computing of the normalisation constant.

        Returns
        -------
        Z : ndarray
            The computed normalisation constant.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        Subclasses of this class are expected to override this method and then
        call it using `super` since it only implements the necessary input
        validation.

        """

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()

    @_validate_once
    def get_EM_element(self, channel_parameter):
        """
        Return the element needed in computing the channel_parameter EM update.

        Parameters
        ----------
        channel_parameter : str
            The channel parameter for which the EM element is needed.

        Returns
        -------
        EM_element : ndarray
            The EM element needed for the channel parameter EM update.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        Subclasses of this class are expected to override this method and then
        call it using `super` since it only implements the necessary input
        validation.

        """

        @_decorate_validation
        def validate_input():
            _generic('channel_parameter', 'string')

        validate_input()


class GWS(ValidatedMMSEInputChannel):
    """
    A General Weighted Sparse MMSE input channel.

    This channel is a an independent but non-identically weighted linear
    combination of a Bernoulli component and a "phi" component from another
    arbitrary distribution.

    Parameters
    ----------
    tau : float or int
        The prior signal "density" (fraction of "phi" to Bernouilli).
    weights : ndarray or None
        The n-by-1 vector of channel weights. If None, a vector of all ones is
        used.
    phi_channel : ValidatedBasicMMSEInputChannel
        The input channel instance implementing the "phi" component.
    phi_channel_parameters : dict
        The dictionary containing the parameters needed to initialise the
        phi_channel.
    use_em : bool
        The indicator of whether or not to use Expectation Maximazion (EM) to
        learn the prior parameters.
    adjust_tau_method : {'truncate', 'reweight'}
        The adjustment method to use if the EM-update of tau gets larger than
        one.

    Notes
    -----
    The above Parameters are the input channel parameters that must be passed
    in a `var` dict to the channel constructor.

    If `use_em` is True, the value given for `tau` is used for
    initialialisation. When using EM, the `phi_channel_parameters` are updated
    in alphabetical order.

    If the tau EM-update gets larger than one, it must be adjusted to avoid
    divergence of the GAMP algorithm. Two methods for this adjustment are
    available:

    * Truncate: Truncate tau to 1. (The default.)
    * Reweight: Adjust the weights to be close to unity weights.

    In addition to the above parameters it is assumed that the `var` dict
    includes the following keys:

    * 'n': The number of variables on which the channel acts.
    * 'convert': The precision conversion callable.

    """

    def __init__(self, var):
        super(GWS, self).__init__(var)

        @_decorate_validation
        def validate_channel_parameters():
            _generic(('var', 'input_channel_parameters'), 'mapping')
            _numeric(('var', 'input_channel_parameters', 'tau'),
                     ('integer', 'floating'), range_='[0;1]')
            _generic(('var', 'input_channel_parameters', 'phi_channel'),
                     'class', superclass=ValidatedBasicMMSEInputChannel)
            _generic(('var', 'input_channel_parameters',
                      'phi_channel_parameters'), 'mapping')
            _numeric(('var', 'input_channel_parameters', 'use_em'),
                     'boolean')
            _numeric(('var', 'n'), 'integer', range_='[0;inf)')
            _numeric(('var', 'input_channel_parameters', 'weights'),
                     'floating', range_='[0;1]', shape=(var['n'], 1),
                     ignore_none=True)
            _generic(('var', 'convert'), type)

            if 'adjust_tau_method' in var['input_channel_parameters']:
                _generic(
                    ('var', 'input_channel_parameters', 'adjust_tau_method'),
                    'string', value_in=('truncate', 'reweight'))

        validate_channel_parameters()

        c_params = var['input_channel_parameters']
        channel_init = copy.copy(var)
        channel_init['input_channel_parameters'] = c_params[
            'phi_channel_parameters']
        channel_init['input_channel_parameters']['use_em'] = False

        self._use_em = c_params['use_em']  # Whether or not to use EM learning
        self._n = var['convert'](var['n'])
        self._adjust_tau_method = c_params.get('adjust_tau_method', 'truncate')
        if c_params['weights'] is not None:
            self._weights = var['convert'](c_params['weights'])
        else:
            self._weights = None

        self.channel_parameters = _Configger(
            {'tau': var['convert'](c_params['tau'])},
            {'tau': _numeric(None, 'floating', range_='[0;1]')})
        self.phi_channel = c_params['phi_channel'](channel_init)
        self.phi_channel_parameters = self.phi_channel.channel_parameters

    def compute(self, var):
        """
        Compute the GWS input channel value.

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

        super(GWS, self).compute(var)

        r = var['r']
        s = var['s']
        tau = self.channel_parameters['tau']
        w = self._weights

        phi_mean, phi_variance = self.phi_channel.compute(var)

        norm_rv_pdf_zero = var['convert'](stats.norm(
            loc=np.float_(r), scale=np.sqrt(s, dtype=np.float_)).pdf(0))

        Z_phi = self.phi_channel.compute_Z(var)
        if w is not None:
            tw = tau * w
            pi = 1 / (1 + ((1 - tw) * norm_rv_pdf_zero / (tw * Z_phi)))
        else:
            pi = 1 / (1 + ((1 - tau) * norm_rv_pdf_zero / (tau * Z_phi)))

        # New values of alpha_bar (mean) and alpha_tilde (variance)
        mean = pi * phi_mean
        variance = pi * (phi_variance + phi_mean**2) - mean**2

        if self._use_em:
            # tau scaling
            if w is not None:
                ts = np.sum(w)
            else:
                ts = self._n

            # tau update
            tau = np.sum(pi) / ts
            if tau > 1:
                tau = self._adjust_tau(tau, pi, self._adjust_tau_method)
            self.channel_parameters['tau'] = var['convert'](tau)

            # phi channel parameters updates
            for phi_parameter in sorted(self.phi_channel_parameters.keys()):
                parameter_update = 1 / (ts * tau) * np.sum(
                    pi * self.phi_channel.get_EM_element(phi_parameter))
                self.phi_channel_parameters[phi_parameter] = var['convert'](
                    parameter_update)

        return mean, variance

    def _adjust_tau(self, tau, pi, method):
        """
        Adjust the value of tau if its EM-update is larger than one.

        Parameters
        ----------
        tau : float
            The current value of tau.
        pi : ndarray
            The current GAMP posterior support probabilities.
        method : {'truncate', 'reweight'}
            The adjustment method to use.

        Returns
        -------
        tau : float
            The adjusted value of tau.

        """

        if method == 'truncate':
            tau = 1
        elif method == 'reweight':
            pi_sum = np.sum(pi)
            while tau > 1:
                if self._weights.mean() > 0.95:
                    # Attempt to avoid (near) infinite loop by forcing
                    # equal weights if it seems reasonable
                    self._weights = np.ones_like(self._weights)
                else:
                    # Adjust weights to (hopefully) get tau=1
                    new_weights = self._weights * tau
                    new_weights[new_weights > 1] = 1
                    self._weights = new_weights

                tau = pi_sum / np.sum(self._weights)

        return tau


class IIDG(ValidatedBasicMMSEInputChannel):
    """
    An i.i.d. Gaussian MMSE input channel.

    Parameters
    ----------
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

    If `use_em` is True, the values given for `theta_bar`, and `theta_tilde`
    are used for initialialisation.

    In addition to the above parameters it is assumed that the `var` dict
    includes the following keys:

    * 'n': The number of variables on which the channel acts.
    * 'convert': The precision conversion callable.

    """

    def __init__(self, var):
        super(IIDG, self).__init__(var)

        @_decorate_validation
        def validate_channel_parameters():
            _generic(('var', 'input_channel_parameters'), 'mapping')
            _numeric(('var', 'input_channel_parameters', 'theta_bar'),
                     ('integer', 'floating'))
            _numeric(('var', 'input_channel_parameters', 'theta_tilde'),
                     ('integer', 'floating'), range_='[0;inf)')
            _numeric(('var', 'input_channel_parameters', 'use_em'),
                     'boolean')
            _numeric(('var', 'n'), 'integer', range_='[0;inf)')
            _generic(('var', 'convert'), type)

        validate_channel_parameters()

        c_params = var['input_channel_parameters']

        self._use_em = c_params['use_em']  # Whether or not to use EM learning
        self._EM_states = {'mean': np.nan, 'variance': np.nan}
        self._n = var['convert'](var['n'])

        self.channel_parameters = _Configger(
            {'theta_bar': var['convert'](c_params['theta_bar']),
             'theta_tilde': var['convert'](c_params['theta_tilde'])},
            {'theta_bar': _numeric(None, 'floating'),
             'theta_tilde': _numeric(None, 'floating', range_='[0;inf)')})

    def compute(self, var):
        """
        Compute the IIDG input channel value.

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

        super(IIDG, self).compute(var)

        s = var['s']
        r = var['r']
        theta_bar = self.channel_parameters['theta_bar']
        theta_tilde = self.channel_parameters['theta_tilde']

        # New values of alpha_bar (mean) and alpha_tilde (variance)
        mean = (theta_bar * s + theta_tilde * r) / (s + theta_tilde)
        variance = 1 / (1 / theta_tilde + 1 / s)

        self._EM_states['mean'] = mean
        self._EM_states['variance'] = variance

        if self._use_em:
            # EM update of theta_bar
            theta_bar = 1.0 / self._n * np.sum(mean)
            self.channel_parameters['theta_bar'] = var['convert'](theta_bar)

            # EM update of theta_tilde
            theta_tilde = 1.0 / self._n * np.sum(
                (mean - theta_bar)**2 + variance)
            self.channel_parameters['theta_tilde'] = var['convert'](
                theta_tilde)

        return mean, variance

    def compute_Z(self, var):
        """
        Compute the IIDG input channel normalisation constant.

        Parameters
        ----------
        var : dict
            The variables used in computing of the normalisation constant.

        Returns
        -------
        Z : ndarray
            The computed normalisation constant.

        """

        super(IIDG, self).compute_Z(var)

        s = var['s']
        r = var['r']
        theta_bar = self.channel_parameters['theta_bar']
        theta_tilde = self.channel_parameters['theta_tilde']

        Z = 1 / np.sqrt(2 * np.pi * (theta_tilde + s)) * np.exp(
            -(theta_bar - r)**2 / (2 * (theta_tilde + s)))

        return Z

    @_validate_once
    def get_EM_element(self, channel_parameter):
        """
        Return the element needed in computing the channel_parameter EM update.

        Parameters
        ----------
        channel_parameter : str
            The channel parameter for which the EM element is needed.

        Returns
        -------
        EM_element : ndarray
            The EM element needed for the channel parameter EM update.

        """

        super(IIDG, self).get_EM_element(channel_parameter)

        @_decorate_validation
        def validate_input():
            _generic('channel_parameter', 'string',
                     value_in=('theta_bar', 'theta_tilde'))

        validate_input()

        mean = self._EM_states['mean']
        variance = self._EM_states['variance']
        theta_bar = self.channel_parameters['theta_bar']

        if channel_parameter == 'theta_bar':
            EM_element = mean
        elif channel_parameter == 'theta_tilde':
            EM_element = (mean - theta_bar)**2 + variance

        return EM_element


class IIDL(ValidatedBasicMMSEInputChannel):
    """
    An i.i.d. Laplace MMSE input channel.

    This channel is a generalisation of the Laplacian prior detailed in [5]_.
    Specifically, the Laplace term is allowed to have a mean different from
    zero.

    Parameters
    ----------
    mu : float or int
        The prior Laplace mean.
    b : float or int
        The prior Laplace scale parameter (i.e. 1/lambda with lambda the rate
        parameter).
    use_em : bool
        The indicator of whether or not to use Expectation Maximazion (EM) to
        learn the prior parameters.

    Notes
    -----
    The above Parameters are the input channel parameters that must be passed
    in a `var` dict to the channel constructor.

    If `use_em` is True, the values given for `mu`, and `b` are
    used for initialialisation.

    In addition to the above parameters it is assumed that the `var` dict
    includes the following keys:

    * 'n': The number of variables on which the channel acts.
    * 'convert': The precision conversion callable.

    References
    ----------
    .. [5] J. Ziniel "Message Passing Approaches to Compressive Inference Under
       Structured Signal Priors", Ph.D. dissertation, Graduate School of The
       Ohio State University, 2014.

    """

    def __init__(self, var):
        super(IIDL, self).__init__(var)

        @_decorate_validation
        def validate_channel_parameters():
            _generic(('var', 'input_channel_parameters'), 'mapping')
            _numeric(('var', 'input_channel_parameters', 'mu'),
                     ('integer', 'floating'))
            _numeric(('var', 'input_channel_parameters', 'b'),
                     ('integer', 'floating'), range_='(0;inf)')
            _numeric(('var', 'input_channel_parameters', 'use_em'),
                     'boolean')
            _numeric(('var', 'n'), 'integer', range_='[0;inf)')
            _generic(('var', 'convert'), type)

        validate_channel_parameters()

        c_params = var['input_channel_parameters']

        self._use_em = c_params['use_em']  # Whether or not to use EM learning
        self._EM_states = {'all_u': np.nan, 'all_o': np.nan}
        self._n = var['convert'](var['n'])

        self._std_norm_rv = stats.norm()

        self.channel_parameters = _Configger(
            {'mu': var['convert'](c_params['mu']),
             'b': var['convert'](c_params['b'])},
            {'mu': _numeric(None, 'floating'),
             'b': _numeric(None, 'floating', range_='(0;inf)')})

    def compute(self, var):
        """
        Compute the IIDL input channel value.

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

        super(IIDL, self).compute(var)

        s = var['s']
        r = var['r']
        mu = self.channel_parameters['mu']
        b = self.channel_parameters['b']

        s_sqrt = np.sqrt(s)
        r_check = r - mu
        r_u = r_check + s / b
        r_o = r_check - s / b
        r_u_s = r_u / s_sqrt
        r_o_s = r_o / s_sqrt

        Z_I_u = 0.5 / b * np.exp(
            0.5 * s / b**2 + r_check / b) * var['convert'](
                self._std_norm_rv.cdf(np.float_(-r_u_s)))
        Z_I_o = 0.5 / b * np.exp(
            0.5 * s / b**2 - r_check / b) * var['convert'](
                self._std_norm_rv.cdf(np.float_(r_o_s)))

        npcr_u = self._npcr(-r_u_s)
        npcr_o = self._npcr(r_o_s)
        ETN_u = (r_u - s_sqrt * npcr_u)
        ETN_o = (r_o + s_sqrt * npcr_o)

        Z_I = Z_I_u + Z_I_o
        Z_I_u_rat = Z_I_u / Z_I
        Z_I_o_rat = Z_I_o / Z_I
        all_u = Z_I_u_rat * ETN_u
        all_o = Z_I_o_rat * ETN_o

        # New values of alpha_bar (mean) and alpha_tilde (variance)
        mean = mu + all_u + all_o
        variance = 2 * mu * mean - mu**2 + (
            Z_I_u_rat * (s * (1 - npcr_u * (npcr_u - r_u_s)) + ETN_u**2) +
            Z_I_o_rat * (s * (1 - npcr_o * (npcr_o + r_o_s)) + ETN_o**2)
        ) - mean**2

        self._EM_states['all_u'] = all_u
        self._EM_states['all_o'] = all_o

        if self._use_em:
            # EM update of b
            b = 1 / self._n * np.sum(all_o - all_u)
            self.channel_parameters['b'] = var['convert'](b)

            # EM update of mu
            mu = 1 / self._n * np.sum(mean)
            self.channel_parameters['mu'] = var['convert'](mu)

        return mean, variance

    def compute_Z(self, var):
        """
        Compute the IIDL input channel normalisation constant.

        Parameters
        ----------
        var : dict
            The variables used in computing of the normalisation constant.

        Returns
        -------
        Z : ndarray
            The computed normalisation constant.

        """

        super(IIDL, self).compute_Z(var)

        s = var['s']
        r = var['r']

        mu = self.channel_parameters['mu']
        b = self.channel_parameters['b']

        s_sqrt = np.sqrt(s)
        r_check = r - mu
        r_u_s = (r_check + s / b) / s_sqrt
        r_o_s = (r_check - s / b) / s_sqrt

        Z = 0.5 / b * np.exp(0.5 * s / b**2) * (
            var['convert'](self._std_norm_rv.cdf(np.float_(-r_u_s))) *
            np.exp(r_check / b) +
            var['convert'](self._std_norm_rv.cdf(np.float_(r_o_s))) /
            np.exp(r_check / b))

        return Z

    @_validate_once
    def get_EM_element(self, channel_parameter):
        """
        Return the element needed in computing the channel_parameter EM update.

        Parameters
        ----------
        channel_parameter : str
            The channel parameter for which the EM element is needed.

        Returns
        -------
        EM_element : ndarray
            The EM element needed for the channel parameter EM update.

        """

        super(IIDL, self).get_EM_element(channel_parameter)

        @_decorate_validation
        def validate_input():
            _generic('channel_parameter', 'string',
                     value_in=('mu', 'b'))

        validate_input()

        all_u = self._EM_states['all_u']
        all_o = self._EM_states['all_o']
        mu = self.channel_parameters['mu']

        if channel_parameter == 'mu':
            EM_element = mu + all_u + all_o
        elif channel_parameter == 'b':
            EM_element = all_o - all_u

        return EM_element

    def _npcr(self, x):
        """
        Return the pdf/cdf ratio of a standard normal RV evaluated at x.

        Parameters
        ----------
        x : ndarray
            The values to evaluate the ratio at.

        Returns
        -------
        npcr_values : ndarray
            The evaluated ratios.

        """

        convert = x.dtype.type
        npcr_values = (2 / np.sqrt(2 * np.pi)) / convert(special.erfcx(
            -np.float_(x / np.sqrt(2))))

        return npcr_values


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

    In addition to the above parameters it is assumed that the `var` dict
    includes the following keys:

    * 'n': The number of variables on which the channel acts.
    * 'convert': The precision conversion callable.

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
                     'boolean')
            _numeric(('var', 'n'), 'integer', range_='[0;inf)')
            _generic(('var', 'convert'), type)

        validate_channel_parameters()

        c_params = var['input_channel_parameters']

        self.use_em = c_params['use_em']  # Whether or not to use EM learning
        self.tau = var['convert'](c_params['tau'])
        self.theta_bar = var['convert'](c_params['theta_bar'])
        self.theta_tilde = var['convert'](c_params['theta_tilde'])
        self.n = var['convert'](var['n'])

        # GWS setup
        GWS_channel_params = {
            'tau': self.tau,
            'weights': None,
            'phi_channel': IIDG,
            'phi_channel_parameters': {
                'theta_bar': self.theta_bar, 'theta_tilde': self.theta_tilde,
                'use_em': False},
            'use_em': self.use_em}
        channel_init = copy.copy(var)
        channel_init['input_channel_parameters'] = GWS_channel_params
        self._GWS_channel = GWS(channel_init)

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

        # Offload computations to GWS channel
        mean, variance = self._GWS_channel.compute(var)
        self.tau = self._GWS_channel.channel_parameters['tau']
        self.theta_bar = self._GWS_channel.phi_channel_parameters['theta_bar']
        self.theta_tilde = self._GWS_channel.phi_channel_parameters[
            'theta_tilde']

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

    In addition to the above parameters it is assumed that the `var` dict
    includes the following keys:

    * 'n': The number of variables on which the channel acts.
    * 'm': The number of measurements on which the estimation is based.
    * 'convert': The precision conversion callable.

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
                     'boolean')
            _numeric(('var', 'n'), 'integer', range_='[0;inf)')
            _numeric(('var', 'm'), 'integer', range_='[0;inf)')
            _generic(('var', 'convert'), type)

            if var['input_channel_parameters']['use_em']:
                _numeric(('var', 'input_channel_parameters', 'em_damping'),
                         ('integer', 'floating'), range_='[0;1)')

        validate_channel_parameters()

        c_params = var['input_channel_parameters']
        m = var['convert'](var['m'])

        self.use_em = c_params['use_em']  # Whether or not to use EM learning
        if self.use_em:
            # EM damping level
            self.em_damp = var['convert'](c_params['em_damping'])
        self.tau = var['convert'](c_params['tau'])
        self.theta_bar = var['convert'](c_params['theta_bar'])
        self.theta_tilde = var['convert'](c_params['theta_tilde'])
        self.n = var['convert'](var['n'])
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
