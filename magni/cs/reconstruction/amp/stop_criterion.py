"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing stop criteria for the Approximate Message Passing (AMP)
algorithm.

Routine listings
----------------
ValidatedStopCriterion(magni.utils.validation.types.StopCriterion)
    A base class for validated `magni.cs.reconstruction.amp` stop criteria.
MSEConvergence(ValidatedStopCriterion)
    A mean square error (MSE) convergence stop criterion.
NormalisedMSEConvergence(ValidatedStopCriterion)
    A normalised mean squaure error (NMSE) convergence stop criterion.
Residual(ValidatedStopCriterion)
    A residual based stop criterion.
ResidualMeasurementsRatio(ValidatedStopCriterion)
    A residual-measurements-ratio based stop criterion.

"""

from __future__ import division

import numpy as np

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_once as _validate_once
from magni.utils.validation.types import StopCriterion as _StopCriterion


class ValidatedStopCriterion(_StopCriterion):
    """
    A base class for validated `magni.cs.reconstruction.amp` stop criteria.

    Parameters
    ----------
    var : dict
        The stop criterion state variables.

    """

    def __init__(self, var):
        super(ValidatedStopCriterion, self).__init__(var)

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()

    @_validate_once
    def compute(self, var):
        """
        Compute the stop criterion value.

        Parameters
        ----------
        var : dict
            The variables used in computing of the stop criterion value.

        Returns
        -------
        stop : bool
            The indicator of whether or not the stop criterion is satisfied.
        value : float
            The stop criterion value.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _generic('var', 'mapping')

        validate_input()


class MSEConvergence(ValidatedStopCriterion):
    """
    A mean square error (MSE) convergence stop criterion.

    Parameters
    ----------
    var : dict
        The stop criterion state variables.

    Notes
    -----
    The following state variables are used in this stop criterion:

    * A
    * tolerance

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.cs.reconstruction.amp.stop_criterion import MSEConvergence
    >>> state = {'tolerance': 1e-3, 'A': np.ones((10,10))}
    >>> variables = {'alpha_bar_prev': np.ones((10, 1)),
    ... 'alpha_bar': np.arange(10).reshape(10, 1)}
    >>> MSE = MSEConvergence(state)
    >>> stop, val = MSE.compute(variables)
    >>> stop
    False
    >>> np.round(val, 2)
    20.5

    """

    def __init__(self, var):
        super(MSEConvergence, self).__init__(var)
        self.n = var['A'].shape[1]
        self.tolerance = var['tolerance']

    def compute(self, var):
        """
        Compute the MSE convergence stop criterion value.

        The AMP algorithm should converge to a fixed point. This criterion
        is based on the mean squared error of the difference between the
        proposed solution in this iteration and the proposed solution in the
        previous solution.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in calculating of the stop criterion.

        Returns
        -------
        stop : bool
            The indicator of whether or not the stop criterion is satisfied.
        value : float
            The stop criterion value.

        """

        super(MSEConvergence, self).compute(var)

        mse = 1/self.n * np.linalg.norm(
            var['alpha_bar_prev'] - var['alpha_bar'])**2
        stop = mse < self.tolerance

        return stop, mse


class NormalisedMSEConvergence(ValidatedStopCriterion):
    """
    A normalised mean squaure error (NMSE) convergence stop criterion.

    Parameters
    ----------
    var : dict
        The stop criterion state variables.

    Notes
    -----
    The following state variables are used in this stop criterion:

    * tolerance

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.cs.reconstruction.amp.stop_criterion import (
    ... NormalisedMSEConvergence)
    >>> state = {'tolerance': 1e-3}
    >>> variables = {'alpha_bar_prev': np.ones((10, 1)),
    ... 'alpha_bar': np.arange(10).reshape(10, 1)}
    >>> NMSE = NormalisedMSEConvergence(state)
    >>> stop, val = NMSE.compute(variables)
    >>> stop
    False
    >>> np.round(val, 2)
    20.5

    """

    def __init__(self, var):
        super(NormalisedMSEConvergence, self).__init__(var)
        self.tolerance = var['tolerance']

    def compute(self, var):
        """
        Compute the normalised MSE convergence stop criterion value.

        The AMP algorithm should converge to a fixed point. This criterion
        is based on the mean squared error of the difference between the
        proposed solution in this iteration and the proposed solution in the
        previous solution normalised by the mean squared error of the proposed
        solution in the previous iteration.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in calculating of the stop criterion.

        Returns
        -------
        stop : bool
            The indicator of whether or not the stop criterion is satisfied.
        value : float
            The stop criterion value.

        """

        super(NormalisedMSEConvergence, self).compute(var)

        se = np.linalg.norm(var['alpha_bar_prev'] - var['alpha_bar'])**2
        norm = np.linalg.norm(var['alpha_bar_prev'])**2
        if norm > 0:
            value = se / norm
            stop = se < self.tolerance * norm
        else:
            # Previous solution was zero-vector (most likely first iteration)
            value = se
            stop = False

        return stop, value


class Residual(ValidatedStopCriterion):
    """
    A residual based stop criterion.

    Parameters
    ----------
    var : dict
        The stop criterion state variables.

    Notes
    -----
    The following state variables are used in this stop criterion:

    * y
    * tolerance
    * A

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.cs.reconstruction.amp.stop_criterion import Residual
    >>> state = {'tolerance': 1e-3, 'y': np.ones((10, 1)),
    ... 'A': np.ones((10,10))}
    >>> variables = {'A_dot_alpha_bar': np.arange(10).reshape(10, 1)}
    >>> Res = Residual(state)
    >>> stop, val = Res.compute(variables)
    >>> stop
    False
    >>> np.round(val, 2)
    20.5

    """

    def __init__(self, var):
        super(Residual, self).__init__(var)
        self.y = var['y']
        self.tolerance = var['tolerance']
        self.m = var['A'].shape[0]

    def compute(self, var):
        """
        Compute the residual stop criterion value.

        If the noise level is (approximately) known, the AMP iterations may be
        stopped once the residual is on the order of the noise level. This
        stopping criterion is based on the mean sqaured error of the residual.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in calculating of the stop criterion.

        Returns
        -------
        stop : bool
            The indicator of whether or not the stop criterion is satisfied.
        value : float
            The stop criterion value.

        """

        super(Residual, self).compute(var)

        r_mse = 1/self.m * np.linalg.norm(self.y - var['A_dot_alpha_bar'])**2
        stop = r_mse < self.tolerance

        return stop, r_mse


class ResidualMeasurementsRatio(ValidatedStopCriterion):
    """
    A residual-measurements-ratio based stop criterion.

    Parameters
    ----------
    var : dict
        The stop criterion state variables.

    Notes
    -----
    The following state variables are used in this stop criterion:

    * y
    * tolerance

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.cs.reconstruction.amp.stop_criterion import (
    ... ResidualMeasurementsRatio)
    >>> state = {'tolerance': 1e-3, 'y': np.ones((10, 1))}
    >>> variables = {'A_dot_alpha_bar': np.arange(10).reshape(10, 1)}
    >>> ResMeasRat = ResidualMeasurementsRatio(state)
    >>> stop, val = ResMeasRat.compute(variables)
    >>> stop
    False
    >>> np.round(val, 2)
    14.32

    """

    def __init__(self, var):
        super(ResidualMeasurementsRatio, self).__init__(var)
        self.y = var['y']
        self.tolerance = var['tolerance']

    def compute(self, var):
        """
        Compute the residual-measurements-ratio stop criterion value.

        If the noise level is (approximately) known, the AMP iterations may be
        stopped once the residual is on the order of the noise level. This
        stopping criterion is based on ratio of the mean sqaured error of the
        residual to the mean squared error of the measurements.

        Parameters
        ----------
        var : dict
            Dictionary of variables used in calculating of the stop criterion.

        Returns
        -------
        stop : bool
            The indicator of whether or not the stop criterion is satisfied.
        value : float
            The stop criterion value.

        """

        super(ResidualMeasurementsRatio, self).compute(var)

        r_norm = np.linalg.norm(self.y - var['A_dot_alpha_bar'])
        stop = r_norm < self.tolerance * np.linalg.norm(self.y)

        return stop, r_norm
