"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing abstract superclasses for validation.

Routine listings
----------------
MatrixBase(object)
    Abstract base class of custom matrix classes.
MMSEInputChannel(object)
    Abstract base class of a Minimum Mean Squaread Error (MMSE) input channel.
StopCriterion(object)
    Abstract base class of a stop criterion.
ThresholdOperator(object)
    Abstract base class of a threshold operator.

"""


class MatrixBase(object):
    """
    Abstract base class of custom matrix classes.

    The `magni.utils.validation.validate_numeric` function accepts built-in
    numeric types, numpy built-in numeric types, and subclasses of the present
    class. In order to perform validation checks, the validation function needs
    to know the data type, the bounds, and the shape of the variable. Thus,
    subclasses must call the init function of the present class with these
    arguments.

    Parameters
    ----------
    dtype : type
        The data type of the values of the instance.
    bounds : list or tuple
        The bounds of the values of the instance.
    shape : list or tuple
        The shape of the instance.

    Attributes
    ----------
    bounds : list or tuple
        The bounds of the values of the instance.
    dtype : type
        The data type of the values of the instance.
    shape : list or tuple
        The shape of the instance.

    Notes
    -----
    `dtype` is either a built-in numeric type or a numpy built-in numeric type.

    If the matrix has complex values, `bounds` is a list with two values; The
    bounds on the real values and the bounds on the imaginary values. If, on
    the other hand, the matrix has real values, `bounds` has one value; The
    bounds on the real values. Each such bounds value is a list with two real,
    numeric values; The lower bound (that is, the minimum value) and the upper
    bound (that is, the maximum value).

    """

    def __init__(self, dtype, bounds, shape):
        self._dtype = dtype
        self._bounds = bounds
        self._shape = shape

    bounds = property(lambda self: self._bounds)
    dtype = property(lambda self: self._dtype)
    shape = property(lambda self: self._shape)


class MMSEInputChannel(object):
    """
    Abstract base class of a Minimum Mean Squaread Error (MMSE) input channel.

    The `magni.cs.reconstruction` algorithms may make use of input channels to
    define the prior knowledge on the sought solution. In order for the
    validation of such an input channel to work, it must be based on this
    class.

    Parameters
    ----------
    var : dict
        The input channel state variables.

    """

    def __init__(self, var):
        pass

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

        """

        raise NotImplementedError(
            'Subclasses of MMSEInputChannel must override this method.')


class MMSEOutputChannel(object):
    """
    Abstract base class of a Minimum Mean Squaread Error (MMSE) output channel.

    The `magni.cs.reconstruction` algorithms may make use of output channels to
    define the observation model. In order for the validation of such an output
    channel to work, it must be based on this class.

    Parameters
    ----------
    var : dict
        The output channel state variables.

    """

    def __init__(self, var):
        pass

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

        """

        raise NotImplementedError(
            'Subclasses of MMSEOutputChannel must override this method.')


class StopCriterion(object):
    """
    Abstract base class of a stop criterion.

    The `magni.cs.reconstruction` algorithms are typically iterative algorithms
    that make use of some stop criterion to determine if it has converged. In
    order for the validation of such a stop criterion to work, it must be based
    on this class.

    Parameters
    ----------
    var : dict
        The stop criterion state variables.

    """

    def __init__(self, var):
        pass

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

        """

        raise NotImplementedError(
            'Subclasses of StopCriterion must override this method.')


class ThresholdOperator(object):
    """
    Abstract base class of a threshold operator.

    The `magni.cs.reconstruction` algorithms may make use of threshold
    operators for "de-noising". In order for the validation of such a threshold
    operator to work, it must be based on this class.

    Parameters
    ----------
    var : dict
        The threshold operator state variables.

    """

    def __init__(self, var):
        pass

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

        """

        raise NotImplementedError(
            'Subclasses of ThresholdOperator must override this method.')

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

        """

        raise NotImplementedError(
            'Subclasses of ThresholdOperator must override this method.')

    def update_threshold_level(self, var):
        """
        Update the threshold level state.

        Parameters
        ----------
        var : dict
            The variables used in computing the threshold level update.

        """

        raise NotImplementedError(
            'Subclasses of ThresholdOperator must override this method.')
