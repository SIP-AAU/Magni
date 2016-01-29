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
