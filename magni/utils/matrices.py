"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing matrix emulators.

The matrix emulators of this module are wrappers of fast linear operations
giving the fast linear operations the same basic interface as a numpy ndarray.
Thereby allowing fast linear operations and numpy ndarrays to be used
interchangably in other parts of the package.

Routine listings
----------------
Matrix()
    Wrap fast linear operations in a matrix emulator.
MatrixCollection()
    Wrap multiple matrix emulators in a single matrix emulator.

See Also
--------
magni.imaging._fastops : Fast linear operations.

"""

from __future__ import division
import types

import numpy as np

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate as _validate
from magni.utils.validation import validate_ndarray as _validate_ndarray


class Matrix():
    """
    Wrap fast linear operations in a matrix emulator.

    `Matrix` defines a few attributes and internal methods which ensures that
    instances have the same basic interface as a numpy matrix instance without
    explicitly forming the matrix. This basic interface allows instances to be
    multiplied with vectors, to be transposed, and to assume a shape. Also,
    instances have an attribute which explicitly forms the matrix.

    Parameters
    ----------
    func : function
        The fast linear operation applied to the vector when multiplying the
        matrix with a vector.
    trans : function
        The fast linear operation applied to the vector when multiplying the
        transposed matrix with a vector.
    args : list or tuple
        The arguments which should be passed to `func` and `trans` in addition
        to the vector.
    shape : list or tuple
        The shape of the emulated matrix.

    Examples
    --------
    For example, the negative identity matrix could be emulated as

    >>> from magni.utils.matrices import Matrix
    >>> func = lambda vec: -vec
    >>> matrix = Matrix(func, func, (), (3, 3))

    The example matrix will have the desired shape:

    >>> matrix.shape
    (3, 3)

    The example matrix will behave just like an explicit matrix:

    >>> vec = np.float64([1, 2, 3]).reshape(3, 1)
    >>> matrix.dot(vec)
    array([[-1.],
           [-2.],
           [-3.]])

    If, at some point, an explicit representation of the matrix is required,
    this can easily be obtained:

    >>> matrix.A
    array([[-1., -0., -0.],
           [-0., -1., -0.],
           [-0., -0., -1.]])

    Likewise, the transpose of the matrix can be obtained:

    >>> matrix.T.A
    array([[-1., -0., -0.],
           [-0., -1., -0.],
           [-0., -0., -1.]])

    """

    @_decorate_validation
    def _validate_init(self, func, trans, args, shape):
        """
        Validate the `__init__` function.

        See Also
        --------
        Matrix.__init__ : The validated function.
        magni.utils.validation.validate : Validation.

        """

        _validate(func, 'func', {'type': types.FunctionType})
        _validate(trans, 'trans', {'type': types.FunctionType})
        _validate(args, 'args', {'type_in': (list, tuple)})
        _validate(shape, 'shape', [{'type_in': (list, tuple)}, {'type': int}])

    def __init__(self, func, trans, args, shape):
        self._validate_init(func, trans, args, shape)

        self._func = func
        self._trans = trans
        self._args = args
        self._shape = shape

    @property
    def A(self):
        """
        Explicitly form the matrix.

        The fast linear operations implicitly define a matrix which is usually
        not explicitly formed. However, some functionality might require a more
        advanced matrix interface than that provided by this class.

        Returns
        -------
        matrix : numpy.ndarray
            The explicit matrix.

        Notes
        -----
        The explicit matrix is formed by multiplying the matrix with the
        columns of an identity matrix and stacking the resulting vectors as
        columns in a matrix.

        """

        output = np.zeros(self.shape, dtype=np.complex128)
        vec = np.zeros((self.shape[1], 1))

        for i in range(self.shape[1]):
            vec[i] = 1
            output[:, i] = self.dot(vec)[:, 0]
            vec[i] = 0

        if np.abs(output.imag).sum() == 0:
            output = output.real

        return output

    @property
    def shape(self):
        """
        Get the shape of the matrix.

        Returns
        -------
        shape : tuple
            The shape of the matrix.

        """

        return self._shape

    @property
    def T(self):
        """
        Get the transpose of the matrix.

        Returns
        -------
        matrix : Matrix
            The transpose of the matrix.

        Notes
        -----
        The fast linear operation and the fast linear transposed operation of
        the resulting matrix are same as those of the current matrix except
        swapped. The shape is modified accordingly.

        """

        return Matrix(self._trans, self._func, self._args, self._shape[::-1])

    @_decorate_validation
    def _validate_dot(self, vec):
        """
        Validate the `dot` function.

        See Also
        --------
        Matrix.dot : The validated function.
        magni.utils.validation.validate_matrix : Validation.

        """

        _validate_ndarray(vec, 'vec', {'shape': (self.shape[1], 1)})

    def dot(self, vec):
        """
        Multiply the matrix with a vector.

        Parameters
        ----------
        vec : numpy.ndarray
            The vector which the matrix is multiplied with.

        Returns
        -------
        vec : numpy.matrix
            The result of the multiplication.

        """

        self._validate_dot(vec)

        return self._func(vec, *self._args)


class MatrixCollection():
    """
    Wrap multiple matrix emulators in a single matrix emulator.

    `MatrixCollection` defines a few attributes and internal methods which
    ensures that instances have the same basic interface as a numpy matrix
    instance without explicitly forming the matrix. This basic interface allows
    instances to be multiplied with vectors, to be transposed, and to assume a
    shape. Also, instances have an attribute which explicitly forms the matrix.

    Parameters
    ----------
    matrices : list or tuple
        The collection of `Matrix` instances.

    See Also
    --------
    Matrix : Matrix emulator.

    Examples
    --------
    For example, two matrix emulators can be combined into one. That is, the
    matrix:

    >>> func = lambda vec: -vec
    >>> negate = magni.utils.matrices.Matrix(func, func, (), (3, 3))
    >>> negate.A
    array([[-1., -0., -0.],
           [-0., -1., -0.],
           [-0., -0., -1.]])

    And the matrix:

    >>> func = lambda vec: vec[::-1]
    >>> reverse = magni.utils.matrices.Matrix(func, func, (), (3, 3))
    >>> reverse.A
    array([[ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.]])

    Can be combined into one matrix emulator using the present class:

    >>> from magni.utils.matrices import MatrixCollection
    >>> matrix = MatrixCollection((negate, reverse))

    The example matrix will have the desired shape:

    >>> matrix.shape
    (3, 3)

    The example matrix will behave just like an explicit matrix:

    >>> vec = np.float64([1, 2, 3]).reshape(3, 1)
    >>> matrix.dot(vec)
    array([[-3.],
           [-2.],
           [-1.]])

    If, at some point, an explicit representation of the matrix is required,
    this can easily be obtained:

    >>> matrix.A
    array([[-0., -0., -1.],
           [-0., -1., -0.],
           [-1., -0., -0.]])

    Likewise, the transpose of the matrix can be obtained:

    >>> matrix.T.A
    array([[-0., -0., -1.],
           [-0., -1., -0.],
           [-1., -0., -0.]])

    """

    @_decorate_validation
    def _validate_init(self, matrices):
        """
        Validate the `__init__` function.

        See Also
        --------
        MatrixCollection.__init__ : The validated function.
        magni.utils.validation.validate : Validation.

        """

        _validate(matrices, 'matrices',
                  [{'type_in': (list, tuple)}, {'class': Matrix}])

        for i in range(len(matrices) - 1):
            if matrices[0].shape[1] != matrices[1].shape[0]:
                raise ValueError('The matrices must have compatible shapes.')

    def __init__(self, matrices):
        self._validate_init(matrices)

        self._matrices = matrices

    @property
    def A(self):
        """
        Explicitly form the matrix.

        The collection of matrices implicitly defines a matrix which is usually
        not explicitly formed. However, some functionality might require a more
        advanced matrix interface than that provided by this class.

        Returns
        -------
        matrix : numpy.ndarray
            The explicit matrix.

        Notes
        -----
        The explicit matrix is formed by multiplying the matrix with the
        columns of an identity matrix and stacking the resulting vectors as
        columns in a matrix.

        """

        output = np.zeros(self.shape, dtype=np.complex128)
        vec = np.zeros((self.shape[1], 1))

        for i in range(self.shape[1]):
            vec[i] = 1
            output[:, i] = self.dot(vec)[:, 0]
            vec[i] = 0

        if np.abs(output.imag).sum() == 0:
            output = output.real

        return output

    @property
    def shape(self):
        """
        Get the shape of the matrix.

        Returns
        -------
        shape : tuple
            The shape of the matrix.

        Notes
        -----
        The shape of the product of a number of matrices is the number of rows
        of the first matrix times the number of columns of the last matrix.

        """

        return (self._matrices[0].shape[0], self._matrices[-1].shape[1])

    @property
    def T(self):
        """
        Get the transpose of the matrix.

        Returns
        -------
        matrix : MatrixCollection
            The transpose of the matrix.

        Notes
        -----
        The transpose of the product of the number of matrices is the product
        of the transpose of the matrices in reverse order.

        """

        return MatrixCollection([matrix.T for matrix in self._matrices[::-1]])

    @_decorate_validation
    def _validate_dot(self, vec):
        """
        Validate the `dot` function.

        See Also
        --------
        MatrixCollection.dot : The validated function.
        magni.utils.validation.validate_matrix : Validation.

        """

        _validate_ndarray(vec, 'vec', {'shape': (self.shape[1], 1)})

    def dot(self, vec):
        """
        Multiply the matrix with a vector.

        Parameters
        ----------
        vec : numpy.matrix
            The vector which the matrix is multiplied with.

        Returns
        -------
        vec : numpy.matrix
            The result of the multiplication.

        """

        self._validate_dot(vec)

        for matrix in self._matrices[::-1]:
            vec = matrix.dot(vec)

        return vec
