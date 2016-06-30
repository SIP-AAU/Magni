"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing matrix emulators.

The matrix emulators of this module are wrappers of fast linear operations
giving the fast linear operations the same basic interface as a numpy ndarray.
Thereby allowing fast linear operations and numpy ndarrays to be used
interchangably in other parts of the package.

Routine listings
----------------
Matrix(magni.utils.validation.types.MatrixBase)
    Wrap fast linear operations in a matrix emulator.
MatrixCollection(magni.utils.validation.types.MatrixBase)
    Wrap multiple matrix emulators in a single matrix emulator.

See Also
--------
magni.imaging._fastops : Fast linear operations.

"""

from __future__ import division
import types

import numpy as np

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation.types import MatrixBase as _MatrixBase
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.validation import validate_once as _validate_once


class Matrix(_MatrixBase):
    """
    Wrap fast linear operations in a matrix emulator.

    `Matrix` defines a few attributes and internal methods which ensures that
    instances have the same basic interface as a numpy ndarray instance without
    explicitly forming the array. This basic interface allows instances to be
    multiplied with vectors, to be transposed, to be complex conjuagted, and to
    assume a shape. Also, instances have an attribute which explicitly forms
    the matrix as an ndarray.

    Parameters
    ----------
    func : function
        The fast linear operation applied to the vector when multiplying the
        matrix with a vector.
    conj_trans : function
        The fast linear operation applied to the vector when multiplying the
        complex conjuated transposed matrix with a vector.
    args : list or tuple
        The arguments which should be passed to `func` and `conj_trans` in
        addition to the vector.
    shape : list or tuple
        The shape of the emulated matrix.
    is_complex : bool
        The indicator of whether or not the emulated matrix is defined for the
        complex numbers (the default is False, which implies that the emulated
        matrix is defined for the real numbers only).
    is_valid : bool
        The indicator of whether or not the fast linear operation corresponds
        to a valid matrix (se discussion below) (the default is True, which
        implies that the matrix is considered valid).

    See Also
    --------
    magni.utils.validation.types.MatrixBase : Superclass of the present class.

    Notes
    -----
    The `is_valid` indicator is an implementation detail used to control
    possibly missing operations in the fast linear operation. For instance,
    consider the Discrete Fourier Transform (DFT) and its inverse transform.
    The forward DFT transform is a matrix-vector product involving the
    corresponding DFT matrix. The inverse transform is a matrix-vetor product
    involving the complex conjugate transpose DFT matrix. That is, it involves
    complex conjugation and transposition, both of which are (individually)
    valid transformations of the DFT matrix. However, when implementing the DFT
    (and its inverse) using a Fast Fourier Transform (FFT), only the combined
    (complex conjugate, transpose) operation is available. Thus, when using an
    FFT based `magni.util.matrices.Matrix`, one can get, e.g., a `Matrix.T`
    object corresponding to its transpose. However, the operation of computing
    a matrix-vector product involving the tranpose DFT matrix is not available
    and the `Matrix.T` is, consequently, considered an invalid matrix. Only the
    combined `Matrix.T.conj()` or `Matrix.conj().T` is considered a valid
    matrix.

    .. warning:: The tranpose operation changed in `magni` 1.5.0.

        For complex matrices, the the `.T` tranpose operation now yields an
        invalid matrix as described above. Prior to `magni` 1.5.0, the `.T`
        would yield the "inverse" which would usually be the complex conjugated
        transpose for complex matrices.


    Examples
    --------
    For example, the negative identity matrix could be emulated as

    >>> import numpy as np, magni
    >>> from magni.utils.matrices import Matrix
    >>> func = lambda vec: -vec
    >>> matrix = Matrix(func, func, (), (3, 3))

    The example matrix will have the desired shape:

    >>> matrix.shape
    (3, 3)

    The example matrix will behave just like an explicit matrix:

    >>> vec = np.float64([1, 2, 3]).reshape(3, 1)
    >>> np.set_printoptions(suppress=True)
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

    def __array__(self):
        """Return ndarray representation of the matrix."""
        return self.A

    def __init__(self, func, conj_trans, args, shape, is_complex=False,
                 is_valid=True):
        _MatrixBase.__init__(self, np.complex_ if is_complex else np.float_,
                             ((-np.inf, np.inf), (-np.inf, np.inf)), shape)

        @_decorate_validation
        def validate_input():
            _generic('func', 'function')
            _generic('conj_trans', 'function')
            _generic('args', 'explicit collection')
            _levels('shape', (_generic(None, 'explicit collection', len_=2),
                              _numeric(None, 'integer')))
            _numeric('is_complex', 'boolean')
            _numeric('is_valid', 'boolean')

        validate_input()

        self._func = func
        self._conj_trans = conj_trans
        self._args = args
        self._is_complex = is_complex
        self._is_valid = is_valid

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

        if not self._is_valid:
            raise ValueError('This matrix has not been implemented. Transpose '
                             'or conjugate it for an implemented matrix.')

        output = np.zeros(
            self.shape, dtype=np.complex_ if self._is_complex else np.float_)
        vec = np.zeros((self.shape[1], 1))

        for i in range(self.shape[1]):
            vec[i] = 1
            output[:, i] = self.dot(vec)[:, 0]
            vec[i] = 0

        return output

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
        the resulting matrix are the same as those of the current matrix except
        swapped. The shape is modified accordingly. This returns an `invalid`
        matrix if the entries are complex numbers as only the complex conjugate
        transpose is considered valid.

        """

        if self._is_complex:
            is_valid = not self._is_valid
        else:
            is_valid = True

        return Matrix(self._conj_trans, self._func, self._args,
                      self._shape[::-1], self._is_complex, is_valid)

    def conj(self):
        """
        Get the complex conjugate of the matrix.

        Returns
        -------
        matrix : Matrix
            The complex conjugate of the matrix.

        Notes
        -----
        The fast linear operation and the fast linear transposed operation of
        the resulting matrix are the same as those of the current matrix.
        This returns an `invalid` matrix if the entries are complex numbers as
        only the complex conjugate transpose is considered valid.

        """

        if self._is_complex:
            is_valid = not self._is_valid
        else:
            is_valid = True

        return Matrix(self._func, self._conj_trans, self._args, self._shape,
                      self._is_complex, is_valid)

    @_validate_once
    def dot(self, vec):
        """
        Multiply the matrix with a vector.

        Parameters
        ----------
        vec : numpy.ndarray
            The vector which the matrix is multiplied with.

        Returns
        -------
        vec : numpy.ndarray
            The result of the multiplication.

        Notes
        -----
        This method honors `magni.utils.validation.enable_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _numeric('vec', ('integer', 'floating', 'complex'),
                     shape=(self.shape[1], 1))

        validate_input()

        if not self._is_valid:
            raise ValueError('This matrix has not been implemented. Transpose '
                             'or conjugate it for an implemented matrix.')

        return self._func(vec, *self._args)


class MatrixCollection(_MatrixBase):
    """
    Wrap multiple matrix emulators in a single matrix emulator.

    `MatrixCollection` defines a few attributes and internal methods which
    ensures that instances have the same basic interface as a numpy ndarray
    instance without explicitly forming the ndarray. This basic interface
    allows instances to be multiplied with vectors, to be transposed, to be
    complex conjugated, and to assume a shape. Also, instances have an
    attribute which explicitly forms the matrix.

    Parameters
    ----------
    matrices : list or tuple
        The collection of `Matrix` instances.

    See Also
    --------
    magni.utils.validation.types.MatrixBase : Superclass of the present class.
    Matrix : Matrix emulator.

    Examples
    --------
    For example, two matrix emulators can be combined into one. That is, the
    matrix:

    >>> import numpy as np, magni
    >>> func = lambda vec: -vec
    >>> negate = magni.utils.matrices.Matrix(func, func, (), (3, 3))
    >>> np.set_printoptions(suppress=True)
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

    def __array__(self):
        """Return ndarray representation of the matrix."""
        return self.A

    def __init__(self, matrices):
        _MatrixBase.__init__(self, np.complex_,
                             ((-np.inf, np.inf), (-np.inf, np.inf)), None)

        @_decorate_validation
        def validate_input():
            _levels('matrices', (_generic(None, 'explicit collection'),
                                 _generic(None, (Matrix, MatrixCollection))))

            for i in range(len(matrices) - 1):
                if matrices[i].shape[1] != matrices[i + 1].shape[0]:
                    msg = ('The value of >>matrices[{}].shape[1]<<, {!r}, '
                           'must be equal to the value of '
                           '>>matrices[{}].shape[0]<<, {!r}.')
                    raise ValueError(msg.format(i, matrices[i].shape[1], i + 1,
                                                matrices[i + 1].shape[0]))

        validate_input()

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

        is_complex = any([matrix._is_complex for matrix in self._matrices])
        output = np.zeros(
            self.shape, dtype=np.complex_ if is_complex else np.float_)
        vec = np.zeros((self.shape[1], 1))

        for i in range(self.shape[1]):
            vec[i] = 1
            output[:, i] = self.dot(vec)[:, 0]
            vec[i] = 0

        return output

    @property
    def shape(self):
        """
        Get the shape of the matrix.

        The shape of the product of a number of matrices is the number of rows
        of the first matrix times the number of columns of the last matrix.

        Returns
        -------
        shape : tuple
            The shape of the matrix.

        """

        return (self._matrices[0].shape[0], self._matrices[-1].shape[1])

    @property
    def T(self):
        """
        Get the transpose of the matrix.

        The transpose of the product of the number of matrices is the product
        of the transpose of the matrices in reverse order.

        Returns
        -------
        matrix : MatrixCollection
            The transpose of the matrix.

        """

        return MatrixCollection([matrix.T for matrix in self._matrices[::-1]])

    def conj(self):
        """
        Get the complex conjugate of the matrix.

        The complex conjugate of the product of the number of matrices is the
        product of the complex conjugates of the matrices.

        Returns
        -------
        matrix : MatrixCollection
            The complex conjugate of the matrix.

        """

        return MatrixCollection([matrix.conj() for matrix in self._matrices])

    @_validate_once
    def dot(self, vec):
        """
        Multiply the matrix with a vector.

        Parameters
        ----------
        vec : numpy.ndarray
            The vector which the matrix is multiplied with.

        Returns
        -------
        vec : numpy.ndarray
            The result of the multiplication.

        Notes
        -----
        This method honors `magni.utils.validation.enable_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _numeric('vec', ('integer', 'floating', 'complex'),
                     shape=(self.shape[1], 1))

        validate_input()

        for matrix in self._matrices[::-1]:
            vec = matrix.dot(vec)

        return vec
