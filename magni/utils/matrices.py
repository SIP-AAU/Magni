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
Separable2DTransform(Matrix)
    Wrap a linear 2D separable transform in a matrix emulator.
SumApproximationMatrix(object)
    Wrap a sum approximation in a matrix emulator.
norm(A, ord=None)
    Compute a norm of a matrix.

See Also
--------
magni.imaging._fastops : Fast linear operations.
magni.imaging._mtx1D : 1D transforms matrices for use 2D separable transforms.

"""

from __future__ import division
import copy

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
        self._args = tuple(args)
        self._is_complex = is_complex
        self._is_valid = is_valid

    @property
    def matrix_state(self):
        """
        Return a copy of the internal matrix state.

        The internal matrix state consists of:

        * func: The forward fast linear operator.
        * conj_trans: The backward fast linear operator.
        * args: The tuple of extra arguments passed to `func` and `conj_trans`.
        * is_complex: The indicator of whether or not the matrix is complex.
        * is_valid: The indicator of whether or not the matrix is valid.

        Returns
        -------
        matrix_state : dict
            A copy of the internal matrix state

        """

        matrix_state = {'func': self._func,
                        'conj_trans': self._conj_trans,
                        'args': copy.deepcopy(self._args),
                        'is_complex': self._is_complex,
                        'is_valid': self._is_valid}

        return matrix_state

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
        The collection of matrices (e.g. ndarrays or `Matrix` instances).

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
                                 _numeric(
                                     None,
                                     ('integer', 'floating', 'complex'),
                                     shape=(-1, -1))))

            for i in range(len(matrices) - 1):
                if matrices[i].shape[1] != matrices[i + 1].shape[0]:
                    msg = ('The value of >>matrices[{}].shape[1]<<, {!r}, '
                           'must be equal to the value of '
                           '>>matrices[{}].shape[0]<<, {!r}.')
                    raise ValueError(msg.format(i, matrices[i].shape[1], i + 1,
                                                matrices[i + 1].shape[0]))

        validate_input()

        self._matrices = tuple(matrices)

    @property
    def matrix_state(self):
        """
        Return a copy of the internal matrix state.

        The internal matrix state consists of:

        * matrices: The tuple of matrices in the matrix collection

        Returns
        -------
        matrix_state : dict
            A copy of the internal matrix state

        """

        matrix_state = {'matrices': copy.deepcopy(self._matrices)}

        return matrix_state

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

        is_complex = any([getattr(matrix, '_is_complex', True)  # Default to
                          for matrix in self._matrices])       # complex matrix
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


class Separable2DTransform(Matrix):
    """
    Wrap a linear 2D separable transform in a matrix emulator.

    A linear 2D separable transform is defined by the two matrices `mtx_l` and
    `mtx_r` in the sense that the Kronecker product kron(`mtx_l`, `mtx_r`)
    yields the full matrix of the linear 2D separable transform when the linear
    transform is implemented as a matrix-vector product. See e.g. [1]_ for the
    details.

    `Separable2DTransform` defines a few attributes and internal methods which
    ensures that instances have the same basic interface as a numpy matrix
    instance without explicitly forming the matrix. This basic interface allows
    instances to be multiplied with vectors, to be transposed, and to assume a
    shape. Also, instances have an attribute which explicitly forms the matrix.

    Parameters
    ----------
    mtx_l : ndarray
        The "left" matrix in the defining Kronecker product.
    mtx_r : ndarray
        The "right" matrix in the defining Kronecker product.

    See Also
    --------
    magni.utils.matrices.Matrix : Superclass of the present class.

    References
    ----------
    .. [1] A.N. Akansu, R.A. Haddad, and P.R. Haddad, *Multiresolution Signal
       Decomposition: Transforms, Subbands, and Wavelets*, Academic Press,
       2000.

    Examples
    --------
    For example, the transform based on a 2-by-3 matrix and a 5-by-4 matrix

    >>> import numpy as np
    >>> from magni.imaging import vec2mat, mat2vec
    >>> from magni.utils.matrices import Separable2DTransform
    >>> mtx_l = np.arange(6).reshape(2, 3)
    >>> mtx_r = np.arange(40)[::2].reshape(5, 4)
    >>> sep_matrix = Separable2DTransform(mtx_l, mtx_r)

    The full transform matrix shape is:

    >>> print(tuple(int(s) for s in sep_matrix.shape))
    (10, 12)

    The matrix behaves like an explicit matrix

    >>> vec = np.arange(36)[::3].reshape(12, 1)
    >>> sep_matrix.dot(vec)
    array([[  936],
           [ 3528],
           [ 2712],
           [10056],
           [ 4488],
           [16584],
           [ 6264],
           [23112],
           [ 8040],
           [29640]])

    which, due to the separability of the transform, may also be computed as

    >>> vec_as_mat = vec2mat(vec, (3, 4))
    >>> mat2vec(mtx_l.dot(vec_as_mat.dot(mtx_r.T)))
    array([[  936],
           [ 3528],
           [ 2712],
           [10056],
           [ 4488],
           [16584],
           [ 6264],
           [23112],
           [ 8040],
           [29640]])

    The explicit matrix is given by the kronecker product of the two matrices

    >>> np.all(sep_matrix.A == np.kron(mtx_l, mtx_r))
    True

    The transpose of the matrix is also easily obtainable

    >>> sep_matrix_transpose = sep_matrix.T
    >>> sep_matrix_transpose.A
    array([[  0,   0,   0,   0,   0,   0,  24,  48,  72,  96],
           [  0,   0,   0,   0,   0,   6,  30,  54,  78, 102],
           [  0,   0,   0,   0,   0,  12,  36,  60,  84, 108],
           [  0,   0,   0,   0,   0,  18,  42,  66,  90, 114],
           [  0,   8,  16,  24,  32,   0,  32,  64,  96, 128],
           [  2,  10,  18,  26,  34,   8,  40,  72, 104, 136],
           [  4,  12,  20,  28,  36,  16,  48,  80, 112, 144],
           [  6,  14,  22,  30,  38,  24,  56,  88, 120, 152],
           [  0,  16,  32,  48,  64,   0,  40,  80, 120, 160],
           [  4,  20,  36,  52,  68,  10,  50,  90, 130, 170],
           [  8,  24,  40,  56,  72,  20,  60, 100, 140, 180],
           [ 12,  28,  44,  60,  76,  30,  70, 110, 150, 190]])

    """

    def __init__(self, mtx_l, mtx_r):
        _MatrixBase.__init__(self, np.complex_,
                             ((-np.inf, np.inf), (-np.inf, np.inf)),
                             (mtx_l.shape[0] * mtx_r.shape[0],
                              mtx_l.shape[1] * mtx_r.shape[1]))

        @_decorate_validation
        def validate_input():
            _numeric('mtx_l', ('integer', 'floating', 'complex'),
                     shape=(-1, -1))
            _numeric('mtx_r', ('integer', 'floating', 'complex'),
                     shape=(-1, -1))

        validate_input()

        self._mtx_l = mtx_l
        self._mtx_r = mtx_r
        self._is_complex = np.iscomplexobj(mtx_l) or np.iscomplexobj(mtx_r)

    @property
    def matrix_state(self):
        """
        Return a copy of the internal matrix state.

        The internal matrix state consists of:

        * mtx_l: The "left" matrix in the defining Kronecker product.
        * mtx_r: The "right" matrix in the defining Kronecker product.

        Returns
        -------
        matrix_state : dict
            A copy of the internal matrix state

        """

        matrix_state = {'mtx_l': self._mtx_l.copy(),
                        'mtx_r': self._mtx_r.copy()}

        return matrix_state

    @property
    def A(self):
        """
        Explicitly form the matrix.

        For a linear separable 2D transform, the full explicit transform matrix
        is given by the Kronecker product of the two matrices that define the
        separable transform.

        Returns
        -------
        matrix : ndarray
            The explicit matrix.

        """

        return np.kron(self._mtx_l, self._mtx_r)

    @property
    def T(self):
        """
        Get the transpose of the matrix.

        The transpose of a Kronekcer product of two matrices is the Kronecker
        product of the transposes of the two matrices.

        Returns
        -------
        matrix : Separable2DTransform
            The transpose of the matrix.

        """

        return Separable2DTransform(self._mtx_l.T, self._mtx_r.T)

    def conj(self):
        """
        Get the complex conjugate of the matrix.

        The complex conjugate of a Kronekcer product of two matrices is the
        Kronecker product of the complex conjugates of the two matrices.

        Returns
        -------
        matrix : Separable2DTransform
            The complex conjugate of the matrix.

        """

        return Separable2DTransform(self._mtx_l.conj(), self._mtx_r.conj())

    @_validate_once
    def dot(self, vec):
        """
        Multiply the matrix with a vector.

        The matrix-vector product is efficiently computed by
        mtx_l.dot(V.dot(mtx_r.T)) where V is the matrix from which `vec` was
        created by stacking its columns.

        Parameters
        ----------
        vec : ndarray
            The vector corresponding to the stacked columns of a matrix which
            this matrix is multiplied with.

        Returns
        -------
        vec : ndarray
            The result of the matrix-vector multiplication.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _numeric('vec', ('integer', 'floating', 'complex'),
                     shape=(self.shape[1], 1))

        validate_input()

        vec_as_mat = vec.reshape(self._mtx_r.shape[1], self._mtx_l.shape[1]).T
        mat_mult_result = self._mtx_l.dot(vec_as_mat.dot(self._mtx_r.T))
        mat_mult_result_as_vec = mat_mult_result.T.reshape(-1, 1)

        return mat_mult_result_as_vec


class SumApproximationMatrix(object):
    """
    Wrap a sum approximation in a matrix emulator.

    This simply emulates computing a scaled sum of the entries of a vector as a
    matrix vector product.

    Parameters
    ----------
    scaling : int or float
        The scaling applied to the sum approximation.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.utils.matrices import SumApproximationMatrix
    >>> np.random.seed(seed=6021)
    >>> n = 10
    >>> vec = np.random.randn(n, 1)
    >>> np_printoptions = np.get_printoptions()
    >>> np.set_printoptions(precision=5)
    >>> np.array([vec.mean()])
    array([ 0.43294])
    >>> matrix = SumApproximationMatrix(1.0/n)
    >>> np.array([matrix.dot(vec)])
    array([ 0.43294])
    >>> np.set_printoptions(**np_printoptions)

    """

    def __init__(self, scaling):
        @_decorate_validation
        def validate_input():
            _numeric('scaling', ('integer', 'floating'))

        validate_input()

        self.scaling = scaling

    @_validate_once
    def dot(self, vec):
        """
        Form the matrix-vector product sum approximation.

        Parameters
        ----------
        vec : ndarray
            The vector in the matrix-vector product.

        Returns
        -------
        scaled sum : float
            The scaled sum of the input vector.

        Notes
        -----
        This method honors `magni.utils.validation.enable_allow_validate_once`.

        """

        @_decorate_validation
        def validate_input():
            _numeric('vec', ('integer', 'floating', 'complex'), shape=(-1, 1))

        validate_input()

        return self.scaling * np.sum(vec)


def norm(A, ord=None):
    """
    Compute a norm of a matrix.

    Efficient norm computations for the `magni.utils.matrices` matrices.

    Parameters
    ----------
    A : magni.util.matrices.Matrix or magni.utils.matrices.MatrixCollection
        The matrix which norm is to be computed.
    ord : str
        The order of the norm to compute.

    Returns
    -------
    norm : float
        The computed matrix norm.

    See Also
    --------
    numpy.linalg.norm : Numpy's function for computing norms of ndarrays

    Notes
    -----
    This function is a simple wrapper around `numpy.linalg.norm`. It exploits
    some properties of the matrix classes in `magni.utils.matrices` to optimize
    the norm computation in terms of speed and/or memory usage.

    .. warning::

        This function builds the ndarray corresponding to the `A` matrix and
        passes that on to `numpy.linalg.norm` if no smarter way of computing
        the norm has been implemented. Beware of possible "out of memory"
        issues.

    Examples
    --------
    Compute the Frobenius norm of a Seperable2DTransform

    >>> import numpy as np
    >>> from magni.utils.matrices import Separable2DTransform, norm
    >>> mtx_l = np.arange(6).reshape(2, 3)
    >>> mtx_r = np.arange(40)[::2].reshape(5, 4)
    >>> sep_matrix = Separable2DTransform(mtx_l, mtx_r)
    >>> round(float(norm(sep_matrix, 'fro')), 2)
    737.16

    for comparison, the Frobenius computed using np.linalg.norm is

    >>> round(float(np.linalg.norm(sep_matrix.A)), 2)
    737.16

    """

    @_decorate_validation
    def validate_input():
        _numeric('A', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _generic('ord', 'string', ignore_none=True)

    validate_input()

    norm = None

    # Special computations
    if ord is None or ord == 'fro':
        if isinstance(A, Separable2DTransform):
            A_s = A.matrix_state
            norm = np.linalg.norm(A_s['mtx_l']) * np.linalg.norm(A_s['mtx_r'])

    # Fallback numpy solution
    if norm is None:
        norm = np.linalg.norm(A)

    return norm
