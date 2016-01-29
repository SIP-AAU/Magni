"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module prodiving performance indicator determination functionality.

Routine listings
----------------
calculate_coherence(Phi, Psi, norm=None)
    Calculate the coherence of the Phi Psi matrix product.
calculate_mutual_coherence(Phi, Psi, norm=None)
    Calculate the mutual coherence of Phi and Psi.
calculate_relative_energy(Phi, Psi, method=None)
    Calculate the relative energy of Phi Psi matrix product atoms.

"""

from __future__ import division

import numpy as np

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.validation import validate_generic as _generic


def calculate_coherence(Phi, Psi, norm=None):
    r"""
    Calculate the coherence of the Phi Psi matrix product.

    In the context of Compressive Sensing, coherence usually refers to the
    maximum absolute correlation between two columns of the Phi Psi matrix
    product. This function allows the usage of a different normalised norm
    where the infinity-norm yields the usual case.

    Parameters
    ----------
    Phi : magni.utils.matrices.Matrix or numpy.ndarray
        The measurement matrix.
    Psi : magni.utils.matrices.Matrix or numpy.ndarray
        The dictionary matrix.
    norm : int or float
        The normalised norm used for the calculation (the default value is None
        which implies that the 0-, 1-, 2-, and infinity-norms are returned).

    Returns
    -------
    coherence : float or dict
        The coherence value(s).

    Notes
    -----
    If `norm` is None, the function returns a dict containing the coherence
    using the 0-, 1-, 2-, and infinity-norms. Otherwise, the function returns
    the coherence using the specified norm.

    The coherence is calculated as:

    .. math::

        \left(\frac{1}{n^2 - n}
        \sum_{i = 1}^n \sum_{\substack{j = 1 \\ j \neq i}}^n \left(
        \frac{|\Psi_{:, i}^T \Phi^T \Phi \Psi_{:, j}|}
        {||\Phi \Psi_{:, i}||_2 ||\Phi \Psi_{:, j}||_2}
        \right)^{\text{norm}}\right)^{\frac{1}{\text{norm}}}

    where `n` is the number of columns in `Psi`. In the case of the 0-norm,
    the coherence is calculated as:

    .. math::

        \frac{1}{n^2 - n}
        \sum_{i = 1}^n \sum_{\substack{j = 1 \\ j \neq i}}^n \mathbf{1}
        \left(\frac{|\Psi_{:, i}^T \Phi^T \Phi \Psi_{:, j}|}
        {||\Phi \Psi_{:, i}||_2 ||\Phi \Psi_{:, j}||_2}\right)

    where :math:`\mathbf{1}(a)` is 1 if `a` is non-zero and 0 otherwise. In the
    case of the infinity-norm, the coherence is calculated as:

    .. math::

        \max_{\substack{i, j \in \{1, \dots, n\} \\ i \neq j}}
        \left(\frac{|\Psi_{:, i}^T \Phi^T \Phi \Psi_{:, j}|}
        {||\Phi \Psi_{:, i}||_2 ||\Phi \Psi_{:, j}||_2}\right)

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> import magni
    >>> from magni.cs.indicators import calculate_coherence
    >>> Phi = np.zeros((5, 9))
    >>> Phi[0, 0] = Phi[1, 2] = Phi[2, 4] = Phi[3, 6] = Phi[4, 8] = 1
    >>> Psi = magni.imaging.dictionaries.get_DCT((3, 3))
    >>> for item in sorted(calculate_coherence(Phi, Psi).items()):
    ...     print('{}-norm: {:.3f}'.format(*item))
    0-norm: 0.222
    1-norm: 0.141
    2-norm: 0.335
    inf-norm: 1.000

    The above values can be calculated individually by specifying a norm:

    >>> for norm in (0, 1, 2, np.inf):
    ...     value = calculate_coherence(Phi, Psi, norm=norm)
    ...     print('{}-norm: {:.3f}'.format(norm, value))
    0-norm: 0.222
    1-norm: 0.141
    2-norm: 0.335
    inf-norm: 1.000

    """

    @_decorate_validation
    def validate_input():
        _numeric('Phi', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _numeric('Psi', ('integer', 'floating', 'complex'),
                 shape=(Phi.shape[1], -1))
        _numeric('norm', ('integer', 'floating'), range_='[0;inf]',
                 ignore_none=True)

    validate_input()

    A = np.zeros((Phi.shape[0], Psi.shape[1]))
    e = np.zeros((A.shape[1], 1))

    for i in range(A.shape[1]):
        e[i] = 1
        A[:, i] = Phi.dot(Psi.dot(e)).reshape(-1)
        e[i] = 0

    M = np.zeros((A.shape[1], A.shape[1]))
    PhiT = Phi.T
    PsiT = Psi.T

    for i in range(A.shape[1]):
        M[:, i] = np.abs(PsiT.dot(PhiT.dot(
            A[:, i].reshape(-1, 1)))).reshape(-1)
        M[i, i] = 0

    w = 1 / np.linalg.norm(A, axis=0).reshape(-1, 1)
    M = M * w * w.T

    if norm is None:
        entries = (M.size - M.shape[0])
        value = {0: np.sum(M > 1e-9) / entries,
                 1: np.sum(M) / entries,
                 2: (np.sum(M**2) / entries)**(1 / 2),
                 np.inf: np.max(M)}
    elif norm == 0:
        value = np.sum(M > 1e-9) / (M.size - M.shape[0])
    elif norm == np.inf:
        value = np.max(M)
    else:
        value = (np.sum(M**norm) / (M.size - M.shape[0]))**(1 / norm)

    return value


def calculate_mutual_coherence(Phi, Psi, norm=None):
    r"""
    Calculate the mutual coherence of Phi and Psi.

    In the context of Compressive Sensing, mutual coherence usually refers to
    the maximum absolute correlation between two columns of Phi and Psi. This
    function allows the usage of a different normalised norm where the
    infinity-norm yields the usual case.

    Parameters
    ----------
    Phi : magni.utils.matrices.Matrix or numpy.ndarray
        The measurement matrix.
    Psi : magni.utils.matrices.Matrix or numpy.ndarray
        The dictionary matrix.
    norm : int or float
        The normalised norm used for the calculation (the default value is None
        which implies that the 0-, 1-, 2-, and infinity-norms are returned).

    Returns
    -------
    mutual_coherence : float or dict
        The mutual_coherence value(s).

    Notes
    -----
    If `norm` is None, the function returns a dict containing the mutual
    coherence using the 0-, 1-, 2-, and infinity-norms. Otherwise, the function
    returns the mutual coherence using the specified norm.

    The mutual coherence is calculated as:

    .. math::

        \left(\frac{1}{m n} \sum_{i = 1}^m \sum_{j = 1}^n
        |\Phi_{i, :} \Psi_{:, j}|^{\text{norm}}\right)^{\frac{1}{\text{norm}}}

    where `m` is the number of rows in `Phi` and `n` is the number of columns
    in `Psi`. In the case of the 0-norm, the mutual coherence is calculated as:

    .. math::

        \frac{1}{m n} \sum_{i = 1}^m \sum_{j = 1}^n \mathbf{1}
        (|\Phi_{i, :} \Psi_{:, j}|)

    where :math:`\mathbf{1}(a)` is 1 if `a` is non-zero and 0 otherwise. In the
    case of the infinity-norm, the mutual coherence is calculated as:

    .. math::

        \max_{i \in \{1, \dots, m\}, j \in \{1, \dots, n\}}
        |\Phi_{i, :} \Psi_{:, j}|

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> import magni
    >>> from magni.cs.indicators import calculate_mutual_coherence
    >>> Phi = np.zeros((5, 9))
    >>> Phi[0, 0] = Phi[1, 2] = Phi[2, 4] = Phi[3, 6] = Phi[4, 8] = 1
    >>> Psi = magni.imaging.dictionaries.get_DCT((3, 3))
    >>> for item in sorted(calculate_mutual_coherence(Phi, Psi).items()):
    ...     print('{}-norm: {:.3f}'.format(*item))
    0-norm: 0.889
    1-norm: 0.298
    2-norm: 0.333
    inf-norm: 0.667

    The above values can be calculated individually by specifying a norm:

    >>> for norm in (0, 1, 2, np.inf):
    ...     value = calculate_mutual_coherence(Phi, Psi, norm=norm)
    ...     print('{}-norm: {:.3f}'.format(norm, value))
    0-norm: 0.889
    1-norm: 0.298
    2-norm: 0.333
    inf-norm: 0.667

    """

    @_decorate_validation
    def validate_input():
        _numeric('Phi', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _numeric('Psi', ('integer', 'floating', 'complex'),
                 shape=(Phi.shape[1], -1))
        _numeric('norm', ('integer', 'floating'), range_='[0;inf]',
                 ignore_none=True)

    validate_input()

    M = np.zeros((Phi.shape[0], Psi.shape[1]))
    e = np.zeros((Psi.shape[1], 1))

    for i in range(M.shape[1]):
        e[i] = 1
        M[:, i] = np.abs(Phi.dot(Psi.dot(e))).reshape(-1)
        e[i] = 0

    if norm is None:
        value = {0: np.sum(M > 1e-9) / M.size,
                 1: np.sum(M) / M.size,
                 2: (np.sum(M**2) / M.size)**(1 / 2),
                 np.inf: np.max(M)}
    elif norm == 0:
        value = np.sum(M > 1e-9) / M.size
    elif norm == np.inf:
        value = np.max(M)
    else:
        value = (np.sum(M**norm) / M.size)**(1 / norm)

    return value


def calculate_relative_energy(Phi, Psi, method=None):
    r"""
    Calculate the relative energy of Phi Psi matrix product atoms.

    Parameters
    ----------
    Phi : magni.utils.matrices.Matrix or numpy.ndarray
        The measurement matrix.
    Psi : magni.utils.matrices.Matrix or numpy.ndarray
        The dictionary matrix.
    method : str
        The method used for summarising the relative energies of the Phi Psi
        matrix product atoms.

    Returns
    -------
    relative_energy : float or dict
        The relative_energy summary value(s).

    Notes
    -----
    The summary `method` used is either 'mean' for mean value, 'std' for
    standard deviation, 'min' for minimum value, or 'diff' for difference
    between the maximum and minimum values.

    If `method` is None, the function returns a dict containing all of the
    above summaries. Otherwise, the function returns the specified summary.

    The relative energies, which are summarised by the given `method`, are
    calculated as:

    .. math::

        \left[
        \frac{||\Phi \Psi_{:, 1}||_2}{||\Psi_{:, 1}||_2},
        \dots,
        \frac{||\Phi \Psi_{:, n}||_2}{||\Psi_{:, n}||_2}
        \right]^T

    where `n` is the number of columns in `Psi`.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> import magni
    >>> from magni.cs.indicators import calculate_relative_energy
    >>> Phi = np.zeros((5, 9))
    >>> Phi[0, 0] = Phi[1, 2] = Phi[2, 4] = Phi[3, 6] = Phi[4, 8] = 1
    >>> Psi = magni.imaging.dictionaries.get_DCT((3, 3))
    >>> for item in sorted(calculate_relative_energy(Phi, Psi).items()):
    ...     print('{}: {:.3f}'.format(*item))
    diff: 0.423
    mean: 0.735
    min: 0.577
    std: 0.126

    The above values can be calculated individually by specifying a norm:

    >>> for method in ('mean', 'std', 'min', 'diff'):
    ...     value = calculate_relative_energy(Phi, Psi, method=method)
    ...     print('{}: {:.3f}'.format(method, value))
    mean: 0.735
    std: 0.126
    min: 0.577
    diff: 0.423

    """

    @_decorate_validation
    def validate_input():
        _numeric('Phi', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _numeric('Psi', ('integer', 'floating', 'complex'),
                 shape=(Phi.shape[1], -1))
        _generic('method', 'string', value_in=('min', 'diff', 'mean', 'std'),
                 ignore_none=True)

    validate_input()

    e = np.zeros((Psi.shape[1], 1))
    x = np.zeros((Psi.shape[0], 1), dtype=Psi.dtype)
    w = np.zeros(Psi.shape[1])

    for i in range(Psi.shape[1]):
        e[i] = 1
        x[:] = Psi.dot(e)
        w[i] = np.linalg.norm(Phi.dot(x)) / np.linalg.norm(x)
        e[i] = 0

    if method is None:
        value = {'min': np.min(w),
                 'diff': np.max(w) - np.min(w),
                 'mean': np.mean(w),
                 'std': np.std(w)}
    elif method == 'min':
        value = np.min(w)
    elif method == 'diff':
        value = np.max(w) - np.min(w)
    elif method == 'mean':
        value = np.mean(w)
    elif method == 'std':
        value = np.std(w)

    return value
