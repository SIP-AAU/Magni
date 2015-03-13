"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing a multi domain image class.

Routine listings
----------------
MultiDomainImage(object)
    Provide access to an image in the domains of a compressed sensing context.

"""

from __future__ import division

from magni.utils.matrices import Matrix as _Matrix
from magni.utils.matrices import MatrixCollection as _MatrixC
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_numeric as _numeric


class MultiDomainImage(object):
    """
    Provide access to an image in the domains of a compressed sensing context.

    Given a measurement matrix and a dictionary, an image can be supplied in
    either the measurement domain, the image domain, or the coefficient domain.
    This class then provides access to the image in all three domains.

    Parameters
    ----------
    Phi : magni.utils.matrices.Matrix, magni.utils.matrices.MatrixCollection,
        or numpy.ndarray
        The measurement matrix.
    Psi : magni.utils.matrices.Matrix, magni.utils.matrices.MatrixCollection,
        or numpy.ndarray
        The dictionary.

    Notes
    -----
    The image is only converted to other domains than the supplied when the
    the image is requested in another domain. The image is, however, stored in
    up to three versions internally in order to reduce computation overhead.
    This may introduce a memory overhead.

    Examples
    --------
    Define a measurement matrix which skips every other sample:

    >>> import numpy as np, magni
    >>> func = lambda vec: vec[::2]
    >>> func_T = lambda vec: np.float64([vec[0], 0, vec[1]]).reshape(3, 1)
    >>> Phi = magni.utils.matrices.Matrix(func, func_T, (), (2, 3))

    Define a dictionary which is simply a rotated identity matrix:

    >>> v = np.sqrt(0.5)
    >>> Psi = np.float64([[ v, -v,  0],
    ...                   [ v,  v,  0],
    ...                   [ 0,  0,  1]])

    Instantiate the current class to handle domains:

    >>> from magni.imaging.domains import MultiDomainImage
    >>> domains = MultiDomainImage(Phi, Psi)

    An image can the be supplied in any domain and likewise retrieved in any
    domain. For example, the image:

    >>> domains.image = np.ones(3).reshape(3, 1)

    Can be retrieved both as measurements:

    >>> np.set_printoptions(suppress=True)
    >>> domains.measurements
    array([[ 1.],
           [ 1.]])

    And as coefficients:

    >>> domains.coefficients
    array([[ 1.41421356],
           [ 0.        ],
           [ 1.        ]])

    """

    def __init__(self, Phi, Psi):
        @_decorate_validation
        def validate_input():
            _numeric('Phi', ('integer', 'floating', 'complex'), shape=(-1, -1))
            _numeric('Psi', ('integer', 'floating', 'complex'),
                     shape=(Phi.shape[1], -1))

        validate_input()

        self._Phi = Phi
        self._Psi = Psi
        self._measurements = None
        self._image = None
        self._coefficients = None

    @property
    def coefficients(self):
        """
        Get the image in the coefficient domain.

        Returns
        -------
        coefficients : numpy.ndarray
            The dictionary coefficients of the image.

        """

        if (self._measurements is not None or self._image is not None or
                self._coefficients is not None):
            if self._coefficients is None:
                if self._image is None:
                    self._image = self._Phi.T.dot(self._measurements)

                self._coefficients = self._Psi.T.dot(self._image)

            return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        """
        Set the image in the coefficient domain.

        Parameters
        ----------
        coefficients : numpy.ndarray
            The dictionary coefficients of the image.

        """

        @_decorate_validation
        def validate_input():
            _numeric('value', ('integer', 'floating', 'complex'),
                     shape=(self._Psi.shape[1], 1))

        validate_input()

        self._measurements = None
        self._image = None
        self._coefficients = value

    @property
    def image(self):
        """
        Get the image in the image domain.

        Returns
        -------
        image : numpy.ndarray
            The image.

        """

        if (self._measurements is not None or self._image is not None or
                self._coefficients is not None):
            if self._image is None:
                if self._measurements is not None:
                    self._image = self._Phi.T.dot(self._measurements)
                elif self._coefficients is not None:
                    self._image = self._Psi.dot(self._coefficients)

            return self._image

    @image.setter
    def image(self, value):
        """
        Set the image in the image domain.

        Parameters
        ----------
        image : numpy.ndarray
            The image.

        """

        @_decorate_validation
        def validate_input():
            _numeric('value', ('integer', 'floating', 'complex'),
                     shape=(self._Phi.shape[1], 1))

        validate_input()

        self._measurements = None
        self._image = value
        self._coefficients = None

    @property
    def measurements(self):
        """
        Get the image in the measurement domain.

        Returns
        -------
        measurements : numpy.ndarray
            The measurements of the image.

        """

        if (self._measurements is not None or self._image is not None or
                self._coefficients is not None):
            if self._measurements is None:
                if self._image is None:
                    self._image = self._Psi.dot(self._coefficients)

                self._measurements = self._Phi.dot(self._image)

            return self._measurements

    @measurements.setter
    def measurements(self, value):
        """
        Set the image in the measurement domain.

        Parameters
        ----------
        measurements : numpy.ndarray
            The measurements of the image.

        """

        @_decorate_validation
        def validate_input():
            _numeric('value', ('integer', 'floating', 'complex'),
                     shape=(self._Phi.shape[0], 1))

        validate_input()

        self._measurements = value
        self._image = None
        self._coefficients = None
