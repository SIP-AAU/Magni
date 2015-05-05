"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing data container classes for .mi spectroscopy files.

The classes of this module can be used either directly or indirectly through
the `io` module by loading an .mi spectroscopy file.

Routine listings
----------------
Buffer(magni.afmtypes.BaseClass)
    Data class for .mi spectroscopy buffer.
Chunk(magni.afmtypes.BaseClass)
    Data class for .mi spectroscopy chunk.
Grid(magni.afmtypes.BaseClass)
    Data class for .mi spectroscopy grid.
Point(magni.afmtypes.BaseClass)
    Data class for .mi spectroscopy point.
Spectroscopy(magni.afm.types.File)
    Data class for .mi spectroscopy.

See Also
--------
magni.afm.io : .mi file loading.

"""

from __future__ import division
from collections import OrderedDict as _OrderedDict

import numpy as np

from magni.afm.types import BaseClass as _BaseClass
from magni.afm.types import File as _File
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


class Buffer(_BaseClass):
    """
    Data class of the .mi spectroscopy buffers.

    Parameters
    ----------
    attrs : dict
        The attributes of the buffer.
    data : list or tuple
        The grids, points, or chunks of the buffer.

    Attributes
    ----------
    data : numpy.ndarray
        The grids, points, or chunks of the buffer.

    See Also
    --------
    magni.utils.types.BaseClass : Superclass of the present class.

    Examples
    --------
    No example .mi spectroscopy file is distributed with magni.

    """

    _params = _OrderedDict((('bufferLabel', str),
                            ('bufferUnit', str)))

    def __init__(self, attrs, data):
        @_decorate_validation
        def validate_input():
            _generic('attrs', 'mapping', has_keys=('bufferLabel',))
            _levels('data', (_generic(None, 'explicit collection'),
                             _generic(None, (Grid, Point, Chunk))))

        _BaseClass.__init__(self, attrs)
        validate_input()

        self._data = tuple(data)

    data = property(lambda self: self._data)


class Chunk(_BaseClass):
    """
    Data class of the .mi spectroscopy chunks.

    Parameters
    ----------
    attrs : dict
        The attributes of the chunk.
    data : numpy.ndarray
        The data of the chunk.

    Attributes
    ----------
    data : numpy.ndarray
        The data of the chunk.

    See Also
    --------
    magni.utils.types.BaseClass : Superclass of the present class.

    Examples
    --------
    No example .mi spectroscopy file is distributed with magni.

    """

    _params = _OrderedDict((('index', int),
                            ('samples', int),
                            ('timeStart', float),
                            ('timeDelta', float),
                            ('sweepStart', float),
                            ('sweepDelta', float),
                            ('continuation', bool),
                            ('label', str),
                            ('pointIndex', int),
                            ('curveIndex', int)))

    def __init__(self, attrs, data):
        @_decorate_validation
        def validate_input():
            _generic('attrs', 'mapping',
                     has_keys=('samples', 'timeStart', 'timeDelta',
                               'sweepStart', 'sweepDelta'))
            _numeric('data', 'floating', shape=(self.attrs['samples'],))

        _BaseClass.__init__(self, attrs)
        validate_input()

        self._time = None
        self._sweep = None
        self._data = data
        self._data.flags['WRITEABLE'] = False

    data = property(lambda self: self._data)

    @property
    def sweep(self):
        """
        Get the sweep property of the chunk.

        The sweep property is the series of the entity which was swept.

        Returns
        -------
        sweep : numpy.ndarray
            The sweep property.

        Notes
        -----
        To reduce the memory footprint of chunks, the series does not exist
        explicitly, until it is requested.

        """

        if self._sweep is None:
            self._sweep = np.linspace(
                self.attrs['sweepStart'],
                self.attrs['sweepStart'] +
                (self.attrs['samples'] - 1) * self.attrs['sweepDelta'],
                self.attrs['samples'])
            self._sweep.flags['WRITEABLE'] = False

        return self._sweep

    @property
    def time(self):
        """
        Get the time property of the chunk.

        Returns
        -------
        time : numpy.ndarray
            The time property.

        Notes
        -----
        To reduce the memory footprint of chunks, the series does not exist
        explicitly, until it is requested.

        """

        if self._time is None:
            self._time = np.linspace(
                self.attrs['timeStart'],
                self.attrs['timeStart'] +
                (self.attrs['samples'] - 1) * self.attrs['timeDelta'],
                self.attrs['samples'])
            self._time.flags['WRITEABLE'] = False

        return self._time


class Grid(_BaseClass):
    """
    Data class of the .mi spectroscopy grids.

    Parameters
    ----------
    attrs : dict
        The attributes of the grid.
    points : list or tuple
        The points of the grid.

    Attributes
    ----------
    points : tuple
        The points of the grid.

    See Also
    --------
    magni.utils.types.BaseClass : Superclass of the present class.

    Notes
    -----
    The points are input and output as a 2D tuple of Point instances.

    Examples
    --------
    No example .mi spectroscopy file is distributed with magni.

    """

    _params = _OrderedDict((('index', int),
                            ('xCenter', float),
                            ('yCenter', float),
                            ('xPoints', int),
                            ('yPoints', int),
                            ('pointSpacing', float),
                            ('angle', int),
                            ('yDirection', str),
                            ('xDirection', str)))

    def __init__(self, attrs, points):
        @_decorate_validation
        def validate_input():
            _generic('attrs', 'mapping', has_keys=(
                'xCenter', 'yCenter', 'xPoints', 'yPoints', 'pointSpacing'))
            _levels('points', (_generic(None, 'explicit collection',
                                        len_=self.attrs['yPoints']),
                               _generic(None, 'explicit collection',
                                        len_=self.attrs['xPoints']),
                               _generic(None, Point, ignore_none=True)))

        _BaseClass.__init__(self, attrs)
        validate_input()

        self._points = tuple(tuple(val) for val in points)

    points = property(lambda self: self._points)


class Point(_BaseClass):
    """
    Data class of the .mi spectroscopy points.

    Parameters
    ----------
    attrs : dict
        The attributes of the point.
    chunks : list or tuple, optional
        The chunks of the point. (the default is (), which implies no chunks)

    Attributes
    ----------
    chunks : tuple
        The chunks of the point.

    See Also
    --------
    magni.utils.types.BaseClass : Superclass of the present class.

    Examples
    --------
    No example .mi spectroscopy file is distributed with magni.

    """

    _params = _OrderedDict((('index', int),
                            ('xCoordinate', float),
                            ('yCoordinate', float)))

    def __init__(self, attrs, chunks=()):
        @_decorate_validation
        def validate_input():
            _generic('attrs', 'mapping', has_keys=Point._params.keys())
            _levels('chunks', (_generic(None, 'explicit collection'),
                               _generic(None, Chunk)))

        _BaseClass.__init__(self, attrs)
        validate_input()

        self._chunks = tuple(chunks)

    chunks = property(lambda self: self._chunks)


class Spectroscopy(_File):
    """
    Data class of the .mi spectroscopy files.

    Parameters
    ----------
    attrs : dict
        The attributes of the image.
    buffers : list or tuple
        The buffers of the image.

    See Also
    --------
    magni.utils.types.File : Superclass of the present class.

    Examples
    --------
    No example .mi spectroscopy file is distributed with magni.

    """

    _params = _OrderedDict((('plotType', str),
                            ('BgImageFile', str)))

    def __init__(self, attrs, buffers):
        @_decorate_validation
        def validate_input():
            _levels('buffers', (_generic(None, 'explicit collection'),
                                _generic(None, Buffer)))

        _File.__init__(self, attrs, buffers)
        validate_input()
