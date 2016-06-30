"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing data container classes for .mi image files.

The classes of this module can be used either directly or indirectly through
the `magni.afm.io` subpackage by loading an .mi image file.

Routine listings
----------------
Buffer(magni.afm.types.BaseClass)
    Data class of the .mi image file buffers.
Image(magni.afm.types.File)
    Data class of the .mi image files.

See Also
--------
magni.afm.io : .mi file loading.

"""

from __future__ import division
from collections import OrderedDict as _OrderedDict
import types

from magni.afm.types import BaseClass as _BaseClass
from magni.afm.types import File as _File
from magni.imaging.preprocessing import detilt as _detilt
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric


class Buffer(_BaseClass):
    """
    Data class of the .mi image file buffers.

    Parameters
    ----------
    attrs : dict
        The attributes of the buffer.
    data : numpy.ndarray
        The 2D data of the buffer.

    Attributes
    ----------
    apply_clipping : bool
        A flag indicating if clipping should be applied to the data.
    data : numpy.ndarray
        The 2D data of the buffer.

    See Also
    --------
    magni.utils.types.BaseClass : Superclass of the present class.

    Examples
    --------
    A subclass of the present class is implicitly instantiated when loading,
    for example, the .mi file provided with the package:

    >>> import os, magni
    >>> path = magni.utils.split_path(magni.__path__[0])[0]
    >>> path = path + 'examples' + os.sep + 'example.mi'
    >>> if os.path.isfile(path):
    ...     image = magni.afm.io.read_mi_file(path)
    ...     buffer_ = image.buffers[0]

    This buffer can have a number of attributes including 'bufferLabel':

    >>> if os.path.isfile(path):
    ...     print('{!r}'.format(buffer_.attrs['bufferLabel']))
    ... else:
    ...     print("'Topography'")
    'Topography'

    The primary purpose of this class is, however, to contain the 2D data of a
    buffer:

    >>> if os.path.isfile(path):
    ...     print('Buffer data shape: {!r}'.format(
    ...     tuple(int(s) for s in buffer_.data.shape)))
    ... else:
    ...     print('Buffer data shape: (256, 256)')
    Buffer data shape: (256, 256)

    """

    _params = _OrderedDict((('bufferLabel', str),
                            ('trace', bool),
                            ('bufferUnit', str),
                            ('bufferRange', float),
                            ('DisplayOffset', float),
                            ('DisplayRange', float),
                            ('filter', str)))

    def __init__(self, attrs, data):
        @_decorate_validation
        def validate_input():
            _generic('attrs', 'mapping', has_keys=('bufferLabel',))
            _numeric('data', ('integer', 'floating'), shape=(-1, -1))

        _BaseClass.__init__(self, attrs)
        validate_input()

        self._apply_clipping = False
        self._data = data
        self._data.flags['WRITEABLE'] = False

    @property
    def apply_clipping(self):
        """
        Get the apply_clipping property of the buffer.

        The clipping is specified by the 'DisplayOffset', 'DisplayRange', and
        'filter' attributes of the buffer.

        Returns
        -------
        apply_clipping : bool
            A flag indicating if clipping should be applied to the data.

        """

        return self._apply_clipping

    @apply_clipping.setter
    def apply_clipping(self, value):
        """
        Set the apply_clipping property of the buffer.

        The clipping is specified by the 'DisplayOffset', 'DisplayRange', and
        'filter' attributes of the buffer.

        Parameters
        ----------
        value : bool
            The desired value of the property.

        """

        @_decorate_validation
        def validate_input():
            _numeric('value', 'boolean')

        validate_input()

        self._apply_clipping = value

    @property
    def data(self):
        """
        Get the data property of the buffer.

        If `apply_clipping` is False, the data is returned as-is. Otherwise,
        the filtering specified by the 'filter' attribute of the buffer and the
        clipping specified by the 'DisplayOffset' and 'DisplayRange' attributes
        of the buffer are applied to the data before it is returned.

        Returns
        -------
        data : numpy.ndarray
            The 2D data of the buffer.

        """

        data = self._data
        attrs = self.attrs

        if self.apply_clipping:
            try:
                filter_ = attrs['filter']
                offset = attrs['DisplayOffset']
                range_ = attrs['DisplayRange']
            except KeyError:
                msg = ("The value of >>self.attrs<<, {!r}, must have the keys "
                       "('filter', 'DisplayOffset', 'DisplayRange'). when "
                       "self.apply_filter is True.")
                raise KeyError(msg.format(attrs))

            if filter_ == 'None':
                data = data.copy()
            elif filter_ == '1st_order':
                data = _detilt(data, mode='line_flatten', degree=1)
            elif filter_ == '2nd_order':
                data = _detilt(data, mode='line_flatten', degree=2)
            elif filter_ == '3rd_order':
                data = _detilt(data, mode='line_flatten', degree=3)
            else:
                msg = ("The value of >>self.attrs['filter']<<, {!r}, must be "
                       "in ('None', '1st_order', '2nd_order', '3rd_order').")
                raise ValueError(msg.format(attrs['filter']))

            data[data < offset - 0.5 * range_] = offset - 0.5 * range_
            data[data > offset + 0.5 * range_] = offset + 0.5 * range_

        return data


class Image(_File):
    """
    Data class of the .mi image files.

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
    The present class is implicitly instantiated when loading, for example, the
    .mi file provided with the package:

    >>> import os, magni
    >>> path = magni.utils.split_path(magni.__path__[0])[0]
    >>> path = path + 'examples' + os.sep + 'example.mi'
    >>> if os.path.isfile(path):
    ...     image = magni.afm.io.read_mi_file(path)

    This file has a number of buffers which each has the 'bufferLabel'
    attribute:

    >>> if os.path.isfile(path):
    ...     for buffer_ in image.buffers[::2][:3]:
    ...         label = buffer_.attrs['bufferLabel']
    ...         print("Buffer with 'bufferLabel': {!r}".format(label))
    ... else:
    ...     for label in ('Topography', 'Deflection', 'Friction'):
    ...         print("Buffer with 'bufferLabel': {!r}".format(label))
    Buffer with 'bufferLabel': 'Topography'
    Buffer with 'bufferLabel': 'Deflection'
    Buffer with 'bufferLabel': 'Friction'

    """

    _params = _OrderedDict((('mode', str),
                            ('xPixels', int),
                            ('yPixels', int),
                            ('scanUp', bool),
                            ('scanSpeed', float),
                            ('acMac', bool),
                            ('acACMode', bool)))

    def __init__(self, attrs, buffers):
        @_decorate_validation
        def validate_input():
            _generic('attrs', 'mapping', has_keys=('xPixels', 'yPixels'))
            _levels('buffers', (_generic(None, 'explicit collection'),
                                _generic(None, Buffer)))

            attrs = self.attrs

            for i, buffer_ in enumerate(buffers):
                if buffer_.attrs['bufferLabel'][:9] != 'Thumbnail':
                    _numeric('buffers[{}].data'.format(i),
                             ('integer', 'floating'),
                             shape=(attrs['yPixels'], attrs['xPixels']),
                             var=buffer_.data)

        _File.__init__(self, attrs, buffers)
        validate_input()
