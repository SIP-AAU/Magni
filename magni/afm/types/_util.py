"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the common functionality of the subpackage.

"""

from __future__ import division
from collections import OrderedDict as _OrderedDict

from magni.utils.types import ClassProperty as _classproperty
from magni.utils.types import ReadOnlyDict as _ReadOnlyDict
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels


class BaseClass(object):
    """
    Base class of every `magni.afm.types` data class.

    The present class validates the attributes passed to its constructor
    against the allowed attributes of the class of the given instance and
    exposes these attributes through a read-only dictionary property, `attrs`.
    Furthermore, the present class exposes the allowed attributes of the class
    of the given instance through a read-only dictionary static property,
    `params`.

    Parameters
    ----------
    attrs : dict
        The desired attributes of the instance.

    Attributes
    ----------
    attrs : magni.utils.types.ReadOnlyDict
        The attributes of the instance.
    params : magni.utils.types.ReadOnlyDict
        The allowed attributes of the instance.

    Examples
    --------
    An example could be the subclass, 'Person' which allows only the string
    attribute, name:

    >>> from magni.afm.types._util import BaseClass
    >>> class Person(BaseClass):
    ...     def __init__(self, attrs):
    ...         BaseClass.__init__(self, attrs)
    ...     _params = {'name': str}

    This class can then be initiated with a name:

    >>> person = Person({'name': 'Murphy'})
    >>> print('{!r}'.format(person.attrs['name']))
    'Murphy'

    Any other attributes, than 'name', passed to the class are ignored:

    >>> person = Person({'name': 'Murphy', 'age': 42})
    >>> for name in person.attrs.keys():
    ...     print('{!r}'.format(name))
    'name'

    """

    def __init__(self, attrs):
        @_decorate_validation
        def validate_input():
            _generic('attrs', 'mapping')
            params = self.params

            for name in attrs.keys():
                if name in params.keys():
                    _generic(('attrs', name), params[name])

        validate_input()

        names = self.params.keys()
        attrs = {name: value for name, value in attrs.items() if name in names}
        self._attrs = _ReadOnlyDict(attrs)

    attrs = property(lambda self: self._attrs)

    @_classproperty
    def params(class_):
        """
        Get the allowed attributes of the class (of the instance).

        Which attributes are allowed is specified by the class and superclasses
        (of the instance).

        Returns
        -------
        params : magni.utils.types.ReadOnlyDict
            The allowed attributes.

        Notes
        -----
        The allowed attributes of the class (of the given instance) are found
        by inspecting the class and its base classes for the static variable
        '_params' and collecting these in a read-only dictionary.

        """

        params = []

        while class_ != BaseClass:
            if hasattr(class_, '_params'):
                params.extend(class_._params.items())

            class_ = class_.__base__

        return _ReadOnlyDict(params)


class File(BaseClass):
    """
    Base class of the `magni.afm.types` file classes.

    The present class specifies the allowed file-level attributes and exposes
    the read-only property, buffers which all .mi files have. Furthermore, the
    present class provides a method for accessing buffers by their
    'bufferLabel' attribute.

    Parameters
    ----------
    attrs : list or tuple
        The desired attributes of the file.
    buffers : list or tuple
        The buffers of the file.

    Attributes
    ----------
    buffers : tuple
        The buffers of the file.

    See Also
    --------
    BaseClass : Superclass of the present class.

    Examples
    --------
    A subclass of the present class is implicitly instantiated when loading,
    for example, the .mi file provided with the package:

    >>> import os, magni
    >>> path = magni.utils.split_path(magni.__path__[0])[0]
    >>> path = path + 'examples' + os.sep + 'example.mi'
    >>> file_ = magni.afm.io.read_mi_file(path)

    This file has a number of buffers which each has the 'bufferLabel'
    attribute:

    >>> for buffer_ in file_.buffers[::2][:3]:
    ...     label = buffer_.attrs['bufferLabel']
    ...     print("Buffer with 'bufferLabel': {!r}".format(label))
    Buffer with 'bufferLabel': 'Topography'
    Buffer with 'bufferLabel': 'Deflection'
    Buffer with 'bufferLabel': 'Friction'

    If only, for example, buffers with 'bufferLabel' equal to 'Topography' are
    desired, the method, `get_buffer` can be called:

    >>> buffers = len(file_.get_buffer('Topography'))
    >>> print("Buffers with 'bufferLabel' == 'Topography': {}".format(buffers))
    Buffers with 'bufferLabel' == 'Topography': 2

    """

    _params = _OrderedDict((('fileType', str),
                            ('dateAcquired', str),
                            ('xOffset', float),
                            ('yOffset', float),
                            ('xLength', float),
                            ('yLength', float),
                            ('data', str)))

    def __init__(self, attrs, buffers):
        @_decorate_validation
        def validate_input():
            _generic('attrs', 'mapping', has_keys=('fileType',))
            _levels('buffers', (_generic(None, 'explicit collection'),
                                _generic(None, BaseClass)))

        BaseClass.__init__(self, attrs)
        validate_input()

        self._buffers = tuple(buffers)

    buffers = property(lambda self: self._buffers)

    def get_buffer(self, label):
        """
        Get the buffers that have the specified buffer label.

        Parameters
        ----------
        label : str
            The desired buffer label of the buffer.

        Returns
        -------
        buffers : list
            The buffers that have the desired buffer label.

        """

        @_decorate_validation
        def validate_input():
            _generic('label', 'string')

        validate_input()

        buffers = []

        for buffer_ in self.buffers:
            if ('bufferLabel' in buffer_.attrs.keys() and
                    buffer_.attrs['bufferLabel'] == label):
                buffers.append(buffer_)

        return buffers


class FileCollection(BaseClass):
    """
    Data class for collections of File instances with identical settings.

    The settings are the following attributes: 'fileType', 'mode', 'xPixels',
    'yPixels', 'xOffset', 'yOffset', 'xLength', 'yLength', 'scanSpeed',
    'acMac', 'acACMode', 'plotType'.

    The present class exposes the files of the collection through a read-only
    tuple property, `files` and the paths of these files through a read-only
    tuple property, `paths`.

    Parameters
    ----------
    files : list or tuple
        The files of the collection.
    paths : list or tuple
        The paths of the files of the collection.

    Attributes
    ----------
    files : tuple
        The files of the collection.
    paths : tuple
        The paths of the files of the collection.

    See Also
    --------
    BaseClass : Superclass of the present class.

    Examples
    --------
    No example .mi file collection is distributed with `magni`.

    """

    _params = _OrderedDict((('fileType', str),
                            ('mode', str),
                            ('xPixels', int),
                            ('yPixels', int),
                            ('xOffset', float),
                            ('yOffset', float),
                            ('xLength', float),
                            ('yLength', float),
                            ('scanSpeed', float),
                            ('acMac', bool),
                            ('acACMode', bool),
                            ('plotType', str)))

    def __init__(self, files, paths):
        @_decorate_validation
        def validate_input():
            _levels('files', (_generic(None, 'explicit collection'),
                              _generic(None, File)))

            if len(files) < 1:
                msg = 'The value of >>len(files)<<, {!r}, must be > 0.'
                raise ValueError(msg.format(len(files)))

            attrs = [{name: value for name, value in file_.attrs.items()
                      if name in self._params} for file_ in files]

            for i in range(1, len(files)):
                if attrs[i] != attrs[0]:
                    raise ValueError('The values of >>files<< must have the '
                                     'same settings.')

            _levels('paths', (_generic(None, 'explicit collection',
                                       len_=len(files)),
                              _generic(None, 'string')))

        # validate before calling the superclass constructor to ensure that
        # 'files[0].attrs' exists
        validate_input()
        BaseClass.__init__(self, files[0].attrs)

        self._files = tuple(files)
        self._paths = tuple(paths)

    files = property(lambda self: self._files)
    paths = property(lambda self: self._paths)
