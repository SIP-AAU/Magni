"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing custom data types.

Routine listings
----------------
ClassProperty(property)
    Class property.
ReadOnlyDict(collections.OrderedDict)
    Read-only ordered dict.

"""

from __future__ import division
from collections import OrderedDict as _OrderedDict

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate as _validate


class ClassProperty(property):
    """
    Class property.

    The present class is a combination of the built-in property type and the
    built-in classmethod type. That is, it is a property which is invoked on a
    class rather than an instance but is available to both the class and its
    instances.

    Parameters
    ----------
    fget : function, optional
        The getter function of the property. (the default is None, which
        implies that the property cannot be read)
    fset : function, optional
        The setter function of the property. (the default is None, which
        implies that the property cannot be written)
    fdel : function, optional
        The deleter function of the property. (the default is None, which
        implies that the property cannot be deleted)
    doc : string, optional
        The docstring of the property. (the default is None, which implies that
        the docstring of the getter function, if any, is used)

    See Also
    --------
    property : Superclass from which all behaviour is inherited or extended.

    Examples
    --------
    The following example illustrates the difference between regular properties
    and class properties. First, the example class is defined and instantiated:

    >>> from magni.utils.types import ClassProperty
    >>> class Example(object):
    ...     _x_class = 0
    ...     def __init__(self, x):
    ...         self._x_instance = x
    ...     @ClassProperty
    ...     def x_class(class_):
    ...         return class_._x_class
    ...     @property
    ...     def x_instance(self):
    ...         return self._x_instance
    >>> example = Example(1)

    The regular read-only property works on the instance:

    >>> print('{!r}'.format(example.x_instance))
    1
    >>> print('Is property? {!r}'.format(isinstance(Example.x_instance,
    ...                                             property)))
    Is property? True

    The class property, on the other hand, works on the class:

    >>> print('{!r}'.format(example.x_class))
    0
    >>> print('Is property? {!r}'.format(isinstance(Example.x_class,
    ...                                             property)))
    Is property? False
    >>> print('{!r}'.format(Example.x_class))
    0

    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        property.__init__(self, fget, fset, fdel, doc)

    def __get__(self, obj, class_=None):
        """
        The get method of the property.

        Parameters
        ----------
        obj : object
            The instance on which the property is requested.
        class_ : type, optional
            The class on which the property is requested. (the default is None,
            which implies that the class is retrieved from `obj`)

        Returns
        -------
        value : None
            The value of the property.

        See Also
        --------
        property.__get__ : The method being extended.

        """

        if class_ is None:
            class_ = type(obj)

        return property.__get__(self, class_)

    def __set__(self, obj, value):
        """
        The set method of the property.

        Parameters
        ----------
        obj : object
            The instance on which the property is requested.
        value : None
            The value which should be assigned to the property.

        See Also
        --------
        property.__set__ : The method being extended.

        """

        return property.__set__(self, type(obj), value)

    def __delete__(self, obj):
        """
        The delete method of the property.

        Parameters
        ----------
        obj : object
            The instance on which the property is requested.

        See Also
        --------
        property.__delete__ : The method being extended.

        """

        return property.__delete__(self, type(obj))


class ReadOnlyDict(_OrderedDict):
    """
    Read-only ordered dict.

    The present ordered dict subclass has its non-read-only methods disabled.

    See Also
    --------
    collections.OrderedDict : Superclass from which all read-only behaviour is
        inherited.

    Examples
    --------
    This ordered dict subclass exposes the same interface as the OrderedDict
    class when not using methods that alter the dict.

    >>> from magni.utils.types import ReadOnlyDict
    >>> d = ReadOnlyDict(key='value')
    >>> for item in d.items():
    ...     print('{!r}'.format(item))
    ('key', 'value')

    However, any attempt to assign another value to the property raises an
    exception:

    >>> try:
    ...     d['key'] = 'new value'
    ... except Exception:
    ...     print('An exception occured.')
    An exception occured.

    """

    def __init__(self, *args, **kwargs):
        self._disabled = True
        _OrderedDict.__init__(self, *args, **kwargs)
        self._disabled = False

    def __delitem__(self, name):
        """
        Prevent deletion of items.

        Parameters
        ----------
        name : str
            The name of the item to be deleted.

        """

        if self._disabled:
            _OrderedDict.__delitem__(self, name)
        else:
            raise AttributeError('{!r} objects are read-only.'
                                 .format(self.__class__))

    def __getattribute__(self, name):
        """
        Return the requested attribute unless it is a non-read-only method.

        The purpose of this overwrite is to disable access to the following
        non-read-only dict methods: `clear`, `pop`, `popitem`, `setdefault`,
        and `update`. The first two methods are disabled otherwise.

        Parameters
        ----------
        name : str
            The name of the requested attribute.

        Returns
        -------
        attribute : None
            The requested attribute.

        Notes
        -----
        `__getattribute__` is implicitly called, when the attribute of an
        object is accessed as `object.attribute`.

        """

        if name in ('clear', 'pop', 'popitem', 'setdefault', 'update'):
            if self._disabled:
                value = _OrderedDict.__getattribute__(self, name)
            else:
                raise TypeError('{!r} objects are read-only.'
                                .format(self.__class__))
        else:
            value = _OrderedDict.__getattribute__(self, name)

        return value

    def __setitem__(self, name, value):
        """
        Prevent overwrite of items.

        Parameters
        ----------
        name : str
            The name of the item to be overwritten.
        value : None
            The value to be written.

        """

        if self._disabled:
            _OrderedDict.__setitem__(self, name, value)
        else:
            raise AttributeError('{!r} objects are read-only.'
                                 .format(self.__class__))
