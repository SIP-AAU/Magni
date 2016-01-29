"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality for attaching data.

The reading of an .mi file is logically separated into four steps of which the
functionality provided by this module performs the third step.

Routine listings
----------------
attach_data(obj, data)
    Attach data to a hierarchical object-structure.

"""

from __future__ import division

import numpy as np


def attach_data(obj, data):
    """
    Attach data to a hierachical object-structure.

    In the case of .mi image files, the data should be attached to the buffers
    of the file. In the case of .mi spectroscopy files, the data should be
    attached to the chunks of the file.

    Parameters
    ----------
    obj : object
        The hierarchical object-structure to attach data to.
    data : numpy.ndarray
        The 1-dimensional data.

    See Also
    --------
    magni.afm.io.read_mi_file : Function using the present function.

    """

    if obj['attrs']['fileType'].lower() == 'image':
        _attach_image_data(obj, data)
    else:  # 'spectroscopy'
        _attach_spectroscopy_data(obj, data)


def _attach_image_data(obj, data):
    """
    Attach data to a hierarchical object-structure representing an image.

    The data should be attached to the buffers of the file.

    Parameters
    ----------
    obj : object
        The hierarchical object-structure to attach data to.
    data : numpy.ndarray
        The 1-dimensional data.

    See Also
    --------
    attach_data : Function using the present function.

    """

    obj, data, thumbnail = _handle_format_inconsistency(obj, data)
    width, height = obj['attrs']['xPixels'], obj['attrs']['yPixels']
    size = width * height
    index = 0

    for buffer_ in obj['buffers']:
        buffer_['data'] = data[index:index + size]
        buffer_['data'] = buffer_['data'].reshape(height, width)
        buffer_['data'] = buffer_['data'][::-1, :]

        if obj['attrs']['data'].lower() in ('binary', 'binary_32', 'ascii'):
            if 'bufferRange' not in buffer_['attrs']:
                raise IOError("Each buffer must have a 'bufferRange' header "
                              "parameter.")

            if not isinstance(buffer_['attrs']['bufferRange'], float):
                msg = "The 'bufferRange' buffer parameter must have type {!r}."
                raise IOError(msg.format(float))

            buffer_['data'] = buffer_['attrs']['bufferRange'] * buffer_['data']

        index = index + size

    obj['buffers'] = obj['buffers'] + thumbnail

    if data.shape[0] != index:
        raise IOError("The file contains different amount of data than "
                      "specified by the file headers.")


def _attach_spectroscopy_data(obj, data):
    """
    Attach data to a hierarchical object-structure representing a spectroscopy.

    The data should be attached to the chunks of the file.

    Parameters
    ----------
    obj : object
        The hierarchical object-structure to attach data to.
    data : numpy.ndarray
        The 1-dimensional data.

    See Also
    --------
    attach_data : Function using the present function.

    """

    index = 0

    for buffer_ in obj['buffers'][1:]:
        for chunk in buffer_['data']:
            chunk['data'] = data[index:index + chunk['attrs']['samples']]
            index = index + chunk['attrs']['samples']

    if data.shape[0] != index:
        raise IOError("The file contains different amount of data than "
                      "specified by the file headers.")


def _handle_format_inconsistency(obj, data):
    """
    Handle format inconsistency.

    The inconsistency is the optional precense of an undocumented thumbnail.
    If the thumbnail is present, the file header contains a thumbnail parameter
    specifying the resolution of the thumbnail image.

    Parameters
    ----------
    obj : object
        The hierarchical object-structure to attach data to.
    data : numpy.ndarray
        The 1-dimensional data.

    Returns
    -------
    obj : object
        The hierarchical object-structure to attach data to.
    data : numpy.ndarray
        The 1-dimensional data excluding any thumbnail data.
    thumbnail : list
        A list of a thumbnail object-structure if any.

    See Also
    --------
    _attach_image_data : Function using the present function.

    Notes
    -----
    The thumbnail is stored as 8-bit unsigned integers with a red, a green, and
    a blue channel. If the thumbnail is present, it is represented by three
    buffer object-structures; one for each channel.

    """

    if 'thumbnail' in obj['attrs'].keys():
        if not isinstance(obj['attrs']['thumbnail'], str):
            msg = "The 'thumbnail' file header must have type {!r}."
            raise IOError(msg.format(str))

        shape = obj['attrs']['thumbnail'].split('x')

        try:
            shape = (int(shape[0]), int(shape[1]))
        except (IndexError, ValueError):
            raise IOError("The 'thumbnail' file header must consist of two "
                          "integers separated by an 'x'.")

        size = shape[0] * shape[1]

        if obj['attrs']['data'].lower() == 'binary':
            data, thumbnail = data[:-3 * size / 2], data[-3 * size / 2:]
            thumbnail = np.int16(thumbnail * 2**15).view(np.uint8)
        elif obj['attrs']['data'].lower() == 'binary_32':
            data, thumbnail = data[:-3 * size / 4], data[-3 * size / 4:]
            thumbnail = np.int32(thumbnail * 2**31).view(np.uint8)
        else:
            msg = ("The 'thumbnail' file header causes unknown behavior for "
                   "{!r} data.")
            raise IOError(msg.format(obj['attrs']['data']))

        thumbnail = [thumbnail[i * size:(i + 1) * size] for i in range(3)]
        labels = ['ThumbnailRed', 'ThumbnailGreen', 'ThumbnailBlue']
        thumbnail = [{'attrs': {'bufferLabel': labels[i]},
                      'data': np.int16(thumbnail[i].reshape(shape))}
                     for i in range(3)]
    else:
        thumbnail = []

    return obj, data, thumbnail
