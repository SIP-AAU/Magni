"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the public functionality of the magni.afm.io subpackage.

"""

from collections import defaultdict as _defaultdict
import os

from magni.afm.io._data_attachment import attach_data as _attach_data
from magni.afm.io._object_building import build_object as _build_object
from magni.afm.io._stream_conversion import convert_stream as _convert_stream
from magni.afm.types.image import Buffer as _ImageBuffer
from magni.afm.types.image import Image as _Image
from magni.afm.types.spectroscopy import Buffer as _SpectroscopyBuffer
from magni.afm.types.spectroscopy import Chunk as _SpectroscopyChunk
from magni.afm.types.spectroscopy import Grid as _SpectroscopyGrid
from magni.afm.types.spectroscopy import Point as _SpectroscopyPoint
from magni.afm.types.spectroscopy import Spectroscopy as _Spectroscopy
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic


def read_mi_file(path):
    """
    Read an .mi file given by a path.

    The supported .mi file types are Image and Spectroscopy.

    Parameters
    ----------
    path : str
        The path of the .mi file to read.

    Returns
    -------
    file_ : magni.afm.types.File
        The read .mi file.

    See Also
    --------
    magni.afm.io._stream_conversion : First step of the reading.
    magni.afm.io._object_building : Second step of the reading.
    magni.afm.io._data_attachment : Third step of the reading.

    Notes
    -----
    Depending on the .mi file type, the returned value will be an instance of a
    subclass of `magni.afm.types.File`: `magni.afm.types.image.Image` or
    `magni.afm.types.spectroscopy.Spectroscopy`.

    The reading of an .mi file is logically separated into four steps:

    1. Converting the file stream to flat, basic variables.
    2. Converting the header parameters to a hierarchical object-structure
       mimicking that of `magni.afm.types`.
    3. Attaching the actual data to the hierarchical object-structure.
    4. Instantiating the classes in `magni.afm.types` from the hierarchical
       object structure.

    The functionality needed for each step is grouped in separate modules with
    the functionality needed for the fourth step being grouped in the this
    module.

    Examples
    --------
    An example of how to use this function to read the example .mi file
    provided with the package:

    >>> import os, magni
    >>> from magni.afm.io import read_mi_file
    >>> path = magni.utils.split_path(magni.__path__[0])[0]
    >>> path = path + 'examples' + os.sep + 'example.mi'
    >>> if os.path.isfile(path):
    ...     mi_file = read_mi_file(path)

    """

    @_decorate_validation
    def validate_input():
        _generic('path', 'string')

    validate_input()

    with open(path, 'rb') as f:
        stream = f.read()

    params, data = _convert_stream(stream)
    obj = _build_object(params)
    _attach_data(obj, data)

    if obj['attrs']['fileType'].lower() == 'image':
        for i, buffer_ in enumerate(obj['buffers']):
            obj['buffers'][i] = _ImageBuffer(buffer_['attrs'], buffer_['data'])

        obj = _Image(obj['attrs'], obj['buffers'])
    else:  # 'spectroscopy'
        obj = _instantiate_spectroscopy(obj)

    return obj


def _instantiate_spectroscopy(obj):
    """
    Instantiate a Spectroscopy object from a hierarchical object-structure.

    Parameters
    ----------
    obj : object
        The hierarchical object-structure.

    Returns
    -------
    spectroscopy : magni.afm.types.spectroscopy.Spectroscopy
        The instantiated Spectroscopy object.

    See Also
    --------
    read_mi_file : Function using the present function.

    Notes
    -----
    This function instantiates the sweep buffer and the spectroscopy file.

    """

    items = []

    for item in obj['buffers'][0]['data']:
        if item['type'] == 'grid':
            points = [[_SpectroscopyPoint(point['attrs'], []) for point in row]
                      for row in item['points']]
            items.append(_SpectroscopyGrid(item['attrs'], points))
        else:  # 'point'
            items.append(_SpectroscopyPoint(item['attrs'], []))

    sweep_buffer = _SpectroscopyBuffer(obj['buffers'][0]['attrs'], items)
    data_buffers = [_instantiate_spectroscopy_buffer(buffer_, sweep_buffer)
                    for buffer_ in obj['buffers'][1:]]

    return _Spectroscopy(obj['attrs'], [sweep_buffer] + data_buffers)


def _instantiate_spectroscopy_buffer(data_buffer, sweep_buffer):
    """
    Instantiate a spectroscopy Buffer object from a hierarchical
    object-structure.

    Parameters
    ----------
    obj : object
        The hierarchical object-structure.

    Returns
    -------
    buffer_ : magni.afm.types.spectroscopy.Buffer
        The instantiated spectroscopy Buffer.

    See Also
    --------
    _instantiate_spectroscopy : Function using the present function.

    """

    chunks = _defaultdict(lambda: [])

    for chunk in data_buffer['data']:
        key = chunk['attrs'].get('pointIndex')
        chunks[key].append(_SpectroscopyChunk(chunk['attrs'], chunk['data']))

    if len(sweep_buffer.data) == 0:
        chunks[None].extend(chunks[0])

    items = []

    for item in sweep_buffer.data:
        if isinstance(item, _SpectroscopyGrid):
            points = [[_SpectroscopyPoint(point.attrs,
                                          chunks[point.attrs['index']])
                       for point in row] for row in item.points]
            items.append(_SpectroscopyGrid(item.attrs, points))
        elif isinstance(item, _SpectroscopyPoint):
            items.append(_SpectroscopyPoint(item.attrs,
                                            chunks[item.attrs['index']]))

    return _SpectroscopyBuffer(data_buffer['attrs'], items + chunks[None])
