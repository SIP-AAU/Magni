"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality for building a hierarchical object-structure.

The reading of an .mi file is logically separated into four steps of which the
functionality provided by this module performs the second step.

Routine listings
----------------
build_object(params)
    Build a hierarchical object-structure from header parameters.

"""

from __future__ import division
from collections import OrderedDict as _OrderedDict

from magni.afm.types.spectroscopy import Chunk as _Chunk
from magni.afm.types.spectroscopy import Grid as _Grid
from magni.afm.types.spectroscopy import Point as _Point


def build_object(params):
    """
    Build a hierarchical object-structure from header parameters.

    The hierarchical object-structure mimics that of `magni.afm.types`.

    Parameters
    ----------
    params : list or tuple
        The header parameters.

    Returns
    -------
    obj : object
        A hierarchical object-structure.

    See Also
    --------
    magni.afm.io.read_mi_file : Function using the present function.

    Notes
    -----
    This function splits the header parameters into file-related parameters and
    buffer-related parameters.

    """

    groups = [[]]

    for name, value in params:
        if name == 'bufferLabel':
            groups.append([])

        groups[-1].append((name, value))

    groups[0].append(groups[-1].pop())
    attrs = dict(groups[0])

    for name in ('xPixels', 'yPixels'):
        if name not in attrs.keys():
            raise IOError('The file must have a {!r} header parameter.'
                          .format(name))

        if not isinstance(attrs[name], int):
            raise IOError('The {!r} file header must have type {!r}.'
                          .format(name, int))

    buffers = [_build_buffer(attrs['fileType'], index, params_)
               for index, params_ in enumerate(groups[1:])]

    if len(buffers) == 0:
        raise IOError("The file must have at least one 'bufferLabel' header.")

    if attrs['fileType'].lower() == 'spectroscopy':
        _expand_buffers(buffers)

    obj = {'attrs': attrs, 'buffers': buffers}
    return _handle_format_inconsistency(obj)


def _build_buffer(file_type, index, params):
    """
    Build a buffer-like object-structure from buffer parameters.

    For spectroscopy buffers, this function converts parameters which contain
    "sub-parameters" to objects with attributes.

    Parameters
    ----------
    file_type : str
        The .mi file type.
    index : int
        The index of the buffer.
    params : list or tuple
        The buffer parameters.

    Returns
    -------
    obj : object
        The buffer-like object-structure.

    See Also
    --------
    build_object : Function using the present function.

    """

    if file_type.lower() == 'image':
        buffer_ = {'attrs': dict(params)}
    else:  # 'spectroscopy'
        allowed_params = ('chunk',) if index > 0 else ('grid', 'point')
        multiparams = {'grid': {'types': _Grid.params, 'min_len': 9},
                       'point': {'types': _Point.params, 'min_len': 3},
                       'chunk': {'types': _Chunk.params, 'min_len': 7}}

        # grid xDirection and yDirection are stored as integers
        multiparams['grid']['types'] = _OrderedDict(_Grid.params.items())
        multiparams['grid']['types']['xDirection'] = int
        multiparams['grid']['types']['yDirection'] = int

        attrs = []
        data = []

        for name, value in params:
            if name in multiparams.keys():
                if name not in allowed_params:
                    raise IOError('Buffer #{!r} may not contain {!r} headers.'
                                  .format(index, name))

                types = multiparams[name]['types']
                min_len = multiparams[name]['min_len']
                params = []

                if len(value) < min_len:
                    msg = ('Each {!r} buffer header must contain at least '
                           '{!r} values.')
                    raise IOError(msg.format(name, len(types)))

                for value, param, type_ in zip(value, *zip(*types.items())):
                    if not isinstance(value, type_):
                        msg = ('The values of each {!r} buffer header must '
                               'have types {!r}.')
                        raise IOError(msg.format(name, types))

                    params.append((param, value))

                data.append({'type': name, 'attrs': dict(params)})
            else:
                attrs.append((name, value))

        buffer_ = {'attrs': dict(attrs), 'data': data}

    return buffer_


def _expand_buffers(buffers):
    """
    Expand the buffer-like object-structures.

    Grid parameters specify implicit point parameters. Furthermore, some chunk
    parameters may be implicitly specified. These implicit parameters should be
    made explicit.

    Parameters
    ----------
    buffers : list or tuple
        The buffer-like object-structures.

    See Also
    --------
    build_object : Function using the present function.
    _generate_grid_points : Make implicit point parameters explicit.
    _generate_implicit_chunks : Make implicit chunk parameters explicit.

    Notes
    -----
    This function relies on the two functions, `_generate_grid_points` and
    `_generate_implicit_chunks` to make the implicit parameters explicit.

    """

    indices = {'grid': 0, 'point': 0}

    for i, item in enumerate(buffers[0]['data']):
        if item['attrs']['index'] != indices[item['type']]:
            raise IOError('The {!r} buffer parameter indices are erroneous.'
                          .format(item['type']))

        if item['type'] == 'grid':
            try:
                attr = 'xDirection'
                item['attrs'][attr] = ('trace', 'retrace')[item['attrs'][attr]]
                attr = 'yDirection'
                item['attrs'][attr] = ('up', 'down')[item['attrs'][attr]]
            except IndexError:
                msg = "The {!r} of 'grid' buffer parameters must be in {!r}."
                raise IOError(msg.format(attr, (0, 1)))

            points = _generate_grid_points(item['attrs'], indices['point'])
            item['points'] = points
            indices['point'] = indices['point'] + len(points) * len(points[0])

        indices[item['type']] = indices[item['type']] + 1

    for buffer_ in buffers[1:]:
        buffer_['data'] = _generate_implicit_chunks(indices['point'],
                                                    buffer_['data'])


def _generate_grid_points(attrs, index):
    """
    Make implicit grid point parameters explicit.

    Parameters
    ----------
    attrs : list or tuple
        The attributes of the grid.
    index : int
        The index of the first grid point.

    Returns
    -------
    points : tuple
        The grid points.

    See Also
    --------
    _expand_buffers : Function using the present function.

    """

    points = []
    spacing = attrs['pointSpacing']
    x_start = attrs['xCenter'] - (attrs['xPoints'] - 1) / 2 * spacing
    y_start = attrs['yCenter'] - (attrs['yPoints'] - 1) / 2 * spacing

    x_dir = slice(None, None, 1 if attrs['xDirection'] == 'trace' else -1)
    y_dir = slice(None, None, 1 if attrs['yDirection'] == 'down' else -1)

    for y_index in range(attrs['yPoints']).__getitem__(y_dir):
        points.append([])
        y = y_start + y_index * spacing

        for x_index in range(attrs['xPoints']).__getitem__(x_dir):
            x = x_start + x_index * spacing
            point = {'index': index, 'xCoordinate': x, 'yCoordinate': y}
            points[-1].append({'attrs': point})
            index = index + 1

        points[-1] = points[-1].__getitem__(x_dir)

    points = points.__getitem__(y_dir)
    return points


def _generate_implicit_chunks(npoints, explicit_chunks):
    """
    Make implicit chunk parameters explicit.

    Parameters
    ----------
    npoints : int
        The number of points.
    explicit_chunks : list
        The explicit chunks.

    Returns
    -------
    chunks : list
        The chunks including both explicit and implicit chunks.

    See Also
    --------
    _expand_buffer : Function using the present function.

    """

    chunks = []
    point_index = 0
    point_chunks = []
    stop_chunk = {'attrs': {'pointIndex': npoints, 'label': '_emptychunk_'}}

    for chunk in explicit_chunks + [stop_chunk]:
        index = chunk['attrs'].get('pointIndex', 0)

        for point_index in range(point_index + 1, index):
            for point_chunk in point_chunks:
                point_chunk = point_chunk.copy()
                point_chunk['attrs'] = point_chunk['attrs'].copy()
                point_chunk['attrs']['pointIndex'] = point_index
                chunks.append(point_chunk)

        if (chunk['attrs'].get('label', '').lower() == '_emptychunk_' and
                index <= npoints):
            break

        if point_index < index < npoints:
            point_index = point_index + 1
            point_chunks = []

        if index != point_index:
            raise IOError("The 'chunk' buffer parameter indices are "
                          "erroneous.")

        chunks.append(chunk)
        point_chunks.append(chunk)

    return chunks


def _handle_format_inconsistency(obj):
    """
    Handle format inconsistency.

    The inconsistency is the usage of string values for the 'trace' header
    parameter which is specified to have a boolean value.

    Parameters
    ----------
    obj : object
        The hierarchical object-structure.

    Returns
    -------
    obj : object
        The updated hierarchical object-structure.

    See Also
    --------
    build_object : Function using the present function.

    """

    if obj['attrs']['fileType'].lower() == 'image':
        for buffer_ in obj['buffers']:
            if ('trace' in buffer_['attrs'].keys() and
                    isinstance(buffer_['attrs']['trace'], str)):
                if buffer_['attrs']['trace'].lower() == 'trace':
                    buffer_['attrs']['trace'] = True
                elif buffer_['attrs']['trace'].lower() == 'retrace':
                    buffer_['attrs']['trace'] = False

    return obj
