"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality for converting a file stream.

The reading of an .mi file is logically separated into four steps of which the
functionality provided by this module performs the first step.

Routine listings
----------------
convert_stream(stream)
    Convert a file stream to flat, basic variables.

"""

from __future__ import division

import numpy as np


def convert_stream(stream):
    """
    Convert a file stream to flat, basic variables.

    The flat, basic variables are separated into header parameters and the
    data. The header parameters are name, value pairs whereas the data is a
    1-dimensional numeric array.

    Parameters
    ----------
    stream : str
        The file stream.

    Returns
    -------
    params : tuple
        The header parameters.
    data : numpy.ndarray
        The data.

    See Also
    --------
    magni.afm.io.read_mi_file : Function using the present function.

    Notes
    -----
    This function splits the file stream into header and data, and splits the
    header into parameter name, value pairs.

    """

    index = stream.find(b'data')
    index = stream.find(b'\n', index)
    stream_params = stream[:index]
    stream_data = stream[index + 1:]

    params = stream_params.split(b'\n')
    params = [(str(param[:14].decode()), str(param[14:].decode()))
              for param in params]
    params = [(name.strip(), _convert_parameter_value(value))
              for name, value in params]
    params = tuple(params)

    if params[0][0] != 'fileType':
        raise IOError("The file must have a 'fileType' header parameter.")

    if params[-1][0] != 'data':
        raise IOError("The file must have a 'data' header parameter.")

    if params[0][1].lower() == 'image':
        data = _convert_image_data(stream_data, params[-1][1])
    elif params[0][1].lower() == 'spectroscopy':
        data = _convert_spectroscopy_data(stream_data, params[-1][1])
    else:
        msg = 'The value of  >>fileType<<, {!r}, must be in {!r}.'
        raise IOError(msg.format(params[0][1], ('Image', 'Spectroscopy')))

    return params, data


def _convert_image_data(buffer_, data_type):
    """
    Convert the file stream image data to a 1-dimensional numeric array.

    Parameters
    ----------
    buffer_ : str
        The file stream image data.
    data_type : str
        The data type of the file stream image data.

    Returns
    -------
    data : numpy.ndarray
        The data as a 1-dimensional numeric array.

    See Also
    --------
    convert_stream : Function using the present function.

    """

    if data_type.lower() == 'binary':
        data = np.frombuffer(buffer_, np.int16)
        data = np.float64(data) / 2**15
    elif data_type.lower() == 'binary_32':
        data = np.frombuffer(buffer_, np.int32)
        data = np.float64(data) / 2**31
    elif data_type.lower() == 'ascii':
        data = map(int, buffer_.strip().split())
        data = np.float64(data) / 2**15
    elif data_type.lower() == 'ascii_absolute':
        data = map(float, buffer_.strip().split())
        data = np.float64(data)
    elif data_type.lower() == 'ascii_multicolumn':
        data = buffer_.strip().split()
        data = [map(float, row.split()) for row in data]
        data = np.float64(data).flatten('F')
    else:
        raise IOError('Unknown data type {!r}.'.format(data_type))

    return data


def _convert_parameter_value(string):
    """
    Convert a file stream parameter value to a basic python value.

    The converted value is either a boolean, string, floating point, or integer
    value or a list containing a mix of these.

    Parameters
    ----------
    string : str
        The file stream parameter value.

    Returns
    -------
    value : None
        The converted, basic python value.

    See Also
    --------
    convert_stream : Function using the present function.

    """

    parts = string.split()

    if len(parts) == 0:
        value = None
    elif len(parts) == 1:
        if string.lower() == 'true':
            value = True
        elif string.lower() == 'false':
            value = False
        else:
            try:
                value = string
                value = float(string)
                value = int(string)
            except ValueError:
                pass
    else:
        value = [_convert_parameter_value(part) for part in parts]
        value = tuple(value) if not isinstance(value[0], str) else string

    return value


def _convert_spectroscopy_data(buffer_, data_type):
    """
    Convert the file stream spectroscopy data to a 1-dimensional numeric array.

    Parameters
    ----------
    buffer_ : str
        The file stream spectroscopy data.
    data_type : str
        The data type of the file stream spectroscopy data.

    Returns
    -------
    data : numpy.ndarray
        The data as a 1-dimensional numeric array.

    See Also
    --------
    convert_stream : Function using the present function.

    """

    if data_type.lower() == 'binary':
        data = np.frombuffer(buffer_, np.float32)
        data = np.float64(data)
    elif data_type.lower() == 'ascii_multicolumn':
        data = buffer_.strip().split()[1:]
        data = [map(float, row.split()[2:]) for row in data]
        data = np.float64(data).flatten('F')
    else:
        raise IOError('Unknown data type {!r}.'.format(data_type))

    return data
