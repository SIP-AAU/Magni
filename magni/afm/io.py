"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing input/output functionality for MI files.

Routine listings
----------------
read_mi_file(path)
    Read MI file and output an instance of an appropriate class.

See Also
--------
magni.afm.types : Data container classes.

"""


from __future__ import division
import os

import numpy as np

from magni.afm import types as _types
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate as _validate


@_decorate_validation
def _validate_read_mi_file(path):
    """
    Validate the `read_mi_file` function.

    See Also
    --------
    read_mi_file : The validated function.
    magni.utils.validation.validate : Validation.

    """

    _validate(path, path, {'type': str})


def read_mi_file(path):
    """
    Read MI file and output an instance of an appropriate class.

    Parameters
    ----------
    path : str
        The path of the MI file.

    Returns
    -------
    obj : None
        An instance of an appropriate class depending on the content of the MI
        file.

    Notes
    -----
    See the specification of the MI file format for an understanding of the
    steps performed in reading the MI file. An MI file can contain different
    types of data and thus the class of the output can vary.

    Examples
    --------
    An example of how to use read_mi_file to read the example MI file provided
    with the package:

    >>> from magni.afm.io import read_mi_file
    >>> path = magni.utils.split_path(magni.__path__[0])[0]
    >>> path = path + 'examples' + os.sep + 'example.mi'
    >>> mi_file = read_mi_file(path)

    """

    _validate_read_mi_file(path)

    if not os.path.isfile(path):
        raise IOError("The file {!r} does not exist.".format(path))

    try:
        f = open(path, 'rb')
        buf = f.read()
    finally:
        f.close()

    index = buf.find(b'data')
    index = buf.find(b'\n', index)
    hdrs = buf[:index].split(b'\n')
    hdrs = [(str(hdr[:14].decode()), str(hdr[14:].decode())) for hdr in hdrs]
    hdrs = [(key.strip(), _convert_mi_value(val)) for key, val in hdrs]

    if hdrs[0][0].lower() != 'filetype' or hdrs[-1][0].lower() != 'data':
        raise IOError("The file {!r} is not a valid MI file.".format(path))

    buf = buf[index + 1:]

    if hdrs[0][1].lower() == 'image':
        data = _convert_mi_image_data(buf, hdrs[-1][1])
        obj = _types.Image(data, hdrs)
    elif hdrs[0][1].lower() == 'spectroscopy':
        raise NotImplementedError("'Spectroscopy' file type not implemented.")
    else:
        raise IOError("Unknown file type {!r}.".format(hdrs[0][1]))

    return obj


def _convert_mi_image_data(buf, datatype):
    """
    Convert the data part of an MI image file to a 1D numpy.ndarray.

    Parameters
    ----------
    buf : str
        The raw data part of an MI image file.
    datatype : str
        A string specifying how to interpret the data in the data buffer.

    Returns
    -------
    data : numpy.ndarray
        The converted data part.

    Notes
    -----
    See the specification of the MI file format for a list of datatypes and an
    explanation of their meaning.

    """

    if datatype == 'BINARY':
        data = np.frombuffer(buf, np.int16)
        data = np.float64(data) / 2**15
    elif datatype == 'BINARY_32':
        data = np.frombuffer(buf, np.int32)
        data = np.float64(data) / 2**31
    elif datatype == 'ASCII':
        data = np.int16([int(val) for val in buf.split()])
        data = np.float64(data) / 2**15
    elif datatype == 'ASCII_ABSOLUTE':
        data = np.float64([float(val) for val in buf.split()])
    elif datatype == 'ASCII_MULTICOLUMN':
        data = np.float64([[float(val) for val in row.split()]
                           for row in buf.split('\n')]).flatten('F')
    else:
        raise IOError("Unknown data type {!r}.".format(datatype))

    return data


def _convert_mi_value(string):
    """
    Convert the value of an MI header line to a meaningful Python type.

    Parameters
    ----------
    string : str
        The string representation of the MI header line value.

    Returns
    -------
    value : None
        The converted value.

    Notes
    -----
    See the specification of the MI file format for an explanation of the
    different value types and the conversion from the string representation.

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
        value = [_convert_mi_value(part) for part in parts]
        value = string if isinstance(value[0], str) else tuple(value)

    return value
