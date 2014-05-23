"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing input/output functions to databases containing results from
reproducible research.

Routine listings
----------------
annotate_database(h5file)
    Function for annotating an existing HDF5 database.
read_annotations(h5file)
    Function for reading annotations in an HDF5 database.
remove_annotations(h5file)
    Function for removing annotations in an HDF5 database.

See Also
--------
magni.reproducibility._annotation.get_conda_info : Conda annotation
magni.reproducibility._annotation.get_git_revision : Git annotation
magni.reproducibility._annotation.get_platform_info : Platform annotation
magni.reproducibility._annotation.get_datetime : Date and time annotation
magni.reproducibility._annotation.get_magni_config : Magni config annotation
magni.reproducibility._annotation.get_magni_info : Magni info annotation

"""

from __future__ import division
import json

import tables

from magni.reproducibility import _annotation
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate as _validate


@_decorate_validation
def _validate_annotate_database(h5file):
    """
    Validate the `annotate_database` function.

    See Also
    --------
    annotate_database : The validated function.
    magni.utils.validation.validate : Validation.

    """

    _validate(h5file, 'h5file', {'class': tables.file.File})


def annotate_database(h5file):
    """
    Annotate an HDF5 database with information about Magni and the platform.

    The annotation consists of a group in the root of the `h5file` having nodes
    that each provide information about Magni or the platform on which this
    function is run.

    Parameters
    ----------
    h5file : tables.file.File
        The handle to the HDF5 database that should be annotated.

    See Also
    --------
    magni.reproducibility._annotation.get_conda_info : Conda annotation
    magni.reproducibility._annotation.get_git_revision : Git annotation
    magni.reproducibility._annotation.get_platform_info : Platform annotation
    magni.reproducibility._annotation.get_datetime : Date and time annotation
    magni.reproducibility._annotation.get_magni_config : Magni config
        annotation
    magni.reproducibility._annotation.get_magni_info : Magni info annotation

    Notes
    -----
    The annotations of the database includes the following:

    * conda_info - Information about Continuum Anacononda install
    * git_revision - Git revision and tag of Magni
    * platform_info - Information about the current platform (system)
    * datetime - The current date and time
    * magni_config - Infomation about the current configuration of Magni
    * magni_info - Information from `help(magni)`

    Examples
    --------
    Annotate the database named 'db.hdf5':

    >>> from magni.reproducibility.io import annotate_database
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='a') as h5file:
    ...     annotate_database(h5file)

    """

    _validate_annotate_database(h5file)

    annotations = {'conda_info': json.dumps(_annotation.get_conda_info()),
                   'git_revision': json.dumps(_annotation.get_git_revision()),
                   'platform_info': json.dumps(
                       _annotation.get_platform_info()),
                   'datetime': json.dumps(_annotation.get_datetime()),
                   'magni_config': json.dumps(_annotation.get_magni_config()),
                   'magni_info': json.dumps(_annotation.get_magni_info())}

    try:
        annotations_group = h5file.create_group('/', 'annotations')
        for annotation in annotations:
            h5file.create_array(annotations_group, annotation,
                                obj=annotations[annotation].encode())
        h5file.flush()
    except tables.NodeError:
        raise tables.NodeError('The database has already been annotated. ' +
                               'Remove the existing annotation prior to ' +
                               '(re)annotating the database.')


@_decorate_validation
def _validate_read_annotations(h5file):
    """
    Validate the `read_annotations` function.

    See Also
    --------
    read_annotations : The validated function.
    magni.utils.validation.validate : Validation.

    """

    _validate(h5file, 'h5file', {'class': tables.file.File})


def read_annotations(h5file):
    """
    Read the annotations to an HDF5 database.

    Parameters
    ----------
    h5file : tables.file.File
        The handle to the HDF5 database from which the annotations is read.

    Returns
    -------
    annotations : dict
        The annotations read from the HDF5 database.

    Raises
    ------
    ValueError
        If the annotations to the HDF5 database does not conform to the Magni
        annotation standard.

    Notes
    -----
    The returned dict holds a key for each annotation in the database. The
    value corresponding to a given key is in itself a dict. See
    `magni.reproducibility.annotate_database` for examples of such annotations.

    Examples
    --------
    Read annotations from the database named 'db.hdf5':

    >>> from magni.reproducibility.io import read_annotations
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='r') as h5file:
    ...    annotations = read_annotations(h5file)

    """

    _validate_read_annotations(h5file)

    try:
        h5_annotations = h5file.get_node('/', name='annotations')
    except tables.NoSuchNodeError:
        raise tables.NoSuchNodeError('The database has not been annotated.')

    h5_annotations_dict = h5_annotations._v_leaves
    annotations = dict()
    try:
        for annotation in h5_annotations_dict:
            annotations[annotation] = json.loads(
                h5_annotations_dict[annotation].read().decode())
    except ValueError as e:
        raise ValueError('Unable to read the {!r} '.format(annotation) +
                         'annotation. It seems that the annotation ' +
                         'does not conform to the Magni annotation ' +
                         'standard ({!r}).'.format(e.args[0]))

    return annotations


@_decorate_validation
def _validate_remove_annotations(h5file):
    """
    Validate the `remove_annotations` function.

    See Also
    --------
    remove_annotations : The validated function.
    magni.utils.validation.validate : Validation.

    """

    _validate(h5file, 'h5file', {'class': tables.file.File})


def remove_annotations(h5file):
    """
    Remove the annotations from an HDF5 database.

    Parameters
    ----------
    h5file : tables.file.File
        The handle to the HDF5 database from which the annotations is removed.

    Examples
    --------
    Remove annotations from the database named 'db.hdf5':

    >>> from magni.reproducibility.io import remove_annotations
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='a') as h5file:
    ...    remove_annotations(h5file)

    """

    _validate_remove_annotations(h5file)

    try:
        h5file.remove_node('/', 'annotations', recursive=True)
        h5file.flush()
    except tables.NoSuchNodeError:
        pass
