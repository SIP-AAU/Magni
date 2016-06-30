"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing input/output functions to databases containing results from
reproducible research.

Routine listings
----------------
annotate_database(h5file)
    Function for annotating an existing HDF5 database.
chase_database(h5file)
    Function for chasing an existing HDF5 database.
create_database(h5file)
    Function for creating a new annotated and chased HDF5 database.
read_annotations(h5file)
    Function for reading annotations in an HDF5 database.
read_chases(h5file)
    Function for reading chases in an HDF5 database.
remove_annotations(h5file)
    Function for removing annotations in an HDF5 database.
remove_chases(h5file)
    Function for removing chases in an HDF5 database.
write_custom_annotation(h5file, annotation_name, annotation_value,
    annotations_sub_group=None)
    Write a custom annotation to an HDF5 database.

See Also
--------
magni.reproducibility._annotation.get_conda_info : Conda annotation
magni.reproducibility._annotation.get_git_revision : Git annotation
magni.reproducibility._annotation.get_platform_info : Platform annotation
magni.reproducibility._annotation.get_datetime : Date and time annotation
magni.reproducibility._annotation.get_magni_config : Magni config annotation
magni.reproducibility._annotation.get_magni_info : Magni info annotation
magni.reproducibility._chase.get_main_file_name : Magni main file name chase
magni.reproducibility._chase.get_main_file_source : Magni source code chase
magni.reproducibility._chase.get_main_source : Magni main source code chase
magni.reproducibility._chase.get_stack_trace : Magni stack trace chase

"""

from __future__ import division
import json
import os

import tables

from magni.reproducibility import _annotation
from magni.reproducibility import _chase
from magni.utils.multiprocessing import File as _File
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric


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

    >>> import magni
    >>> from magni.reproducibility.io import annotate_database
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='a') as h5file:
    ...     annotate_database(h5file)

    """

    @_decorate_validation
    def validate_input():
        _generic('h5file', tables.file.File)

    validate_input()

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


def chase_database(h5file):
    """
    Chase an HDF5 database to track information about stack and source code.

    The chase consist of a group in the root of the `h5file` having nodes that
    each profide information about the program execution that led to this chase
    of the database.

    Parameters
    ----------
    h5file : tables.file.File
        The handle to the HDF5 database that should be chased.

    See Also
    --------
    magni.reproducibility._chase.get_main_file_name : Name of main file
    magni.reproducibility._chase.get_main_file_source : Main file source code
    magni.reproducibility._chase.get_main_source : Source code around main
    magni.reproducibility._chase.get_stack_trace : Complete stack trace

    Notes
    -----
    The chase include the following information:

    * main_file_name - Name of the main file/script that called this function
    * main_file_source - Full source code of the main file/script
    * main_source - Extract of main file source code that called this function
    * stack_trace - Complete stack trace up until the call to this function

    Examples
    --------
    Chase the database named 'db.hdf5':

    >>> import magni
    >>> from magni.reproducibility.io import chase_database
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='a') as h5file:
    ...     chase_database(h5file)

    """

    @_decorate_validation
    def validate_input():
        _generic('h5file', tables.file.File)

    validate_input()

    chases = {'main_file_name': json.dumps(_chase.get_main_file_name()),
              'main_file_source': json.dumps(_chase.get_main_file_source()),
              'main_source': json.dumps(_chase.get_main_source()),
              'stack_trace': json.dumps(_chase.get_stack_trace())}

    try:
        chase_group = h5file.create_group('/', 'chases')
        for chase in chases:
            h5file.create_array(chase_group, chase, obj=chases[chase].encode())
        h5file.flush()

    except tables.NodeError:
        raise tables.NodeError('The database has already been chased. ' +
                               'Remove the existing chase prior to ' +
                               '(re)chasing the database.')


def create_database(path, overwrite=True):
    """
    Create a new HDF database that is annotated and chased.

    A new HDF database is created and it is annotated using
    `magni.reproducibility.io.annotate_database` and chased using
    `magni.reproducibility.io.annotate_database`. If the `overwrite` flag is
    true and existing database at `path` is overwritten.

    Parameters
    ----------
    path : str
        The path to the HDF file that is to be created.
    overwrite : bool
        The flag that indicates if an existing database should be overwritten.

    See Also
    --------
    magni.reproducibility.io.annotate_database : Database annotation
    magni.reproducibility.io.chase_database : Database chase

    Examples
    --------
    Create a new database named 'new_db.hdf5':

    >>> from magni.reproducibility.io import create_database
    >>> create_database('new_db.hdf5')

    """

    @_decorate_validation
    def validate_input():
        _generic('path', 'string')
        _numeric('overwrite', 'boolean')

    validate_input()

    if not overwrite and os.path.exists(path):
        raise IOError('{!r} already exists in filesystem.'.format(path))

    with _File(path, mode='w') as h5file:
        annotate_database(h5file)
        chase_database(h5file)


def read_annotations(h5file):
    """
    Read the annotations to an HDF5 database.

    Parameters
    ----------
    h5file : tables.file.File
        The handle to the HDF5 database from which the annotations are read.

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

    >>> import magni
    >>> from magni.reproducibility.io import read_annotations
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='r') as h5file:
    ...    annotations = read_annotations(h5file)

    """

    @_decorate_validation
    def validate_input():
        _generic('h5file', tables.file.File)

    validate_input()

    try:
        h5_annotations = h5file.get_node('/', name='annotations')
    except tables.NoSuchNodeError:
        raise tables.NoSuchNodeError('The database has not been annotated.')

    annotations = dict()
    _recursive_annotation_read(h5_annotations, annotations)

    return annotations


def read_chases(h5file):
    """
    Read the chases to an HDF5 database.

    Parameters
    ----------
    h5file : tables.file.File
        The handle to the HDF5 database from which the chases are read.

    Returns
    -------
    chasess : dict
        The chases read from the HDF5 database.

    Raises
    ------
    ValueError
        If the chases to the HDF5 database does not conform to the Magni chases
        standard.

    Notes
    -----
    The returned dict holds a key for each chase in the database. The value
    corresponding to a given key is a string. See
    `magni.reproducibility.chase_database` for examples of such chases.

    Examples
    --------
    Read chases from the database named 'db.hdf5':

    >>> import magni
    >>> from magni.reproducibility.io import read_chases
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='r') as h5file:
    ...    chases = read_chases(h5file)

    """

    @_decorate_validation
    def validate_input():
        _generic('h5file', tables.file.File)

    validate_input()

    try:
        h5_chases = h5file.get_node('/', name='chases')
    except tables.NoSuchNodeError:
        raise tables.NoSuchNodeError('The database has not been chased.')

    h5_chase_dict = h5_chases._v_leaves
    chases = dict()
    try:
        for chase in h5_chase_dict:
            chases[chase] = json.loads(h5_chase_dict[chase].read().decode())
    except ValueError as e:
        raise ValueError('Unable to read the {!r} chase '.format(chase) +
                         'It seems that the chase does not conform to the ' +
                         'Magni chase standard ({!r}).'.format(e.args[0]))

    return chases


def remove_annotations(h5file):
    """
    Remove the annotations from an HDF5 database.

    Parameters
    ----------
    h5file : tables.file.File
        The handle to the HDF5 database from which the annotations are removed.

    Examples
    --------
    Remove annotations from the database named 'db.hdf5':

    >>> import magni
    >>> from magni.reproducibility.io import remove_annotations
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='a') as h5file:
    ...    remove_annotations(h5file)

    """

    @_decorate_validation
    def validate_input():
        _generic('h5file', tables.file.File)

    validate_input()

    try:
        h5file.remove_node('/', 'annotations', recursive=True)
        h5file.flush()
    except tables.NoSuchNodeError:
        pass


def remove_chases(h5file):
    """
    Remove the chases from an HDF5 database.

    Parameters
    ----------
    h5file : tables.file.File
        The handle to the HDF5 database from which the chases are removed.

    Examples
    --------
    Remove chases from the database named 'db.hdf5':

    >>> import magni
    >>> from magni.reproducibility.io import remove_chases
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='a') as h5file:
    ...    remove_chases(h5file)

    """

    @_decorate_validation
    def validate_input():
        _generic('h5file', tables.file.File)

    validate_input()

    try:
        h5file.remove_node('/', 'chases', recursive=True)
        h5file.flush()
    except tables.NoSuchNodeError:
        pass


def write_custom_annotation(h5file, annotation_name, annotation_value,
                            annotations_sub_group=None):
    """
    Write a custom annotation to an HDF5 database.

    The annotation is written to the `h5file` under the `annotation_name` such
    that it holds the `annotation_value`.

    Parameters
    ----------
    h5file : tables.file.File
        The handle to the HDF5 database to which the annotation is written.
    annotation_name : str
        The name of the annotation to write.
    annotation_value : a JSON serialisable object
        The annotation value to write.
    annotations_sub_group : str
        The group node under "/annotations" to which the custom annotation is
        written (the default is None which implies that the custom annotation
        is written directly under "/annotations").

    Notes
    -----
    The `annotation_value` must be a JSON seriablisable object.

    Examples
    --------
    Write a custom annotation to an HDF5 database.

    >>> import magni
    >>> from magni.reproducibility.io import write_custom_annotation
    >>> annotation_name = 'custom_annotation'
    >>> annotation_value = 'the value'
    >>> with magni.utils.multiprocessing.File('db.hdf5', mode='a') as h5file:
    ...    write_custom_annotation(h5file, annotation_name, annotation_value)
    ...    annotations = magni.reproducibility.io.read_annotations(h5file)
    >>> str(annotations['custom_annotation'])
    'the value'

    """

    @_decorate_validation
    def validate_input():
        _generic('h5file', tables.file.File)
        _generic('annotation_name', 'string')
        _generic('annotations_sub_group', 'string', ignore_none=True)

    validate_input()

    if annotations_sub_group is not None:
        annotations_group = '/'.join(['/annotations', annotations_sub_group])
    else:
        annotations_group = '/annotations'

    try:
        ann_val = json.dumps(annotation_value)
    except TypeError:
        raise TypeError('The annotation value does not have a valid JSON ' +
                        'representation. It may not be used as an annotation.')

    try:
        h5file.create_array(annotations_group, annotation_name,
                            obj=ann_val.encode(), createparents=True)
        h5file.flush()
    except tables.NodeError:
        raise tables.NodeError(
            'The annotation "{!r}" already exists '.format(annotation_name) +
            'in the database. Remove the old annotation before placing a ' +
            'new one.')


def _recursive_annotation_read(h5_annotations, out_annotations_dict):
    """
    Recursively read annotations from an annotation group

    Parameters
    ----------
    h5_annotations : tables.group.Group
        The group to read annotations from.
    out_annotations_dict : dict
        The dictionary to store the read annotations in.

    """

    leaves = h5_annotations._v_leaves
    subgroups = h5_annotations._v_groups

    # Read leaves
    try:
        for annotation_name, annotation_value in leaves.items():
            out_annotations_dict[annotation_name] = json.loads(
                annotation_value.read().decode())
    except ValueError as e:
        raise ValueError('Unable to read the {!r} '.format(annotation_name) +
                         'annotation. It seems that the annotation ' +
                         'does not conform to the Magni annotation ' +
                         'standard ({!r}).'.format(e.args[0]))

    # Recursively handle subgroups
    for subgroup in subgroups:
        out_annotations_dict[subgroup] = dict()
        _recursive_annotation_read(
            subgroups[subgroup], out_annotations_dict[subgroup])
