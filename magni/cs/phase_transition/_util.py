"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the public function of the magni.cs.phase_transition
subpackage.

"""

from __future__ import division
import os
import re
import types

from magni.cs.phase_transition import _analysis
from magni.cs.phase_transition import _simulation
from magni.utils.multiprocessing import File as _File
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate as _validate


@_decorate_validation
def _validate_determine(algorithm, path, label, overwrite):
    """
    Validate the `determine` function.

    See Also
    --------
    determine : The validated function.
    magni.utils.validation.validate : Validation.

    """

    _validate(algorithm, 'algorithm', {'type': types.FunctionType})
    _validate(path, 'path', {'type': str})
    _validate(label, 'label', {'type': str})

    # regular expression matching invalid characters
    regexp = re.compile(r'[^a-zA-Z0-9 ,.\-_/]')

    if regexp.search(label):
        raise RuntimeError("label may not contain {!r}."
                           .format(regexp.search(label).group()))

    # regular expression matching labels without empty path components
    regexp = re.compile(r'^([^/]+/)*[^/]+$')

    if not regexp.search(label):
        raise RuntimeError('label may not contain empty path components.')

    _validate(overwrite, 'overwrite', {'type': bool})


def determine(algorithm, path, label='default', overwrite=False):
    """
    Determine the phase transition of a reconstruction algorithm.

    The phase transition is determined from a number of monte carlo simulations
    on a delta-rho grid for a given problem suite.

    Parameters
    ----------
    algorithm : function
        A function handle to the reconstruction algorithm.
    path : str
        The path of the HDF5 database where the results should be stored.
    label : str
        The label assigned to the phase transition (the default is 'default').
    overwrite : bool
        A flag indicating if an existing phase transition should be overwritten
        if it has the same path and label (the default is False).

    See Also
    --------
    magni.cs.phase_transition.config : Configuration options.
    magni.cs.phase_transition._simulation.run : The actual simulation.
    magni.cs.phase_transition._analysis.run : The actual phase determination.

    Examples
    --------
    An example of how to use this function is provided in the `examples` folder
    in the `cs-phase_transition.ipynb` ipython notebook file.

    """

    _validate_determine(algorithm, path, label, overwrite)

    if os.path.isfile(path):
        with _File(path, 'r') as f:
            if '/' + label in f:
                if overwrite:
                    f.remove_node('/' + label, recursive=True)
                else:
                    raise IOError("{!r} already uses the label {!r}."
                                  .format(path, label))

    _simulation.run(algorithm, path, label)
    _analysis.run(path, label)
