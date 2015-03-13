"""
..
    Copyright (c) 2014-2015, Magni developers.
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
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric


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
    magni.cs.phase_transition._simulation.run : The actual simulation.
    magni.cs.phase_transition._analysis.run : The actual phase determination.

    Examples
    --------
    An example of how to use this function is provided in the `examples` folder
    in the `cs-phase_transition.ipynb` ipython notebook file.

    """

    @_decorate_validation
    def validate_input():
        _generic('algorithm', 'function')
        _generic('path', 'string')
        _generic('label', 'string')

        # regular expression matching invalid characters
        match = re.search(r'[^a-zA-Z0-9 ,.\-_/]', label)

        if match is not None:
            msg = 'The value of >>label<<, {!r}, may not contain {!r}.'
            raise RuntimeError(msg.format(label, match.group()))

        # regular expression matching labels without empty path components
        match = re.search(r'^([^/]+/)*[^/]+$', label)

        if match is None:
            msg = "The value of >>label<<, {!r}, may not contain '' folders."
            raise RuntimeError(msg.format(label))

        _numeric('overwrite', 'boolean')

    validate_input()

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
