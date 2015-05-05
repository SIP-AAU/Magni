"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing the actual simulation functionality.

Routine listings
----------------
run(algorithm, path, label)
     Simulate a reconstruction algorithm.

See Also
--------
magni.cs.phase_transition._config : Configuration options.

Notes
-----
The results of the simulation are backed up throughout the simulation. In case
the simulation is interrupted during execution, the simulation will resume from
the last backup point when run again.

"""

from __future__ import division
import os
import time

import numpy as np

from magni.cs.phase_transition import _backup
from magni.cs.phase_transition import config as _conf
from magni.cs.phase_transition import _data
from magni.utils.multiprocessing import File as _File
from magni.utils.multiprocessing import process as _process
from magni.utils import split_path as _split_path


def run(algorithm, path, label):
    """
    Simulate a reconstruction algorithm.

    The simulation results are stored in a HDF5 database rather than returned
    by the function.

    Parameters
    ----------
    algorithm : function
        A function handle to the reconstruction algorithm.
    path : str
        The path of the HDF5 database where the results should be stored.
    label : str
        The label assigned to the simulation results.

    """

    tmp_dir = _split_path(path)[0] + '.tmp' + os.sep
    tmp_file = tmp_dir + label.replace('/', '#') + '.hdf5'

    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    if not os.path.isfile(tmp_file):
        _backup.create(tmp_file)

    done = _backup.get(tmp_file)
    shape = [len(_conf['delta']), len(_conf['rho']), _conf['monte_carlo']]
    np.random.seed(_conf['seed'])
    seeds = np.random.randint(0, 2**30, shape)

    tasks = [(algorithm, (i, j), seeds[i, j], tmp_file)
             for i in range(shape[0]) for j in range(shape[1])
             if not done[i, j]]

    _process(_simulate, args_list=tasks)

    with _File(tmp_file, 'r') as f:
        stat_time = f.root.time[:]
        stat_dist = f.root.dist[:]

    with _File(path, 'a') as f:
        f.create_array('/' + label, 'time', stat_time, createparents=True)
        f.create_array('/' + label, 'dist', stat_dist, createparents=True)

    os.remove(tmp_file)

    if len(os.listdir(tmp_dir)) == 0:
        os.removedirs(tmp_dir)


def _simulate(algorithm, ij_tuple, seeds, path):
    """
    Run a number of monte carlo simulations in a single delta-rho point.

    The result of a simulation is the simulation error distance, i.e., the
    ratio between the energy of the coefficient residual and the energy of the
    coefficient vector. The time of the simulation is the execution time of the
    reconstruction attempt.

    Parameters
    ----------
    algorithm : function
        A function handle to the reconstruction algorithm.
    ij_tuple : tuple
        A tuple (i, j) containing the parameters i, j as listed below.
    i : int
        The delta-index of the point in the delta-rho grid.
    j : int
        The rho-index of the point in the delta-rho grid.
    seeds : ndarray
        The seeds to pass to numpy.random when generating the problem suite
        instances.
    path : str
        The path of the HDF5 backup database.

    See Also
    --------
    magni.cs.phase_transition._data.generate_matrix : Matrix generation.
    magni.cs.phase_transition._data.generate_vector : Coefficient vector
        generation.

    """

    i, j = ij_tuple

    n = _conf['problem_size']
    m = int(np.round(n * _conf['delta'][i]))
    k = int(np.round(m * _conf['rho'][j]))

    stat_time = np.zeros(_conf['monte_carlo'], dtype=np.float64)
    stat_dist = stat_time.copy()

    if k > 0:
        for l in range(_conf['monte_carlo']):
            np.random.seed(seeds[l])
            A = _data.generate_matrix(m, n)
            alpha = _data.generate_vector(n, k)
            y = A.dot(alpha)

            start = time.time()
            alpha_hat = algorithm(y, A)
            end = time.time()

            stat_time[l] = end - start
            stat_dist[l] = (np.linalg.norm(alpha_hat - alpha) /
                            np.linalg.norm(alpha))**2

    _backup.set(path, (i, j), stat_time, stat_dist)
