"""
..
    Copyright (c) 2014-2016, Magni developers.
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
import random
import sys
import time

import numpy as np

from magni.cs.phase_transition import _backup
from magni.cs.phase_transition import config as _conf
from magni.cs.phase_transition import _data
from magni.utils.multiprocessing import File as _File
from magni.utils.multiprocessing import process as _process
from magni.utils import split_path as _split_path

if sys.version_info[0] == 2:
    iter_range = xrange(2**30)
else:
    iter_range = range(2**32)


def run(algorithm, path, label, pre_simulation_hook=None):
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
        The label assigned to the simulation results
    pre_simumlation_hook : callable
        A handle to a callable which should be run *just* before the call to
        the reconstruction algorithm (the default is None, which implies that
        no pre hook is run).

    """

    tmp_dir = _split_path(path)[0] + '.tmp' + os.sep
    tmp_file = tmp_dir + label.replace('/', '#') + '.hdf5'

    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    if not os.path.isfile(tmp_file):
        _backup.create(tmp_file)

    done = _backup.get(tmp_file)
    shape = [len(_conf['delta']), len(_conf['rho'])]
    random.seed(_conf['seed'])
    seeds = np.array(
        random.sample(iter_range, shape[0] * shape[1])).reshape(shape)

    tasks = [(algorithm, (i, j), seeds[i, j], tmp_file, pre_simulation_hook)
             for i in range(shape[0]) for j in range(shape[1])
             if not done[i, j]]

    _process(_simulate, args_list=tasks, maxtasks=_conf['maxpoints'])

    with _File(tmp_file, 'r') as f:
        stat_time = f.root.time[:]
        stat_dist = f.root.dist[:]
        stat_mse = f.root.mse[:]
        stat_norm = f.root.norm[:]

    with _File(path, 'a') as f:
        f.create_array('/' + label, 'time', stat_time, createparents=True)
        f.create_array('/' + label, 'dist', stat_dist, createparents=True)
        f.create_array('/' + label, 'mse', stat_mse, createparents=True)
        f.create_array('/' + label, 'norm', stat_norm, createparents=True)

    os.remove(tmp_file)

    if len(os.listdir(tmp_dir)) == 0:
        os.removedirs(tmp_dir)


def _simulate(algorithm, ij_tuple, seed, path, pre_simulation_hook=None):
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
    seed : int
        The seed to use in the random number generator when generating the
        problem suite instances.
    path : str
        The path of the HDF5 backup database.
    pre_simulation_hook : callable
        A handle to a callable which should be run *just* before the call to
        the reconstruction algorithm (the default is None, which implies that
        no pre hook is run).

    See Also
    --------
    magni.cs.phase_transition._config: Configuration options.
    magni.cs.phase_transition._data.generate_matrix : Matrix generation.
    magni.cs.phase_transition._data.generate_vector : Coefficient vector
        generation.

    Notes
    -----
    The `pre_simulation_hook` may be used to setup the simulation to match the
    specfic simulation parameters, e.g. if an oracle estimator is used in the
    reconstruction algorithm. The `pre_simulation_hook` takes one argument
    which is the locals() dict.

    The following reconstruction statistics are computed:

    * time: Measured algorithm run time in seconds.
    * dist: Normalised mean squared error (NMSE) - (
      ||alpha_hat - alpha|| / ||alpha||)^2
    * mse: Mean squared error (MSE) - 1/n * ||alpha_hat - alpha||^2
    * norm: True vector norm - ||alpha||

    """

    i, j = ij_tuple

    n = _conf['problem_size']
    m = int(np.round(n * _conf['delta'][i]))
    k = int(np.round(m * _conf['rho'][j]))
    noise = _conf['noise']

    stat_time = np.zeros(_conf['monte_carlo'], dtype=np.float64)
    stat_dist = stat_time.copy()
    stat_mse = stat_time.copy()
    stat_norm = stat_time.copy()

    if k > 0:
        np.random.seed(seed)
        for l in range(_conf['monte_carlo']):
            A = _data.generate_matrix(m, n)
            alpha = _data.generate_vector(n, k)

            if noise is not None:
                e = _data.generate_noise(m, n, k)
                y = A.dot(alpha) + e
            else:
                y = A.dot(alpha)

            if pre_simulation_hook is not None:
                pre_simulation_hook(locals())

            start = time.time()
            alpha_hat = algorithm(y, A, **_conf['algorithm_kwargs'])
            end = time.time()

            error_norm = np.linalg.norm(alpha_hat - alpha)
            alpha_norm = np.linalg.norm(alpha)

            stat_time[l] = end - start
            stat_dist[l] = (error_norm / alpha_norm)**2
            stat_mse[l] = 1.0/n * error_norm**2
            stat_norm[l] = alpha_norm

    _backup.set(path, (i, j), stat_time, stat_dist, stat_mse, stat_norm)
