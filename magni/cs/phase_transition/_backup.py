"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing backup capabilities for the monte carlo simulations.

The backup stores the simulation results and the simulation timings pointwise
for the points in the delta-rho simulation grid. The set function targets a
specific point while the get function targets the entire grid in order to keep
the overhead low.

Routine listings
----------------
create(path)
    Create the HDF5 backup database with the required arrays.
get(path)
    Return which of the results have been stored.
set(path, ij_tuple, stat_time, stat_dist)
    Store the simulation data of a specified point.

See Also
--------
magni.cs.phase_transition.config : Configuration options.

Notes
-----
In practice, the backup database includes an additional array for tracking for
which points data has been stored. By first storing the data and then modifying
this array, the data is guaranteed to have been stored, when the array is
modified.

"""

from __future__ import division
import os

import numpy as np

from magni.cs.phase_transition import config as _conf
from magni.utils.multiprocessing import File as _File


def create(path):
    """
    Create the HDF5 backup database with the required arrays.

    The required arrays are an array for the simulation results, an array for
    the simulation timings, and an array for tracking the status.

    Parameters
    ----------
    path : str
        The path of the HDF5 backup database.

    See Also
    --------
    magni.cs.phase_transition.config : Configuration options.

    """

    shape = [_conf[key] for key in ['delta', 'rho', 'monte_carlo']]
    shape = [len(shape[0]), len(shape[1]), shape[2]]
    time = dist = np.zeros(shape)
    status = np.zeros(shape[:2], np.bool8)

    try:
        with _File(path, 'w') as f:
            f.create_array('/', 'time', time)
            f.create_array('/', 'dist', dist)
            f.create_array('/', 'status', status)
    except BaseException as e:
        if os.path.exists(path):
            os.remove(path)

        raise e


def get(path):
    """
    Return which of the results have been stored.

    The returned value is a copy of the boolean status array indicating which
    of the results have been stored.

    Parameters
    ----------
    path : str
        The path of the HDF5 backup database.

    Returns
    -------
    status : ndarray
        The boolean status array.

    """

    with _File(path, 'r') as f:
        done = f.root.status[:]

    return done


def set(path, ij_tuple, stat_time, stat_dist):
    """
    Store the simulation data of a specified point.

    Parameters
    ----------
    path : str
        The path of the HDF5 backup database.
    ij_tuple : tuple
        A tuple (i, j) containing the parameters i, j as listed below.
    i : int
        The delta-index of the point in the delta-rho grid.
    j : int
        The rho-index of the point in the delta-rho grid.
    stat_dist : ndarray
        The simulation results of the specified point.
    stat_time : ndarray
        The simulation timings of the specified point.

    """

    i, j = ij_tuple

    with _File(path, 'a') as f:
        f.root.time[i, j] = stat_time
        f.root.dist[i, j] = stat_dist
        f.root.status[i, j] = True
