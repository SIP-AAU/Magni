"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing plotting for the `magni.cs.phase_transition` subpackage.

Routine listings
----------------
plot_phase_transitions(curves, plot_l1=True, output_path=None)
    Function for plotting phase transition boundary curves.
plot_phase_transition_colormap(dist, delta, rho, plot_l1=True,
    output_path=None)
    Function for plotting reconstruction probabilities in the phase space.
plot_phase_transition_computation_times(time, delta, rho, output_path=None)
    Function for plotting average computation times in the phase space.

"""

from __future__ import division
from itertools import cycle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pkg_resources import parse_version as _parse_version

from magni.cs.phase_transition import config as _conf
from magni.utils.plotting import linestyles as _linestyles
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric

if _parse_version(mpl.__version__) >= _parse_version('1.5.0'):
    _mpl_prop_era = True
else:
    _mpl_prop_era = False


def plot_phase_transitions(curves, plot_l1=True, output_path=None,
                           legend_loc='upper left', errorevery=None,
                           reference_curves=None):
    r"""
    Plot of a set of phase transition boundary curves.

    The set of phase transition boundary curves are plotted an saved under the
    `output_path`, if specified. The `curves` must be a list of dictionaries
    each having keys *delta*, *rho*, and *label*. *delta* must be an `ndarray`
    of :math:`\delta` values in the phase space. *rho* must be an `ndarray` of
    the corresponding :math:`\rho` values in the phase space. *label* must be a
    `str` describing the curve. Optionally, a `curves` dictionary may have an
    *yerr* key with a corresponding 2-by-"len(delta)" array as value. The first
    row in this array specifices the location of the upper error bars whereas
    the second row specifies the location of the lower error bars.

    Parameters
    ----------
    curves : list
        A list of dicts describing the curves to plot.
    plot_l1 : bool, optional
        Whether or not to plot the theoretical :math:`\ell_1` curve (the
        default is True).
    output_path : str, optional
        Path (including file type extension) under which the plot is saved (the
        default value is None, which implies that the plot is not saved).
    legend_loc : str
        Location of legend as a `matplotlib` legend location string (the
        default is 'upper left', which implies that the legend is placed in the
        upper left corner of the plot.)
    errorevery : int
        The subsampling of error bars if used (the default value is None, which
        implies that the default Matplotlib value for errorevery is used.)
    reference_curves : list, optional
        The list of dicts describing the reference curves to plot.

    Notes
    -----
    The plotting is done using `matplotlib`, which implies that an open figure
    containing the plot will result from using this function.

    Tabulated values of the theoretical :math:`\ell_1` phase transition
    boundary is available at
    http://people.maths.ox.ac.uk/tanner/polytopes.shtml

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.cs.phase_transition.plotting import plot_phase_transitions
    >>> delta = np.array([0.1, 0.2, 0.9])
    >>> rho = np.array([0.1, 0.3, 0.8])
    >>> curves = [{'delta': delta, 'rho': rho, 'label': 'data1'}]
    >>> output_path = 'phase_transitions.pdf'
    >>> plot_phase_transitions(curves, output_path=output_path)

    """

    @_decorate_validation
    def validate_input():
        _levels('curves', (_generic(None, 'collection'),
                           _generic(None, 'mapping',
                                    has_keys=('delta', 'rho', 'label'))))

        for i, curve in enumerate(curves):
            _numeric(('curves', i, 'delta'), 'floating', range_='[0;1]',
                     shape=(-1,))
            _numeric(('curves', i, 'rho'), 'floating', range_='[0;1]',
                     shape=(curve['delta'].shape[0],))
            _generic(('curves', i, 'label'), 'string')

            if 'yerr' in curve:
                _numeric(('curves', i, 'yerr'), 'floating', range_='[0;1]',
                         shape=(2, curve['delta'].shape[0]))

        _numeric('plot_l1', 'boolean')
        _generic('output_path', 'string', ignore_none=True)
        _generic('legend_loc', 'string')
        _numeric('errorevery', 'integer', range_='[1;inf)', ignore_none=True)
        _levels('reference_curves', (_generic(None, 'collection',
                                              ignore_none=True),
                                     _generic(None, 'mapping',
                                              has_keys=(
                                                  'delta', 'rho', 'label'))))
        if reference_curves is not None:
            for i, ref_curve in enumerate(reference_curves):
                _numeric(('reference_curves', i, 'delta'), 'floating',
                         range_='[0;1]', shape=(-1,))
                _numeric(('reference_curves', i, 'rho'), 'floating',
                         range_='[0;1]', shape=(ref_curve['delta'].shape[0],))
                _generic(('reference_curves', i, 'label'), 'string')

                if 'style' in ref_curve:
                    _generic(('reference_curves', i, 'style'), 'mapping')

    validate_input()

    fig, axes = plt.subplots(1, 1)
    if not _mpl_prop_era:
        # Emulate matplotlib cycler
        colors = plt.rcParams['axes.color_cycle']
        style_cycle = cycle([{'color': color, 'ls': linestyle}
                             for linestyle in _linestyles for color in colors])
    else:
        # Handled by matplotlib cycler
        style_cycle = cycle([{}])

    error_style = {'capthick': 2}
    if errorevery is not None:
        error_style['errorevery'] = errorevery

    for curve in curves:
        if 'yerr' in curve:
            yerr = [curve['yerr'][0, :] - curve['rho'],
                    curve['rho'] - curve['yerr'][1, :]]
        else:
            yerr = None

        kwargs = dict(next(style_cycle), **error_style)
        axes.errorbar(curve['delta'], curve['rho'], label=curve['label'],
                      yerr=yerr, **kwargs)

    _plot_extra_curves(axes, plot_l1, reference_curves)

    handles, labels = axes.get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels),
                                  key=lambda t: t[1].lower()))
    leg = axes.legend(handles, labels, loc=legend_loc)
    leg.get_frame().set_facecolor('1.0')
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    axes.set_xlabel(r'$\delta = m/n$')
    axes.set_ylabel(r'$\rho = k/m$')

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)


def plot_phase_transition_colormap(dist, delta, rho, plot_l1=True,
                                   output_path=None):
    r"""
    Create a colormap of the phase space reconstruction probabilities.

    The `delta` and `rho` values span a 2D grid in the phase space.
    Reconstruction probabilities are then calculated from the `dist` 3D array
    of reconstruction error distances. The resulting 2D grid of reconstruction
    probabilites is visualised over the square centers in this 2D grid using a
    colormap. Values in this grid at lower indicies correspond to lower values
    of :math:`\delta` and :math:`\rho`. If `plot_l1` is True, then the
    theoretical l1 curve is overlayed the colormap. The colormap is saved under
    the `output_path`, if specified.

    Parameters
    ----------
    dist : ndarray
        A 3D array of reconstruction error distances.
    delta : ndarray
        :math:`\delta` values used in the 2D grid.
    rho : ndarray
        :math:`\rho` values used in the 2D grid.
    plot_l1 : bool
        Whether or not to plot the theoretical :math:`\ell_1` curve. (the
        default is True)
    output_path : str, optional
        Path (including file type extension) under which the plot is saved (the
        default value is None which implies, that the plot is not saved).

    See Also
    --------
    magni.cs.phase_transition.plotting.plot_phase_transition_computation_times
        Plot average computation times.
    magni.cs.phase_transition.io.load_phase_transition : Loading phase
        transitions from an HDF database.

    Notes
    -----
    The plotting is done using `matplotlib`, which implies that an open figure
    containing the plot will result from using this function.

    The values in `delta` and `rho` are assumed to be equally spaced.

    Due to the *centering* of the color coded rectangles, they are not
    necessarily square towards the ends of the intervals defined by `delta` and
    `rho`.

    Tabulated values of the theoretical :math:`\ell_1` phase transition
    boundary is available at
    http://people.maths.ox.ac.uk/tanner/polytopes.shtml

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.cs.phase_transition.plotting import (
    ...     plot_phase_transition_colormap)
    >>> delta = np.array([0.2, 0.5, 0.8])
    >>> rho = np.array([0.3, 0.6])
    >>> dist = np.array([[[1.35e-08, 1.80e-08], [1.08, 1.11]],
    ... [[1.40e-12, 8.32e-12], [8.57e-01, 7.28e-01]], [[1.92e-13, 1.17e-13],
    ... [2.10e-10,   1.12e-10]]])
    >>> out_path = 'phase_transition_cmap.pdf'
    >>> plot_phase_transition_colormap(dist, delta, rho, output_path=out_path)

    """

    @_decorate_validation
    def validate_input():
        _numeric('delta', 'floating', range_='[0;1]', shape=(-1,))
        _numeric('rho', 'floating', range_='[0;1]', shape=(-1,))
        _numeric('dist', 'floating', range_='[0;inf]',
                 shape=(delta.shape[0], rho.shape[0], -1))
        _numeric('plot_l1', 'boolean')
        _generic('output_path', 'string', ignore_none=True)

    validate_input()

    NMSE_tolerance = 10**(-_conf['SNR'] / 10)
    probs = np.sum(
        dist < NMSE_tolerance, axis=-1, dtype=float) / np.size(dist, -1)

    d_delta = (delta[1] - delta[0]) / 2
    d_rho = (rho[1] - rho[0]) / 2

    x = np.hstack((np.array([0]), delta[1:] - d_delta, np.array([1])))
    y = np.hstack((np.array([0]), rho[1:] - d_rho, np.array([1])))

    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 1)
    p_mesh = axes.pcolormesh(X, Y, probs.T, vmin=0, vmax=1, edgecolor='face')
    c_bar = plt.colorbar(p_mesh)
    c_bar.set_label('Estimated probability of reconstruction')

    # Apply vector graphics render workaround to avoid white gaps in colorbar
    # See: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.colorbar
    c_bar.solids.set_edgecolor("face")

    if plot_l1:
        _plot_theoretical_l1(axes)
        leg = axes.legend(loc='upper left')
        leg.get_frame().set_facecolor('1.0')

    axes.set_xticks(np.arange(11) / 10)
    axes.set_yticks(np.arange(11) / 10)
    axes.set_xlabel(r'$\delta = m/n$')
    axes.set_ylabel(r'$\rho = k/m$')

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)


def plot_phase_transition_computation_times(time, delta, rho,
                                            output_path=None):
    r"""
    Create a colormap of the phase space average algorithm computation time.

    The `delta` and `rho` values span a 2D grid in the phase space.
    Average computation times are then calculated from the `time` 3D array of
    reconstruction times. The resulting 2D grid of average reconstruction times
    is visualised over the square centers in this 2D grid using a colormap.
    Values in this grid at lower indicies correspond to lower values of
    :math:`\delta` and :math:`\rho`. The colormap is saved under the
    `output_path`, if specified.

    Parameters
    ----------
    time : ndarray
        A 3D array of algorithm computation times.
    delta : ndarray
        :math:`\delta` values used in the 2D grid.
    rho : ndarray
        :math:`\rho` values used in the 2D grid.
    output_path : str, optional
        Path (including file type extension) under which the plot is saved (the
        default value is None which implies, that the plot is not saved).

    See Also
    --------
    magni.cs.phase_transition.plotting.plot_phase_transition_colormap : Plot
        phase transition probabilities.
    magni.cs.phase_transition.io.load_phase_transition : Loading phase
        transitions from an HDF database.

    Notes
    -----
    The plotting is done using `matplotlib`, which implies that an open figure
    containing the plot will result from using this function.

    The values in `delta` and `rho` are assumed to be equally spaced.

    Due to the *centering* of the color coded rectangles, they are not
    necessarily square towards the ends of the intervals defined by `delta` and
    `rho`.

    Examples
    --------
    For example,

    >>> import numpy as np
    >>> from magni.cs.phase_transition.plotting import (
    ...     plot_phase_transition_computation_times)
    >>> delta = np.array([0.2, 0.5, 0.8])
    >>> rho = np.array([0.3, 0.6])
    >>> times = np.array([[[1.35e-08, 1.80e-08], [1.08, 1.11]],
    ... [[1.40e-12, 8.32e-12], [8.57e-01, 7.28e-01]], [[1.92e-13, 1.17e-13],
    ... [2.10e-10,   1.12e-10]]])
    >>> out_path = 'computation_times_cmap.pdf'
    >>> plot_phase_transition_computation_times(
    ...     times, delta, rho, output_path=out_path)

    """

    @_decorate_validation
    def validate_input():
        _numeric('delta', 'floating', range_='[0;1]', shape=(-1,))
        _numeric('rho', 'floating', range_='[0;1]', shape=(-1,))
        _numeric('time', 'floating', range_='[0;inf]',
                 shape=(delta.shape[0], rho.shape[0], -1))
        _generic('output_path', 'string', ignore_none=True)

    validate_input()

    mean_times = time.mean(axis=-1)

    d_delta = (delta[1] - delta[0]) / 2
    d_rho = (rho[1] - rho[0]) / 2

    x = np.hstack((np.array([0]), delta[1:] - d_delta, np.array([1])))
    y = np.hstack((np.array([0]), rho[1:] - d_rho, np.array([1])))

    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 1)
    p_mesh = axes.pcolormesh(X, Y, mean_times.T, edgecolor='face')
    c_bar = plt.colorbar(p_mesh)
    c_bar.set_label('Average computation time [s]')

    # Apply vector graphics render workaround to avoid white gaps in colorbar
    # See: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.colorbar
    c_bar.solids.set_edgecolor("face")

    axes.set_xticks(np.arange(11) / 10)
    axes.set_yticks(np.arange(11) / 10)
    axes.set_xlabel(r'$\delta = m/n$')
    axes.set_ylabel(r'$\rho = k/m$')

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)


def _plot_extra_curves(axes, plot_l1, reference_curves):
    """
    Plot any extra curves that might be needed in a phase space plot.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The matplotlib Axes instance on which the theoretical l1 phase
        transition should be plotted.
    plot_l1 : bool, optional
        Whether or not to plot the theoretical :math:`\ell_1` curve (the
        default is True).
    reference_curves : list
        The list of dicts describing the reference curves to plot.

    """

    if plot_l1:
        # Add theoretical ell_1 plot
        _plot_theoretical_l1(axes)

    if reference_curves is not None:
        # Add additional reference curves
        for ref_curve in reference_curves:
            axes.plot(ref_curve['delta'], ref_curve['rho'],
                      label=ref_curve['label'], **ref_curve['style'])


def _plot_theoretical_l1(axes):
    """
    Plot the theoretical l1 phase transition on the `axes`.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The matplotlib Axes instance on which the theoretical l1 phase
        transition should be plotted.

    Notes
    -----
    The plotted theoretical :math:`\ell1` phase transition is based on
    tabulated values of available at
    http://people.maths.ox.ac.uk/tanner/polytopes.shtml

    """

    delta = [0, 0, 0, 0.001, 0.008, 0.021, 0.038, 0.058, 0.078, 0.1, 0.122,
             0.144, 0.167, 0.19, 0.212, 0.235, 0.258, 0.282, 0.305, 0.329,
             0.352, 0.375, 0.399, 0.422, 0.445, 0.468, 0.49, 0.513, 0.535,
             0.558, 0.58, 0.603, 0.626, 0.647, 0.669, 0.691, 0.712, 0.733,
             0.754, 0.774, 0.794, 0.813, 0.832, 0.851, 0.868, 0.884, 0.9,
             0.915, 0.929, 0.942, 0.953, 0.963, 0.972, 0.98, 0.986, 0.991,
             0.996, 0.998, 1]
    rho = [0, 0.025, 0.051, 0.077, 0.103, 0.125, 0.144, 0.16, 0.176, 0.19,
           0.202, 0.215, 0.227, 0.238, 0.249, 0.26, 0.271, 0.282, 0.293, 0.304,
           0.315, 0.326, 0.337, 0.348, 0.359, 0.37, 0.381, 0.392, 0.404, 0.415,
           0.428, 0.44, 0.453, 0.466, 0.479, 0.493, 0.507, 0.522, 0.537, 0.552,
           0.568, 0.585, 0.602, 0.621, 0.639, 0.658, 0.678, 0.699, 0.721,
           0.744, 0.767, 0.791, 0.815, 0.84, 0.864, 0.891, 0.921, 0.949, 1]

    axes.plot(delta, rho, c='k', ls='--', label=r'Theoretical $\ell_1$')
