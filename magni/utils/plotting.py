"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing utilities for control of plotting using `matplotlib`.

The module has a number of public attributes which provide settings for
colormap cycles, linestyle cycles, and marker cycles that may be used in
combination with `matplotlib`.

Routine listings
----------------
setup_matplotlib(settings={}, cmap=None)
    Function that set the Magni default `matplotlib` configuration.
colour_collections : dict
    Collections of colours that may be used in e.g., a `matplotlib`
    color_cycle / prop_cycle.
seq_cmaps : list
    Names of `matplotlib.cm` colormaps optimized for sequential data.
div_cmaps : list
    Names of `matplotlib.cm` colormaps optimized for diverging data.
linestyles : list
    A subset of linestyles from `matplotlib.lines`
markers : list
    A subset of markers from `matplotlib.markers`

Examples
--------
Use the default Magni matplotlib settings.

>>> import magni
>>> magni.utils.plotting.setup_matplotlib()

Get the normalised 'Blue' colour brew from the psp colour map:

>>> magni.utils.plotting.colour_collections['psp']['Blue']
((0.1255, 0.502, 0.8745),)

"""

from __future__ import division
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pkg_resources import parse_version as _parse_version

from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric

if _parse_version(mpl.__version__) >= _parse_version('1.5.0'):
    import cycler
    _mpl_prop_era = True
else:
    _mpl_prop_era = False


class _ColourCollection(object):
    """
    A container for colour maps.

    A single colour is stored as an RGB 3-tuple of integers in the interval
    [0,255]. A set of related colours is termed a colour brew and is stored as
    a list of colours. A set of related colour brews is termed a colour
    collection and is stored as a dictionary. The dictionary key identifies
    the name of the colour collection whereas the value is the list of colour
    brews.

    The default colour collections named "cb*" are colorblind safe, print
    friendly, and photocopy-able. They have been created using the online
    ColorBrewer 2.0 tool [1]_.

    Parameters
    ----------
    brews : dict
        The dictionary of colour brews from which the colour collection is
        created.

    Notes
    -----
    Each colour brew is a list (or tuple) of length 3 lists (or tuples) of RGB
    values.

    References
    ----------
    .. [1] M. Harrower and C. A. Brewer, "ColorBrewer.org: An Online Tool for
       Selecting Colour Schemes for Maps", *The Cartographic Journal*, vol. 40,
       pp. 27-37, 2003 (See also: http://colorbrewer2.org/)

    """

    def __init__(self, brews):
        @_decorate_validation
        def validate_input():
            _levels('brews', (_generic(None, 'mapping'),
                              _generic(None, 'explicit collection'),
                              _generic(None, 'explicit collection', len_=3),
                              _numeric(None, 'integer', range_='[0;255]')))

        validate_input()

        self._brews = brews

    def __getitem__(self, name):
        """
        Return a single colour brew.

        The returned colour brew is normalised in the sense of matplotlib
        normalised rgb values, i.e., colours are 3-tuples of floats in the
        interval [0, 1].

        Parameters
        ----------
        name : str
            Name of the colour brew to return.

        Returns
        -------
        brew : tuple
            A colour brew list.

        """

        @_decorate_validation
        def validate_input():
            _generic('name', 'string', value_in=tuple(self._brews.keys()))

        validate_input()

        return tuple([tuple([round(val / 255, 4) for val in colour])
                      for colour in self._brews[name]])


colour_collections = {
    'cb4': _ColourCollection({
        'OrRd': ((254, 240, 217), (253, 204, 138), (252, 141, 89),
                 (215, 48, 31)),
        'PuOr': ((230, 97, 1), (253, 184, 99), (178, 171, 210),
                 (94, 60, 153))}),
    'cb3': _ColourCollection({
        'BuGn': ((229, 245, 249), (153, 216, 201), (44, 162, 95)),
        'BuPu': ((224, 236, 244), (158, 188, 218), (136, 86, 167)),
        'GuBu': ((224, 243, 219), (168, 221, 181), (67, 162, 202)),
        'OrRd': ((254, 232, 200), (253, 187, 132), (227, 74, 51)),
        'PuBu': ((236, 231, 242), (166, 189, 219), (43, 140, 190)),
        'PuBuGn': ((236, 226, 240), (166, 189, 219), (28, 144, 153)),
        'PuRd': ((231, 225, 239), (201, 148, 199), (221, 28, 119)),
        'RdPu': ((253, 224, 221), (250, 159, 181), (197, 27, 138)),
        'YlGn': ((247, 252, 185), (173, 221, 142), (49, 163, 84)),
        'YlGnBu': ((237, 248, 177), (127, 205, 187), (44, 127, 184)),
        'YlOrBr': ((255, 247, 188), (254, 196, 79), (217, 95, 14)),
        'YlOrRd': ((255, 237, 160), (254, 178, 76), (240, 59, 32)),
        'Blues': ((222, 235, 247), (158, 202, 225), (49, 130, 189)),
        'Greens': ((229, 245, 224), (161, 217, 155), (49, 163, 84)),
        'Greys': ((240, 240, 240), (189, 189, 189), (99, 99, 99)),
        'Purples': ((239, 237, 245), (188, 189, 220), (117, 107, 177)),
        'Reds': ((254, 224, 210), (252, 146, 114), (222, 45, 38)),
        'PuOr': ((241, 163, 64), (247, 247, 247), (153, 142, 195))}),
    'psp': _ColourCollection({
        'Blue': ((32, 128, 223),),
        'Orange': ((223, 128, 32),),
        'GreenY': ((128, 223, 32),),
        'Purple': ((128, 32, 223),),
        'Red': ((223, 32, 128),),
        'GreenB': ((223, 32, 128),)}),
    'bgg': _ColourCollection({
        'Black': ((0, 0, 0),),
        'Green': ((0, 191, 0),),
        'Grey': ((170, 170, 170),)})}


seq_cmaps = ['YlOrRd', 'YlOrRd_r', 'YlGnBu', 'YlGnBu_r', 'PuBuGn', 'PuBuGn_r',
             'YlOrBr', 'YlOrBr_r', 'BuGn', 'BuGn_r', 'GnBu', 'GnBu_r',
             'PuBu', 'PuBu_r', 'PuRd', 'PuRd_r']
div_cmaps = ['PRGn', 'PRGn_r', 'PiYG', 'PiYG_r', 'RdBu', 'RdBu_r', 'PuOr',
             'PuOr_r', 'RdGy', 'RdGy_r']
linestyles = ['-', '--', '-.', ':']
markers = ['o', '^', 'x', '+', 'd']


def setup_matplotlib(settings={}, cmap=None):
    """
    Adjust the configuration of `matplotlib`.

    Sets the default configuration of `matplotlib` to optimize for producing
    high quality plots of the data produced by the functionality provided in
    the Magni.

    Parameters
    ----------
    settings : dict, optional
       A dictionary of custom matplotlibrc settings. See examples for details
       about the structure of the dictionary.
    cmap : str or tuple, optional
       Colormap to be used by matplotlib (the default is None, which implices
       that the 'coolwarm' colormap is used). If a tuple is supplied it must
       be a ('colormap_name', matplotlib.colors.Colormap()) tuple.

    Raises
    ------
    UserWarning
       If the supplied custom settings are invalid.

    Examples
    --------
    For example, set lines.linewidth=2 and lines.color='r'.

    >>> from magni.utils.plotting import setup_matplotlib
    >>> custom_settings = {'lines': {'linewidth': 2, 'color': 'r'}}
    >>> setup_matplotlib(custom_settings)

    """

    @_decorate_validation
    def validate_input():
        _levels('settings', (_generic(None, 'mapping'),
                             _generic(None, 'mapping')))
        _generic('cmap', ('string', tuple), ignore_none=True)
        if isinstance(cmap, tuple):
            _generic(('cmap', 0), 'string')
            _generic(('cmap', 1), mpl.colors.Colormap)

    validate_input()

    global _settings, _cmap

    for name, setting in settings.items():
        if name in _settings:
            _settings[name].update(setting)
        else:
            _settings[name] = setting

    for name, setting in _settings.items():
        try:
            mpl.rc(name, **setting)
        except (AttributeError, KeyError):
            warnings.warn('Setting {!r} ignored.'.format(name), UserWarning)

    if cmap is not None:
        if isinstance(cmap, tuple):
            mpl.cm.register_cmap(name=cmap[0], cmap=cmap[1])
            cmap = cmap[0]
        plt.set_cmap(cmap)
    elif _cmap is not None:
        plt.set_cmap(_cmap)

    _settings = {}
    _cmap = None

if _mpl_prop_era:
    # Matplotlib >= 1.5.0
    _style_cycle = cycler.cycler('linestyle', linestyles)
    _color_cycle = cycler.cycler('color', colour_collections['cb4']['PuOr'])
    _prop_settings = {'axes': {'prop_cycle': _style_cycle * _color_cycle}}

else:
    _prop_settings = {
        'axes': {'color_cycle': colour_collections['cb4']['PuOr']}}

_settings = dict({'text': {'usetex': False},
                  'font': {'size': 12},
                  'mathtext': {'fontset': 'cm'},
                  'pdf': {'fonttype': 42},
                  'ps': {'fonttype': 42},
                  'legend': {'fontsize': 11},
                  'lines': {'linewidth': 2},
                  'figure': {
                      'figsize': (8.0, float(8.0 / ((1 + np.sqrt(5)) / 2))),
                      'dpi': 600},
                  'image': {'interpolation': 'none'}}, **_prop_settings)
_cmap = 'coolwarm'
