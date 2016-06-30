"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functionality for the analysis of dictionaries.

Routine listings
----------------
get_reconstructions(img, transform, fractions)
    Function to get a set of transform reconstructions.
show_coefficient_histogram(img, transforms, bins=None, range=None,
    output_path=None, fig_ext='pdf')
    Function to show a transform coefficient histogram.
show_psnr_energy_rolloff(img, reconstructions, fractions, return_vals=False,
    output_path=None, fig_ext='pdf')
    Function to show PSNR and retained energy rolloff of reconstructions.
show_reconstructions(coefficients, reconstructions, transform, fractions,
    output_path=None, fig_ext='pdf')
    Function to show tranforms reconstructions and coefficients.
show_sorted_coefficients(img, transforms, output_path=None, fig_ext='pdf')
    Function to show a plot of transform coefficients in sorted order.
show_transform_coefficients(img, transforms, output_path=None, fig_ext='pdf')
    Function to show transform coefficients.
show_transform_quantiles(img, transform, fraction=1.0, area_mask=None)
    Function to show quantiles of transform coefficients.

"""

from __future__ import division
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable

from magni.imaging import evaluation as _evaluation
from magni.imaging import visualisation as _visualisation
from magni.imaging import mat2vec as _mat2vec
from magni.imaging import vec2mat as _vec2mat
from magni.imaging.dictionaries import utils as _utils
from magni.imaging.visualisation import imshow as _imshow
from magni.imaging.visualisation import imsubplot as _imsubplot
from magni.reproducibility import io as _io
from magni.utils import split_path as _split_path
from magni.utils.multiprocessing import File as _File
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric
from magni.utils.validation import validate_levels as _levels


def get_reconstructions(img, transform, fractions):
    """
    Return transform reconstructions with different fractions of coefficents.

    The image `img` is transform coded using the specified `transform`.
    Reconstructions for the `fractions` of transform coefficients kept are
    returned along with the coefficients used in the reconstructions.

    Parameters
    ----------
    img : ndarray
        The image to get reconstructions of.
    transform : str
        The transform to use to obtain the reconstructions.
    fractions : list or tuple
        The fractions of coefficents to be used in the reconstructions.

    Returns
    -------
    coefficients : list
        The list of coefficients (each an ndarray) used in the reconstructions.
    reconstructions : list
        The list of reconstructions.

    Examples
    --------
    Get reconstructions from DCT based on 20% and 40% of the coefficients:

    >>> import numpy as np
    >>> from magni.imaging.dictionaries.analysis import get_reconstructions
    >>> img = np.arange(64).astype(np.float).reshape(8, 8)
    >>> transform = 'DCT'
    >>> fractions = (0.2, 0.4)
    >>> coefs, recons = get_reconstructions(img, transform, fractions)
    >>> len(recons)
    2
    >>> tuple(int(s) for s in coefs[0].shape)
    (8, 8)
    >>> tuple(int(s) for s in recons[0].shape)
    (8, 8)

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _generic('transform', 'string', value_in=_utils.get_transform_names())
        _levels('fractions', (_generic(None, 'explicit collection'),
                              _numeric(None, 'floating', range_='[0;1]')))

    @_decorate_validation
    def validate_output():
        _levels('coefficients',
                (_generic(None, 'explicit collection'),
                 _numeric(None, ('integer', 'floating', 'complex'),
                          shape=img.shape)))
        _levels('reconstructions',
                (_generic(None, 'explicit collection'),
                 _numeric(None, ('integer', 'floating', 'complex'),
                          shape=img.shape)))

    validate_input()

    coefficients = []
    reconstructions = []

    transform_matrix = _utils.get_function_handle(
        'matrix', transform)(img.shape)
    all_coefficients = _vec2mat(transform_matrix.conj().T.dot(_mat2vec(img)),
                                img.shape)
    sorted_coefficients = np.sort(np.abs(all_coefficients), axis=None)[::-1]

    for fraction in fractions:
        if int(fraction) == 1:
            used_coefficients = all_coefficients
        else:
            used_coefficients = np.zeros_like(all_coefficients)
            mask = np.abs(all_coefficients) > sorted_coefficients[
                int(np.round(fraction * all_coefficients.size)) - 1]
            used_coefficients[mask] = all_coefficients[mask]

        reconstruction = _vec2mat(
            transform_matrix.dot(_mat2vec(used_coefficients)).real, img.shape)

        coefficients.append(used_coefficients)
        reconstructions.append(reconstruction)

    validate_output()

    return coefficients, reconstructions


def show_coefficient_histogram(img, transforms, bins=None, range=None,
                               output_path=None, fig_ext='pdf'):
    """
    Show a histogram of coefficient values for different transforms.

    A histogram of the transform coefficient values for `img` using the
    different `transforms` are shown. If `output_path` is not None, the
    resulting figure and data used in the figure are saved.

    Parameters
    ----------
    img : ndarray
        The image to get transform coefficients for.
    transforms : list or tuple
        The names as strings of the transforms to use.
    bins : int
        The number of bins to use in the histogram (the default is None, which
        implies that the number of bins is determined based on the size of
        `img`).
    range : tuple
        The lower and upper range of the bins to use in the histogram (the
        default is None, which implies that the min and max values of `img`
        are used).
    output_path : str
        The output path (see notes below) to save the figure and data to (the
        default is None, which implies that the figure and data are not saved).
    fig_ext : str
        The figure extension determining the format of the saved figure (the
        default is 'pdf' which implies that the figure is saved as a PDF).

    See Also
    --------
    matplotlib.pyplot.hist : The underlying histogram plot function.

    Notes
    -----
    The `output_path` is specified as a path to a folder + an optional prefix
    to the file name. The remaining file name is fixed. If e.g, the fixed part
    of the file name was 'plot', then:

    * output_path = '/home/user/' would save the figure under /home/user/ with
      the name plot.pdf.
    * output_path = '/home/user/best' would save the figure under /home/user
      with the name best_plot.pdf.

    In addition to the saved figures, an annotated and chased HDF database with
    the data used to create the figures are also saved. The name of the HDF
    database is the same as for the figure with the exception that the file
    extension is '.hdf5'.

    Examples
    --------
    Save a coefficient histogram using the DCT and the DFT as transforms

    >>> import os, numpy as np
    >>> from magni.imaging.dictionaries import analysis as _a
    >>> img = np.arange(64).astype(np.float).reshape(8, 8)
    >>> transforms = ('DCT', 'DFT')
    >>> output_path = './histogram_test'
    >>> _a.show_coefficient_histogram(img, transforms, output_path=output_path)
    >>> current_dir = os.listdir('./')
    >>> for file in sorted(current_dir):
    ...     if 'histogram_test' in file:
    ...         print(file)
    histogram_test_coefficient_histogram.hdf5
    histogram_test_coefficient_histogram.pdf

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _levels('transforms',
                (_generic(None, 'explicit collection'),
                 _generic(None, 'string',
                          value_in=_utils.get_transform_names())))
        _numeric('bins', 'integer', range_='[1;inf)', ignore_none=True)
        _generic('output_path', 'string', ignore_none=True)
        _generic('fig_ext', 'string')
        # Range validated by matplotlib

    validate_input()

    fig, ax = plt.subplots(1, 1)

    datasets = dict()
    for transform in transforms:
        matrix_handle = _utils.get_function_handle('matrix', transform)
        coefficients = matrix_handle(img.shape).conj().T.dot(_mat2vec(img))

        if np.issubdtype(coefficients.dtype, np.complex):
            coefficients = np.abs(coefficients)

        label = '{} coefficients'.format(transform)
        alpha = 0.6
        edgecolor = 'none'

        if bins is not None:
            n, b, p = ax.hist(coefficients, bins=bins, range=range, log=True,
                              label=label, alpha=alpha, ec=edgecolor)
        else:
            n, b, p = ax.hist(coefficients, bins=10**int(np.log10(img.size)-1),
                              range=range, log=True, label=label, alpha=alpha,
                              ec=edgecolor)

        datasets[transform] = {'coefficients': coefficients, 'n': n, 'bins': b}

    leg = ax.legend()
    leg.get_frame().set_facecolor('1.0')
    ax.set_xlabel('Coefficient value (modulo for complex values)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of transform coefficients')
    for patch in leg.get_patches():
        patch.set_edgecolor(edgecolor)

    # Save figures and data
    if output_path is not None:
        _save_output(output_path, 'coefficient_histogram', fig, fig_ext,
                     datasets)


def show_psnr_energy_rolloff(img, reconstructions, fractions,
                             return_vals=False, output_path=None,
                             fig_ext='pdf'):
    """
    Show the PSNR and energy rolloff for the reconstructions.

    A plot of the Peak Signal to Noise Ratio (PSNR) and retained energy in the
    `recontructions` versus the `fractions` of coefficients used in the
    reconstructions is shown. If return_vals is True, the data used in the plot
    is returned. If `output_path` is not None, the resulting figure and data
    used in the figure are saved.

    Parameters
    ----------
    img : ndarray
        The image which the reconstructions are based on.
    reconstructions : list or tuple
        The reconstructions (each an ndarray) to show rolloff for.
    fractions : list or tuple
        The fractions of coefficents used in the reconstructions.
    return_vals : bool
        The flag indicating wheter or not to return the PSNR and energy values
        (the default is False, which indicate that the values are not
        returned).
    output_path : str
        The output path (see notes below) to save the figure and data to (the
        default is None, which implies that the figure and data are not saved).
    fig_ext : str
        The figure extension determining the format of the saved figure (the
        default is 'pdf' which implies that the figure is saved as a PDF).

    Returns
    -------
    psnrs : ndarray
       The PSNR values shown in the figure (only returned if return_vals=True).
    energy : ndarray
       The retained energy values shown in the figure (only returned if
       return_vals=True).

    Notes
    -----
    The `output_path` is specified as a path to a folder + an optional prefix
    to the file name. The remaining file name is fixed. If e.g, the fixed part
    of the file name was 'plot', then:

    * output_path = '/home/user/' would save the figure under /home/user/ with
      the name plot.pdf.
    * output_path = '/home/user/best' would save the figure under /home/user
      with the name best_plot.pdf.

    In addition to the saved figures, an annotated and chased HDF database with
    the data used to create the figures are also saved. The name of the HDF
    database is the same as for the figure with the exception that the file
    extension is '.hdf5'.

    Examples
    --------
    Save a PSNR and energy rolloff plot for reconstructions bases on the DCT:

    >>> import os, numpy as np
    >>> from magni.imaging.dictionaries import analysis as _a
    >>> img = np.arange(64).astype(np.float).reshape(8, 8)
    >>> transform = 'DCT'
    >>> fractions = (0.2, 0.4)
    >>> coefs, recons = _a.get_reconstructions(img, transform, fractions)
    >>> o_p = './rolloff_test'
    >>> _a.show_psnr_energy_rolloff(img, recons, fractions, output_path=o_p)
    >>> current_dir = os.listdir('./')
    >>> for file in sorted(current_dir):
    ...     if 'rolloff_test' in file:
    ...         print(file)
    rolloff_test_psnr_energy_rolloff.hdf5
    rolloff_test_psnr_energy_rolloff.pdf

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _levels('reconstructions',
                (_generic(None, 'explicit collection'),
                 _numeric(None, ('integer', 'floating', 'complex'),
                          shape=img.shape)))
        _levels('fractions', (_generic(None, 'explicit collection'),
                              _numeric(None, 'floating', range_='[0;1]')))
        _numeric('return_vals', 'boolean')
        _generic('output_path', 'string', ignore_none=True)
        _generic('fig_ext', 'string')

    validate_input()

    psnrs = np.zeros_like(fractions)
    energy = np.zeros_like(fractions)

    for k, reconstruction in enumerate(reconstructions):
        psnrs[k] = _evaluation.calculate_psnr(img, reconstruction,
                                              float(img.max()))
        energy[k] = _evaluation.calculate_retained_energy(img, reconstruction)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(fractions, psnrs)
    axes[0].set_xlabel('Fraction of coefficients')
    axes[0].set_ylabel('PSNR [dB]')
    axes[1].plot(fractions, energy)
    axes[1].set_xlabel('Fraction of coefficients')
    axes[1].set_ylabel('Retained energy [%]')
    fig.suptitle(('PSNR/Energy vs. fraction of coefficients used in ' +
                  'reconstruction'), fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # Save figures and data
    if output_path is not None:
        datasets = {'psnrs': {'fractions': fractions, 'values': psnrs},
                    'energy': {'fractions': fractions, 'values': energy}}
        _save_output(output_path, 'psnr_energy_rolloff', fig, fig_ext,
                     datasets)

    # Return values
    if return_vals:
        return psnrs, energy


def show_reconstructions(coefficients, reconstructions, transform, fractions,
                         output_path=None, fig_ext='pdf'):
    """
    Show reconstructions and corresponding coefficients.

    Parameters
    ----------
    coefficients : list or tuple
        The coefficients (each an ndarray) used in the reconstructions.
    reconstructions : list or tuple
        The reconstructions (each an ndarray) to show.
    transform : str
        The transform used to obtain the reconstructions.
    fractions : list or tuple
        The fractions of coefficents used in the reconstructions.
    output_path : str
        The output path (see notes below) to save the figure and data to (the
        default is None, which implies that the figure and data are not saved).
    fig_ext : str
        The figure extension determining the format of the saved figure (the
        default is 'pdf' which implies that the figure is saved as a PDF).

    Notes
    -----
    The `output_path` is specified as a path to a folder + an optional prefix
    to the file name. The remaining file name is fixed. If e.g, the fixed part
    of the file name was 'plot', then:

    * output_path = '/home/user/' would save the figure under /home/user/ with
      the name plot.pdf.
    * output_path = '/home/user/best' would save the figure under /home/user
      with the name best_plot.pdf.

    In addition to the saved figures, an annotated and chased HDF database with
    the data used to create the figures are also saved. The name of the HDF
    database is the same as for the figure with the exception that the file
    extension is '.hdf5'.

    Examples
    --------
    Save images of coefficients and reconstructions based on the DCT:

    >>> import os, numpy as np
    >>> from magni.imaging.dictionaries import analysis as _a
    >>> img = np.arange(64).astype(np.float).reshape(8, 8)
    >>> trans = 'DCT'
    >>> fracs = (0.2, 0.4)
    >>> coefs, recons = _a.get_reconstructions(img, trans, fracs)
    >>> o_p = './reconstruction_test'
    >>> _a.show_reconstructions(coefs, recons, trans, fracs, output_path=o_p)
    >>> current_dir = os.listdir('./')
    >>> for file in sorted(current_dir):
    ...     if 'reconstruction_test' in file:
    ...         print(file)
    reconstruction_test_reconstruction_coefficients.hdf5
    reconstruction_test_reconstruction_coefficients.pdf
    reconstruction_test_reconstructions.hdf5
    reconstruction_test_reconstructions.pdf

    """

    @_decorate_validation
    def validate_input():
        _levels('coefficients',
                (_generic(None, 'explicit collection'),
                 _numeric(None, ('integer', 'floating', 'complex'),
                          shape=(-1, -1))))
        _levels('reconstructions',
                (_generic(None, 'explicit collection', len_=len(coefficients)),
                 _numeric(None, ('integer', 'floating', 'complex'),
                          shape=(-1, -1))))
        _generic('transform', 'string', value_in=_utils.get_transform_names())
        _levels('fractions', (_generic(None, 'explicit collection',
                                       len_=len(coefficients)),
                              _numeric(None, 'floating', range_='[0;1]')))
        _generic('output_path', 'string', ignore_none=True)
        _generic('fig_ext', 'string')

    validate_input()

    disp, axes_extent = _utils.get_function_handle(
        'visualisation', transform)(coefficients[0].shape)

    if len(coefficients) < 4:
        cols = 1
    else:
        cols = 4

    rows = int(np.ceil(len(coefficients) / cols))

    # Coefficients
    coef_imgs = [disp(_mat2vec(
        _visualisation.stretch_image(np.abs(coef), 1.0) + 1e-6))
        for coef in coefficients]
    labels = ['Frac: {:.3f}'.format(frac) for frac in fractions]
    fig_coef = _imsubplot(coef_imgs, rows, x_labels=labels, normalise=True)
    fig_coef.suptitle('Absolute value of transform coefficients used in ' +
                      'reconstructions (log-scale)', fontsize=14)

    # Reconstructions
    fig_recon = _imsubplot(reconstructions, rows, x_labels=labels,
                           normalise=True)
    fig_recon.suptitle('Reconstructions', fontsize=14)

    # Save figures
    if output_path is not None:
        fracs = ['frac' + str(fraction).replace('.', '')
                 for fraction in fractions]
        datasets_coef = {transform: {fraction: coefficients[k]
                                     for k, fraction in enumerate(fracs)}}
        datasets_recon = {transform: {fraction: reconstructions[k]
                                      for k, fraction in enumerate(fracs)}}
        _save_output(output_path, 'reconstruction_coefficients', fig_coef,
                     fig_ext, datasets_coef)
        _save_output(output_path, 'reconstructions', fig_recon, fig_ext,
                     datasets_recon)


def show_sorted_coefficients(img, transforms, output_path=None, fig_ext='pdf'):
    """
    Show the transform coefficient values in sorted order.

    A plot of the sorted coefficient values vs array index number is shown. If
    `output_path` is not None, the resulting figure and data used in the figure
    are saved.

    Parameters
    ----------
    img : ndarray
        The image to show the sorted transform coefficients values for.
    transforms : list or tuple
        The names as strings of the transforms to use.
    output_path : str
        The output path (see notes below) to save the figure and data to (the
        default is None, which implies that the figure and data are not saved).
    fig_ext : str
        The figure extension determining the format of the saved figure (the
        default is 'pdf' which implies that the figure is saved as a PDF).

    Notes
    -----
    The `output_path` is specified as a path to a folder + an optional prefix
    to the file name. The remaining file name is fixed. If e.g, the fixed part
    of the file name was 'plot', then:

    * output_path = '/home/user/' would save the figure under /home/user/ with
      the name plot.pdf.
    * output_path = '/home/user/best' would save the figure under /home/user
      with the name best_plot.pdf.

    In addition to the saved figures, an annotated and chased HDF database with
    the data used to create the figures are also saved. The name of the HDF
    database is the same as for the figure with the exception that the file
    extension is '.hdf5'.

    Examples
    --------
    Save a sorted transform coefficient plot for the DCT and DFT transforms:

    >>> import os, numpy as np
    >>> from magni.imaging.dictionaries import analysis as _a
    >>> img = np.arange(64).astype(np.float).reshape(8, 8)
    >>> transforms = ('DCT', 'DFT')
    >>> output_path = './sorted_test'
    >>> _a.show_sorted_coefficients(img, transforms, output_path=output_path)
    >>> current_dir = os.listdir('./')
    >>> for file in sorted(current_dir):
    ...     if 'sorted_test' in file:
    ...         print(file)
    sorted_test_sorted_coefficients.hdf5
    sorted_test_sorted_coefficients.pdf

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _levels('transforms',
                (_generic(None, 'explicit collection'),
                 _generic(None, 'string',
                          value_in=_utils.get_transform_names())))
        _generic('output_path', 'string', ignore_none=True)
        _generic('fig_ext', 'string')

    validate_input()

    fig, ax = plt.subplots(1, 1)

    datasets = dict()
    for transform in transforms:
        matrix_handle = _utils.get_function_handle('matrix', transform)
        coefficients = matrix_handle(img.shape).conj().T.dot(_mat2vec(img))
        sorted_coefficients = np.sort(np.abs(coefficients), axis=None)[::-1]

        ax.loglog(sorted_coefficients,
                  label='{} coefficients'.format(transform))

        leg = ax.legend()
        leg.get_frame().set_facecolor('1.0')
        ax.set_xlabel('Index value')
        ax.set_ylabel('Absolute coefficient value')
        ax.set_title('Absolute value of coefficients in sorted order')

        datasets[transform] = {'sorted_coefficients': sorted_coefficients}

    # Save figures and data
    if output_path is not None:
        _save_output(output_path, 'sorted_coefficients', fig, fig_ext,
                     datasets)


def show_transform_coefficients(img, transforms, output_path=None,
                                fig_ext='pdf'):
    """
    Show the transform coefficients.

    The transform coefficient of `img` are shown for the `tranforms`. If
    `output_path` is not None, the resulting figure and data used in the figure
    are saved.

    Parameters
    ----------
    img : ndarray
        The image to show the transform coefficients for.
    transforms : list or tuple
        The names as strings of the transforms to use.
    output_path : str
        The output path (see notes below) to save the figure and data to (the
        default is None, which implies that the figure and data are not saved).
    fig_ext : str
        The figure extension determining the format of the saved figure (the
        default is 'pdf' which implies that the figure is saved as a PDF).

    Notes
    -----
    The `output_path` is specified as a path to a folder + an optional prefix
    to the file name. The remaining file name is fixed. If e.g, the fixed part
    of the file name was 'plot', then:

    * output_path = '/home/user/' would save the figure under /home/user/ with
      the name plot.pdf.
    * output_path = '/home/user/best' would save the figure under /home/user
      with the name best_plot.pdf.

    In addition to the saved figures, an annotated and chased HDF database with
    the data used to create the figures are also saved. The name of the HDF
    database is the same as for the figure with the exception that the file
    extension is '.hdf5'.

    Examples
    --------
    Save a figure showing the coefficients for the DCT and DFT transforms:

    >>> import os, numpy as np
    >>> from magni.imaging.dictionaries import analysis as _a
    >>> img = np.arange(64).astype(np.float).reshape(8, 8)
    >>> transforms = ('DCT', 'DFT')
    >>> o_p = './coefficient_test'
    >>> _a.show_transform_coefficients(img, transforms, output_path=o_p)
    >>> current_dir = os.listdir('./')
    >>> for file in sorted(current_dir):
    ...     if 'coefficient_test' in file:
    ...         print(file)
    coefficient_test_transform_coefficients.hdf5
    coefficient_test_transform_coefficients.pdf

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _levels('transforms',
                (_generic(None, 'explicit collection'),
                 _generic(None, 'string',
                          value_in=_utils.get_transform_names())))
        _generic('output_path', 'string', ignore_none=True)
        _generic('fig_ext', 'string')

    validate_input()

    if len(transforms) == 1:
        fig, axes = plt.subplots(1, 1, squeeze=False)
    else:
        rows = int(np.ceil(len(transforms) / 2))
        fig, axes = plt.subplots(rows, 2, squeeze=False)

    axes = axes.flatten()

    datasets = dict()
    for k, transform in enumerate(transforms):
        matrix_handle = _utils.get_function_handle('matrix', transform)
        visual_handle = _utils.get_function_handle('visualisation', transform)

        coefficients = matrix_handle(img.shape).conj().T.dot(_mat2vec(img))
        disp, axes_extent = visual_handle(img.shape)

        scaled_coefficients = _visualisation.stretch_image(
            np.abs(coefficients), 1.0) + 1e-6
        _imshow(disp(scaled_coefficients), ax=axes[k], show_axis='top',
                extent=axes_extent)
        axes[k].set_title(transform, y=1.05)

        datasets[transform] = {'coefficients': coefficients}

    # Save figures
    if output_path is not None:
        _save_output(output_path, 'transform_coefficients', fig, fig_ext,
                     datasets)


def show_transform_quantiles(img, transform, fraction=1.0, area_mask=None,
                             ax=None):
    """
    Show a plot of the quantiles of the transform coefficients.

    The `fraction` of `transform` coefficients holding the most energy for the
    image `img` is considered. The four quantiles within this fraction of
    coefficients are illustrated in the `transform` domain by showing
    coefficients between the different quantiles in different colours. If an
    `area_mask` is specified, only this area in the plot is higlighted whereas
    the rest is darkened.

    Parameters
    ----------
    img : ndarray
        The image to show the transform quantiles for.
    transform : str
        The transform to use.
    fraction : float
        The fraction of coefficients used in the quantiles calculation (the
        default value is 1.0, which implies that all coefficients are used).
    area_mask : ndarray
        Bool array of the same shape as `img` which indicates the area of the
        image to highlight (the default value is None, which implies that no
        particular part of the image is highlighted).
    ax : matplotlib.axes.Axes
        The axes on which the image is displayed (the default is None, which
        implies that a separate figure is created).

    Returns
    -------
    coef_count : dict
        Different counts of coeffcients within the `area_mask` (only returned
        if `area_mask` is not None).

    Notes
    -----
    The ticks on the colorbar shown below the figure are the percentiles of the
    entire set of coefficients corresponding to the quantiles with fraction of
    coefficients. For instance, if fraction is 0.10, then the percentiles are
    92.5, 95.0, 97.5, 100.0, corresponding to the four quantiles within the 10
    percent coefficients holding the most energy.

    The `coef_count` dictionary holds the following keys:

    * C_total : Total number of considered coefficients.
    * Q_potential : Number of potential coefficients within `mask_area`.
    * P_fraction : The fraction of Q_potential to the pixel count in `img`.
    * Q_total : Total number of (considered) coefficients within `mask_area`.
    * Q_fraction : The fraction of Q_total to Q_potential
    * QC_fraction : The fraction of Q_total to C_total.
    * Q0-Q1 : Number of coefficients smaller than the first quantile.
    * Q1-Q2 : Number of coefficients between the first and second quantile.
    * Q2-Q3 : Number of coefficients between the second and third quantile.
    * Q3-Q4 : Number of coefficients between the third and fourth quantile.

    Each of the QX-QY holds a tuple containing two values:

    1. The number of coefficients.
    2. The fraction of the number of coefficients to Q_total.

    Examples
    --------
    For example, show quantiles for a fraction of 0.2 of the DCT coefficients:

    >>> import numpy as np
    >>> from magni.imaging.dictionaries import analysis as _a
    >>> img = np.arange(64).astype(np.float).reshape(8, 8)
    >>> transforms = 'DCT'
    >>> fraction = 0.2
    >>> _a.show_transform_quantiles(img, transform, fraction=fraction)

    """

    @_decorate_validation
    def validate_input():
        _numeric('img', ('integer', 'floating', 'complex'), shape=(-1, -1))
        _generic('transform', 'string', value_in=_utils.get_transform_names())
        _numeric('fraction', 'floating', range_='[0;1]')
        _numeric('area_mask', 'boolean', shape=img.shape, ignore_none=True)
        _generic('ax', mpl.axes.Axes, ignore_none=True)

    @_decorate_validation
    def validate_output():
        _generic('coef_counts', 'mapping',
                 has_keys=('Q_total', 'Q_potential', 'Q0_Q1', 'Q_fraction',
                           'Q3_Q4', 'Q2_Q3', 'Q1_Q2', 'C_total',
                           'QC_fraction'))

    validate_input()

    # Colorbrewer qualitative 5-class Set 1 as colormap
    colours = [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
               (255, 127, 0)]
    norm_colours = [tuple([round(val / 255, 4) for val in colour])
                    for colour in colours[::-1]]
    norm_colours = [norm_colours[0]] * 2 + norm_colours
    quantile_cmap = mpl.colors.ListedColormap(norm_colours)

    # Transform
    transform_matrix = _utils.get_function_handle(
        'matrix', transform)(img.shape)
    all_coefficients = _vec2mat(transform_matrix.conj().T.dot(_mat2vec(img)),
                                img.shape)
    # Force very low values to zero to avoid false visualisations
    all_coefficients[all_coefficients < np.finfo(np.float).eps * 10] = 0

    # Masked coefficients
    sorted_coefficients = np.sort(np.abs(all_coefficients), axis=None)[::-1]
    mask = np.abs(all_coefficients) > sorted_coefficients[
        int(np.round(fraction * all_coefficients.size)) - 1]

    used_coefficients = np.zeros_like(all_coefficients, dtype=np.float)
    used_coefficients[mask] = np.abs(all_coefficients[mask])

    # Quantiles
    q_linspace = np.linspace((1 - fraction) * 100, 100, 5)
    q_percentiles = tuple(q_linspace[1:4])
    quantiles = np.percentile(used_coefficients, q_percentiles)
    disp_coefficients = np.zeros_like(used_coefficients)
    disp_coefficients[(0 < used_coefficients) &
                      (used_coefficients <= quantiles[0])] = 1
    disp_coefficients[(quantiles[0] < used_coefficients) &
                      (used_coefficients <= quantiles[1])] = 2
    disp_coefficients[(quantiles[1] < used_coefficients) &
                      (used_coefficients <= quantiles[2])] = 3
    disp_coefficients[quantiles[2] < used_coefficients] = 4

    # Quantile figure
    disp, axes_extent = _utils.get_function_handle(
        'visualisation', transform)(img.shape)
    if ax is None:
        fig, axes = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
        axes = ax
    im = _imshow(disp(_mat2vec(10**disp_coefficients)), ax=axes,  # anti-log10
                 cmap=quantile_cmap, show_axis='top', interpolation='none',
                 extent=axes_extent)
    divider = _make_axes_locatable(axes)
    c_bar_ax = divider.append_axes('bottom', '5%', pad='3%')
    cbar = fig.colorbar(im, c_bar_ax, orientation='horizontal')
    cbar.solids.set_edgecolor("face")
    cbar.set_ticks([0.85, 1.705, 2.278, 2.85, 3.419, 4.0])
    cbar.set_ticklabels(['Excluded'] + [str(q) for q in q_linspace])

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # Area mask
    if area_mask is not None:
        _imshow(np.ma.array(np.ones_like(disp_coefficients), mask=area_mask),
                ax=axes, cmap='gray', show_axis='top', interpolation='none',
                extent=axes_extent, alpha=0.15)

        # Count of coefficients
        Q_total = np.sum(disp(_mat2vec(10**disp_coefficients))[area_mask] != 0)
        Qs = [np.sum(disp(_mat2vec(10**disp_coefficients))[area_mask] == k)
              for k in [1, 2, 3, 4]]
        coef_counts = {'Q' + str(k-1) + '_Q' + str(k):
                       (Qs[k-1], round(Qs[k-1] / Q_total, 2))
                       for k in [1, 2, 3, 4]}
        coef_counts['Q_total'] = Q_total
        coef_counts['Q_potential'] = np.sum(area_mask)
        coef_counts['Q_fraction'] = round(Q_total /
                                          coef_counts['Q_potential'], 2)
        coef_counts['C_total'] = np.sum(used_coefficients != 0)
        coef_counts['P_fraction'] = round(coef_counts['Q_potential'] /
                                          img.size, 2)
        coef_counts['QC_fraction'] = round(Q_total / coef_counts['C_total'], 2)

        validate_output()

        return coef_counts


def _save_output(output_path, name, fig, fig_ext, datasets):
    """
    Save figure and data output.

    Parameters
    ----------
    output_path : str
        The output_path to save to.
    name : str
        The 'fixed' part of the file name saved to.
    fig : matplotlib.figure.Figure
        The figure instance to save.
    fig_ext : str
        The file extension to use for the saved figure.
    datasets : dict
        The dict of dicts for datasets to save in a HDF database.

    """

    @_decorate_validation
    def validate_input():
        _generic('output_path', 'string')
        _generic('name', 'string')
        _generic('fig', mpl.figure.Figure)
        _generic('fig_ext', 'string')
        _levels('datasets', (_generic(None, 'mapping'),
                             _generic(None, 'mapping')))

    validate_input()

    if output_path[-1] == os.sep:
        path = output_path
        prefix = ''

    else:
        path, prefix, no_ext = _split_path(output_path)
        prefix = prefix + '_'

    fig.savefig(path + prefix + name + os.path.extsep + fig_ext)

    db_path = path + prefix + name + '.hdf5'
    _io.create_database(db_path)
    with _File(db_path, mode='a') as h5file:
        data_group = h5file.create_group('/', 'data', __name__ + ': ' + name)
        for dataset in datasets:
            set_group = h5file.create_group(data_group, dataset, dataset)
            for array in datasets[dataset]:
                h5file.create_array(set_group, array, datasets[dataset][array])
