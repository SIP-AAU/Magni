"""..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module for wrapping the Magni IPython Notebook examples.

**This module is based on the "ipnbdoctest.py" script by Benjamin
Ragan-Kelley (MinRK)**, source: https://gist.github.com/minrk/2620735.

This assumes comparison of IPython Notebooks in nbformat.v3

"""

from __future__ import division, print_function
import base64
import contextlib
from datetime import datetime
import os
import shutil
import subprocess
import unittest
import types
import warnings
try:
    from Queue import Empty  # Python 2
except ImportError:
    from queue import Empty  # Python 3
try:
    from StringIO import StringIO as BytesIO  # Python 2
except ImportError:
    from io import BytesIO  # Python 3

import numpy as np
from pkg_resources import parse_version
import scipy.misc

import magni

# The great "support IPython 2, 3, 4" strat begins
try:
    import jupyter
except ImportError:
    jupyter_era = False
else:
    jupyter_era = True

if jupyter_era:
    # Jupyter / IPython 4.x
    from jupyter_client import KernelManager
    from nbformat import reads, NotebookNode

    def mod_reads(file_):
        return reads(file_, 3)  # Read notebooks as v3

else:
    from IPython.kernel import KernelManager
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            # IPython 2.x
            from IPython.nbformat.current import reads, NotebookNode

            def mod_reads(file_):
                return reads(file_, 'json')

        except UserWarning:
            # IPython 3.x
            from IPython.nbformat import reads, NotebookNode

            def mod_reads(file_):
                return reads(file_, 3)  # Read notebooks as v3

# End of the great "support IPython 2, 3, 4" strat

# Test for freetype library version
try:
    if parse_version(
            subprocess.check_output(
                ['freetype-config', '--ftversion']).decode().strip()
            ) <= parse_version('2.5.2'):
        _skip_display_data_tests = False
    else:
        _skip_display_data_tests = True
except OSError:
    _skip_display_data_tests = True

if _skip_display_data_tests:
    warnings.warn('Skipping display data ipynb tests.', RuntimeWarning)


class _Meta(type):
    """
    Identification of IPython Notebook examples and construction of test class.

    """

    def __new__(class_, name, bases, attrs):
        path = magni.__path__[0].rsplit(os.sep, 1)[0]
        path = path + os.path.sep + 'examples' + os.path.sep

        for filename in os.listdir(path):
            if filename[-6:] == '.ipynb':
                name = 'test_' + filename[:-6].replace('-', '_')
                func = attrs['_run_example']
                func = types.FunctionType(func.__code__, func.__globals__,
                                          name, (path + filename,))
                func.__doc__ = func.__doc__.format(filename)
                attrs[name] = func

        return type.__new__(class_, name, bases, attrs)


# For python 2 and 3 compatibility
class _Hack(_Meta):
    def __new__(class_, name, bases, attrs):
        return _Meta(name, (unittest.TestCase,), attrs)


_TestCase = type.__new__(_Hack, 'temp', (), {})


class TestIPythonExamples(_TestCase):
    """
    Test of Ipython Notebook examples for equality of output to reference.

    """

    def setUp(self):
        """
        Identify IPython Notebook examples to run.

        """

        path = magni.__path__[0].rsplit(os.sep, 1)[0]
        path = path + os.path.sep + 'examples' + os.path.sep
        files_to_copy = ['example.mi', 'data.hdf5', 'display.py']

        for cfile in files_to_copy:
            shutil.copy(os.path.join(path, cfile), '.')

    def _run_example(self, ipynb):
        """
        Test of {} Magni IPython Notebook example.

        """

        with open(ipynb) as f_ipynb:
            notebook = mod_reads(f_ipynb.read())

        notebook_result = _check_ipynb(notebook)
        passed, successes, failures, errors, report = notebook_result

        self.assertTrue(passed, msg=report)
        error_msg = ('Magni IPython Notebook example status:\n' +
                     'Successes: {}, Failures: {}, Errors: {}').format(
                         successes, failures, errors)
        self.assertEqual(errors + failures, 0, msg=error_msg)


def _check_ipynb(notebook):
    """
    Check an IPython Notebook for matching input and output.

    Each cell input in the `notebook` is executed and the result is compared
    to the cell output saved in the `notebook`.

    Parameters
    ----------
    notebook : IPython.nbformat.current.NotebookNode
        The notebook to check for matching input and output.

    Returns
    -------
    passed : Bool
        The indicator of a successful check (or not).
    sucessess : int
        The number of cell outputs that matched.
    failures : int
        The number of cell outputs that failed to match.
    errors : int
        The number of cell executions that resulted in errors.
    report : str
        The report detailing possible failures and errors.

    """

    kernel_manager = KernelManager()
    kernel_manager.start_kernel()
    kernel_client = kernel_manager.client()
    kernel_client.start_channels()

    try:
        # IPython 3.x
        kernel_client.wait_for_ready()
        iopub = kernel_client
        shell = kernel_client
    except AttributeError:
        # Ipython 2.x
        # Based on https://github.com/paulgb/runipy/pull/49/files
        iopub = kernel_client.iopub_channel
        shell = kernel_client.shell_channel
        shell.get_shell_msg = shell.get_msg
        iopub.get_iopub_msg = iopub.get_msg

    successes = 0
    failures = 0
    errors = 0

    report = ''
    for worksheet in notebook.worksheets:
        for cell in worksheet.cells:
            if cell.cell_type == 'code':
                try:
                    test_results = _execute_cell(cell, shell, iopub)
                except RuntimeError as e:
                    report += ('{!s} in cell number: {}'
                               .format(e, cell.prompt_number))
                    errors += 1
                    break

                identical_output = all(
                    [_compare_cell_output(test_result, reference)
                     for test_result, reference in
                     zip(test_results, cell.outputs)])

                if identical_output:
                    successes += 1
                else:
                    failures += 1

                    try:
                        str_test_results = [
                            '(for out {})\n'.format(k) + '\n'.join(
                                [' : '.join([str(key), str(val)])
                                 for key, val in t.items()
                                 if key not in ('metadata', 'png')]
                            ) for k, t in enumerate(test_results)]
                        str_cell_outputs = [
                            '(for out {})\n'.format(k) + '\n'.join(
                                [' : '.join([str(key), str(val)])
                                 for key, val in t.items()
                                 if key not in ('metadata', 'png')]
                            ) for k, t in enumerate(cell.outputs)]
                    except TypeError as e:
                        report += 'TypeError in ipynb_examples test\n\n'
                        for entry in cell.outputs:
                            if 'traceback' in entry.keys():
                                for item in entry['traceback']:
                                    report += str(item) + '\n'
                    else:
                        report += '\n' * 2 + '~' * 40
                        report += (
                            '\nFailure in {}:{}\nGot: {}\n\n\nExpected: {}'
                        ).format(notebook.metadata.name,
                                 cell.prompt_number,
                                 '\n'.join(str_test_results),
                                 '\n'.join(str_cell_outputs))

    kernel_client.stop_channels()
    kernel_manager.shutdown_kernel()

    passed = not (failures or errors)

    return passed, successes, failures, errors, report


def _compare_cell_output(test_result, reference):
    """
    Compare a cell test output to a reference output.

    Parameters
    ----------
    test_results : IPython.nbformat.current.NotebookNode
        The cell test result that must be compared to the reference.
    reference : IPython.nbformat.current.NotebookNode
        The reference cell output to compare to.

    Returns
    -------
    comparison_result : bool
        The indicator of equality between the test output and the reference.

    """

    skip_compare = ['traceback', 'latex', 'prompt_number']

    if _skip_display_data_tests:
        # Skip graphics comparison
        skip_compare.append('png')

    if test_result['output_type'] == 'display_data':
        # Prevent comparison of matplotlib figure instance memory addresses
        skip_compare.append('text')
        skip_compare.append('metadata')

    for key in reference:

        if key not in test_result:
            raise Exception(str(reference) + '!!!!!' + str(test_result))
            return False
        elif key not in skip_compare:
            if key == 'text':
                if test_result[key].strip() != reference[key].strip():
                    return False
            elif key == 'png':
                reference_img = reference[key]
                test_img = test_result[key]
                if not _compare_images(reference_img, test_img):
                    return False
            else:
                if test_result[key] != reference[key]:
                    return False

    return True


def _compare_images(reference_img, test_img):
    """
    Compare reference and test image to determine if they depict the same.

    Two images are considered to depict the same unless:

    - The image shapes differ
    - The number of differences in non-transparant pixel values which are not
      (likely to be) part of the image border exceeds 2.

    Parameters
    ----------
    reference_img : str
        The base64 encoded reference image.
    test_img : str
        The base64 encoded test image.

    Returns
    -------
    comparison_result : bool
        The idenfifier of a positive match, i.e. True if images are the same.

    """

    ref_png = base64.b64decode(reference_img)
    ref_ndarray = scipy.misc.imread(BytesIO(ref_png))
    cmp_png = base64.b64decode(test_img)
    cmp_ndarray = scipy.misc.imread(BytesIO(cmp_png))

    # check shape of images
    if cmp_ndarray.shape != ref_ndarray.shape:
        print('Image shapes differ')
        return False

    # mask of channels in pixels with different values
    diff = cmp_ndarray != ref_ndarray
    # mask of pixels with different values
    diff = np.any(diff, axis=2)
    # mask of non-transparent pixels with different values
    diff = diff * np.bool_(ref_ndarray[:, :, 3])

    # check if all non-transparent pixels match
    if diff.sum() == 0:
        # Accept difference in tranparent pixels
        return True

    # The rest is all about checking if (it is likely to be) only the image
    # border that has changed. The border may render differently across
    # matplotlib versions.

    # mask of black pixels
    mask = ((ref_ndarray[:, :, 0] == 0) *
            (ref_ndarray[:, :, 1] == 0) *
            (ref_ndarray[:, :, 2] == 0) *
            (ref_ndarray[:, :, 3] == 255))

    # lookup table of the top most connected black pixel of the
    # looked up pixel
    C_N = np.zeros(mask.shape, dtype=np.int16)

    for i in range(0, mask.shape[0] - 1):
        C_N[i + 1, :] = np.logical_not(mask[i, :]) * i + mask[i, :] * C_N[i, :]

    # lookup table of the right most connected black pixel of the
    # looked up pixel
    C_E = np.zeros(mask.shape, dtype=np.int16)

    for i in range(mask.shape[1] - 1, 0, -1):
        C_E[:, i - 1] = np.logical_not(mask[:, i]) * i + mask[:, i] * C_E[:, i]

    # lookup table of the bottom most connected black pixel of the
    # looked up pixel
    C_S = np.zeros(mask.shape, dtype=np.int16)

    for i in range(mask.shape[0] - 1, 0, -1):
        C_S[i - 1, :] = np.logical_not(mask[i, :]) * i + mask[i, :] * C_S[i, :]

    # lookup table of the left most connected black pixel of the
    # looked up pixel
    C_W = np.zeros(mask.shape, dtype=np.int16)

    for i in range(0, mask.shape[1] - 1):
        C_W[:, i + 1] = np.logical_not(mask[:, i]) * i + mask[:, i] * C_W[:, i]

    # coordinates of non-transparent pixels with different values
    points = np.nonzero(diff)
    points = np.int32(points + (np.zeros(points[0].shape),)).T

    # loop over non-transparent pixels with different values
    for i, point in enumerate(points):
        y, x = point[:2]

        # find other non-transparent pixels with different values
        # ... with the same y-coordinate
        matches_y = np.nonzero(points[:, 0] == y)[0]
        # ... with an x-coordinate at least 10 pixels away
        matches_y = matches_y[np.abs(points[matches_y, 1] - x) > 10]
        # ... which is connected by black pixels
        matches_y = matches_y[
            (points[matches_y, 1] >= C_W[y, x]) *
            (points[matches_y, 1] <= C_E[y, x])]

        # find other non-transparent pixels with different values
        # ... with the same x-coordinate
        matches_x = np.nonzero(points[:, 1] == x)[0]
        # ... with a y-coordinate at least 10 pixels away
        matches_x = matches_x[np.abs(points[matches_x, 0] - y) > 10]
        # ... which is connected by black pixels
        matches_x = matches_x[
            (points[matches_x, 0] >= C_N[y, x]) *
            (points[matches_x, 0] <= C_S[y, x])]

        if len(matches_y) + len(matches_x) == 0:
            # this pixel cannot be the corner of a box
            break

        for j in matches_y:
            for k in matches_x:
                # loop over combinations of possible boxes
                y_test = points[k, 0]
                x_test = points[j, 1]

                if not C_W[y_test, x] <= x_test <= C_E[y_test, x]:
                    # one horizontal line of the box isn't black
                    continue

                if not C_N[y, x_test] <= y_test <= C_S[y, x_test]:
                    # one vertical line of the box isn't black
                    continue

                # the box is a box and the corners are flagged
                points[i, 2] = points[j, 2] = points[k, 2] = 1

    if points.shape[0] - np.sum(points[:, 2]) > 2:
        print('The images differ by {} pixels'.format(
            points.shape[0] - np.sum(points[:, 2])))

        # Save images and their difference for visual inspection
        fail_txt = 'Notebook test fail '
        utcnow = datetime.utcnow
        scipy.misc.imsave(fail_txt + str(utcnow()) + 'r' + '.png', ref_ndarray)
        scipy.misc.imsave(fail_txt + str(utcnow()) + 't' + '.png', cmp_ndarray)
        img_diff = cmp_ndarray - ref_ndarray
        scipy.misc.imsave(fail_txt + str(utcnow()) + 'd' + '.png', img_diff)

        return False

    return True


def _execute_cell(cell, shell, iopub, timeout=300):
    """
    Execute an IPython Notebook Cell and return the cell output.

    Parameters
    ----------
    cell : IPython.nbformat.current.NotebookNode
        The IPython Notebook cell to execute.
    shell : IPython.kernel.blocking.channels.BlockingShellChannel
        The shell channel which the cell is submitted to for execution.
    iopub : IPython.kernel.blocking.channels.BlockingIOPubChannel
        The iopub channel used to retrieve the result of the execution.
    timeout : int
        The number of seconds to wait for the execution to finish before giving
        up.

    Returns
    -------
    cell_outputs : list
        The list of NotebookNodes holding the result of the execution.

    """

    # Execute input
    shell.execute(cell.input)
    exe_result = shell.get_shell_msg(timeout=timeout)
    if exe_result['content']['status'] == 'error':
        raise RuntimeError('Failed to execute cell due to error: {!r}'.format(
            str(exe_result['content']['evalue'])))

    cell_outputs = list()

    # Poll for iopub messages until no more messages are available
    while True:
        try:
            msg = iopub.get_iopub_msg(timeout=0.5)
        except Empty:
            break

        msg_type = msg['msg_type']
        if msg_type in ('status', 'pyin', 'execute_input', 'execute_result'):
            continue

        content = msg['content']
        node = NotebookNode(output_type=msg_type)

        if msg_type == 'stream':
            node.stream = content['name']
            if 'text' in content:
                # v4 notebook format
                node.text = content['text']
            else:
                # v3 notebook format
                node.text = content['data']
        elif msg_type in ('display_data', 'pyout'):
            node['metadata'] = content['metadata']
            for mime, data in content['data'].items():
                attr = mime.split('/')[-1].lower()
                attr = attr.replace('+xml', '').replace('plain', 'text')
                setattr(node, attr, data)
            if msg_type == 'pyout':
                node.prompt_number = content['execution_count']
        elif msg_type == 'pyerr':
            node.ename = content['ename']
            node.evalue = content['evalue']
            node.traceback = content['traceback']
        else:
            raise RuntimeError('Unhandled iopub message of type: {}'.format(
                msg_type))

        cell_outputs.append(node)

    return cell_outputs
