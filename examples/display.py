import numpy as np
np.set_printoptions(linewidth=120)

from magni.utils.validation import (
    decorate_validation as _decorate_validation,
    validate as _validate)


@_decorate_validation
def _validate_print_table(columns, headers, bars):
    """
    Validate the `print_table` function.

    See Also
    --------
    print_table : The validated function.
    magni.utils.validation.validate : Validation.

    """

    lst = (list, tuple)
    _validate(columns, 'columns', [{'type_in': lst}, {'type_in': lst}])
    _validate(columns, 'columns', [{}, {'len': len(columns[0])}])
    cols = len(columns)
    _validate(headers, 'headers', [{'type_in': lst, 'len': cols},
                                   {'type': str}], True)

    if type(bars) is not str:
        _validate(bars, 'bars', [{'type_in': lst, 'len': cols - 1},
                                 {'type': str}])


def print_table(columns, headers=None, bars='|'):
    """
    Print a nicely formatted table to the screen.

    A top rule is printed followed by an optional header line and a mid rule
    before the actual data is printed followed by an end rule. It is optional
    whether vertical bars between the columns should be present.

    Parameters
    ----------
    columns : list or tuple
        The columns that should be printed.
    headers : list or tuple
        The headers describing the columns (optional).
    bars : str or list or tuple
        The vertical bars used to separate the columns (optional).

    Notes
    -----
    The `bars` parameter can either be a string containing the vertical bar
    symbol(s) used to separate all columns or a list of strings containing the
    vertical bars symbol(s) used to separate the individual columns. If a list
    is supplied its length must be len(columns) - 1.

    """

    _validate_print_table(columns, headers, bars)

    if headers is not None:
        hr_index = 0
        columns = list(zip(headers, *zip(*columns)))
    else:
        hr_index = -1

    widths = [max([len(str(val)) for val in col]) + 3 for col in columns]
    widths[-1] = widths[-1] - 3

    if type(bars) is str:
        bars = [bars] * (len(widths) - 1)

    seps = []
    hr = widths

    for bar in bars:
        seps = seps + [' ' + bar + ' ']
        hr = hr + [len(bar) + 2]

    seps = seps + ['']
    hr = (sum(hr) + 6) * '-'

    print('\n')
    print(hr)

    for i, row in enumerate(zip(*columns)):
        line = '   '

        for field, width, sep in zip(row, widths, seps):
            line = line + str(field).ljust(width) + sep

        print(line)

        if i == hr_index:
            print(hr)

    print(hr)
    print('\n')
