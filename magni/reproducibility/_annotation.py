"""
..
    Copyright (c) 2014-2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions that may be used to annotate data.

Routine Listings
----------------
get_conda_info()
    Function that returns information about an Anaconda install.
get_datetime()
    Function that returns information about the current date and time.
get_git_revision(git_root_dir=None)
    Function that returns information about the current git revision.
get_file_hashes(path, blocksize=2**30)
    Function that returns the md5 and sha256 checksums of a file.
get_magni_config()
    Function that returns information about the current configuration of Magni.
get_magni_info()
    Function that returns genral information about Magni.
get_platform_info()
    Function that returns information about the platform used to run the code.

Notes
-----
The returned annotations are any nested level of dicts of dicts of strings.

"""

from __future__ import division
import contextlib
import datetime
import hashlib
import json
import os
import pkgutil
import platform
import pydoc
import re
import subprocess
import sys

import magni as _magni
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_numeric as _numeric

__all__ = ['get_conda_info', 'get_datetime', 'get_git_revision',
           'get_file_hashes', 'get_magni_config', 'get_magni_info',
           'get_platform_info']


def get_conda_info():
    """
    Return a dictionary contianing information from Conda.

    `Conda <http://conda.pydata.org/>`_ is the package manager for the Anaconda
    scientific Python distribution. This function will return various
    information about the Anaconda installation on the system by querying the
    Conda package database.

    .. warning::

        THIS IS HIGHLY EXPERIMENTAL AND MAY BREAK WITHOUT FURTHER NOTICE.

    .. note::

        Only infomation about the conda root environment is captured.

    Returns
    -------
    conda_info : dict
        Various information from conda (see notes below for further details).

    Notes
    -----
    If the Python intepreter is unable to locate and import the conda package,
    an empty dicionary is returned.

    The returned dictionary contains the same infomation that is returned by
    "conda info" in addition to an overview of the linked modules in the
    Anaconda installation. Specifically, the returned dictionary has the
    following keys:

    * platform
    * conda_version
    * root_prefix
    * default_prefix
    * envs_dirs
    * package_cache
    * channels
    * config_file
    * is_foreign_system
    * linked_modules
    * env_export

    Additionally, the returned dictionary has a key named *status*, which can
    have either of the following values:

    * 'Succeeded' (Everything seems to be OK)
    * 'Failed' (Import of conda failed - nothing else is returned)

    If "conda-env" is installed on the system, the `env_export` essentially
    holds the infomation from "conda env export -n root" as a dictionary. The
    information provided by this key partially overlaps with the infomation in
    the `linked_modules` and `modules_info` keys.

    """

    try:
        import conda
        import conda.config
        import conda.install
    except ImportError:
        return {'status': 'Failed'}

    try:
        import conda_env.env
        has_conda_env = True
    except ImportError:
        has_conda_env = False

    # Ugly hack to silence the
    # "Using Anaconda Cloud api site https://api.anaconda.org"
    # message being sent to stderr by the binstar/anaconda client.
    with open(os.devnull, 'wb') as null:
        with _catch_stderr(null):
            conda_channel_urls = conda.config.get_channel_urls()

    conda_info = {'platform': conda.config.subdir,
                  'conda_version': conda.__version__,
                  'root_prefix': conda.config.root_dir,
                  'default_prefix': conda.config.default_prefix,
                  'envs_dirs': json.dumps(conda.config.envs_dirs),
                  'package_cache': json.dumps(conda.config.pkgs_dirs),
                  'channels': json.dumps(conda_channel_urls),
                  'config_file': json.dumps(conda.config.rc_path),
                  'is_foreign_system': json.dumps(bool(conda.config.foreign)),
                  'linked_modules': sorted(
                      conda.install.linked(conda.config.root_dir))}

    modules_info = {module:
                    conda.install.is_linked(conda_info['root_prefix'], module)
                    for module in conda_info['linked_modules']}
    conda_info['modules_info'] = json.dumps(modules_info)
    conda_info['linked_modules'] = json.dumps(conda_info['linked_modules'])

    if has_conda_env:
        conda_info['env_export'] = json.dumps(
            conda_env.env.from_environment(
                'root', conda.config.root_dir).to_dict())
    else:
        conda_info['env_export'] = 'Failed: conda-env not available'

    conda_info['status'] = 'Succeeded'

    return conda_info


def get_datetime():
    """
    Return a dictionary holding the current date and time.

    Returns
    -------
    date_time : dict
        The dictionary holding the current date and time.

    Notes
    -----
    The returned dictionary has the following keys:

    * today (date and time including timezone offset)
    * utcnow (UTC date and time)
    * pretty_utc (UTC date and time formatted according to current locale)
    * status

    The status entry informs about the success of the pretty_utc formatting.
    It has one of the follwing values:

    * Succeeded (Everything seems OK)
    * Failed (It was not possible to format the time)

    """

    date_time = {'today': repr(datetime.datetime.today()),
                 'utcnow': datetime.datetime.utcnow(),
                 'pretty_utc': '',
                 'status': 'Succeeded'}

    try:
        date_time['pretty_utc'] = datetime.datetime.strftime(
            date_time['utcnow'], '%c')

    except ValueError:
        date_time['status'] = 'Failed'

    date_time['utcnow'] = repr(date_time['utcnow'])

    return date_time


def get_git_revision(git_root_dir=None):
    """
    Return a dictionary containing information about the current git revision.

    Parameters
    ----------
    git_root_dir : str
        The path to the git root directory to get git revision for (the default
        is None, which implies that the git revision of the `magni` directory
        is returned).

    Returns
    -------
    git_revision : dict
        Information about the current git revision.

    Notes
    -----
    If the git revision extract succeeded, the returned dictionary has the
    following keys:

    * status (with value 'Succeeded')
    * tag (output of "git describe")
    * branch (output of "git describe --all")
    * remote (output of "git remote -v")

    If the git revision extract failed, the returned dictionary has the
    following keys:

    * status (with value 'Failed')
    * returncode (returncode from failing git command)
    * output (output from failing git command)

    The "git" commands are run in the git root directory.

    """

    @_decorate_validation
    def validate_input():
        _generic('git_root_dir', 'string', ignore_none=True)

    validate_input()

    cur_dir = os.getcwd()
    if git_root_dir is not None:
        try:
            os.chdir(git_root_dir)
        except (IOError, OSError):
            raise OSError(
                'The git_root_dir directory "{!r}".format(git_root_dir)' +
                'does not exist')
    else:
        os.chdir(os.path.split(_magni.__path__[0])[0])

    try:
        git_revision = {
            'tag': str(subprocess.check_output(
                ['git', 'describe'],
                stderr=subprocess.STDOUT)[:-1].decode()),
            'branch': str(subprocess.check_output(
                ['git', 'describe', '--all'],
                stderr=subprocess.STDOUT)[:-1].decode()),
            'remote': str(subprocess.check_output(
                ['git', 'remote', '-v'],
                stderr=subprocess.STDOUT)[:-1].decode()),
            'status': 'Succeeded'}

    except subprocess.CalledProcessError as e:
        try:
            e_output = e.output.decode()
        except AttributeError:
            e_output = e.output

        git_revision = {'status': 'Failed: CallProcessError',
                        'returncode': e.returncode,
                        'output': e_output}

    except OSError as e:
        try:
            e_strerror = e.strerror.decode()
        except AttributeError:
            e_strerror = e.strerror

        git_revision = {'status': 'Failed: OSError',
                        'errno': e.errno,
                        'strrror': e_strerror}

    os.chdir(cur_dir)

    return git_revision


def get_file_hashes(path, blocksize=2**30):
    """
    Return a dictionary with md5 and sha256 checksums of a file.

    Parameters
    ----------
    path : str
        The path to the file to checksum.
    blocksize : int
        The chunksize (in bytes) to read from the file one at a time.

    Returns
    -------
    file_hashes : dict
        The dictionary holding the md5 and sha256 hexdigests of the file.

    """

    @_decorate_validation
    def validate_input():
        _generic('path', 'string')
        _numeric('blocksize', 'integer', range_='[1;inf)')

    validate_input()

    md5sum = hashlib.md5()
    sha256sum = hashlib.sha256()

    with open(path, mode='rb') as f_handle:
        buf = f_handle.read(blocksize)
        while buf != ''.encode():
            md5sum.update(buf)
            sha256sum.update(buf)
            buf = f_handle.read(blocksize)

    file_hashes = {'md5sum': md5sum.hexdigest(),
                   'sha256sum': sha256sum.hexdigest()}

    return file_hashes


def get_magni_config():
    """
    Return a dictionary holding the current configuration of Magni.

    Returns
    -------
    magni_config : dict
        The dictionary holding the current configuration of Magni.

    Notes
    -----
    The returned dictionary has a key for each of the `config` modules in Magni
    and its subpackages. The value of a given key is a dictionary with the
    current configuration of the corresponding `config` module. Furthermore,
    the returned dictionary has a status key, which can have either of the
    following values:

    * Succeeded (The entire configuration was extracted)
    * Failed (It was not possible to get information from one or more modules)

    """

    packages = pkgutil.walk_packages(path=_magni.__path__,
                                     prefix=_magni.__name__ + '.')

    magni_config = dict()
    try:
        for importer, modname, ispkg in packages:
            if modname[-8:] == '._config':
                try:
                    settings = dict(eval('_' + modname).configger.items())
                except AttributeError:
                    # Skip base Configgers, e.g. cs.reconcstruction.config
                    pass
                else:
                    for setting in settings:
                        if not isinstance(settings[setting], str):
                            settings[setting] = repr(settings[setting])

                    magni_config[modname[:-7] + modname[-6:]] = settings

        magni_config['status'] = 'Succeeded'

    except AttributeError:
        magni_config['status'] = 'Failed'

    return magni_config


def get_magni_info():
    """
    Return a string representation of the output of help(magni).

    Returns
    -------
    magni_info : dict
        Information about magni.

    Notes
    -----
    The returned dictionary has a single key:

    * help_magni (a string representation of help(magni))

    """

    magni_info = pydoc.render_doc(_magni)
    magni_info, subs = re.subn(r'\x08([A-Z]|[a-z]|_)?', '', magni_info)

    return {'help_magni': magni_info}


def get_platform_info():
    """
    Return a dictionary containing information about the system platform.

    Returns
    -------
    platform_info : dict
        Various information about the system platform.

    See Also
    --------
    platform : The Python module used to query information about the system.

    Notes
    -----
    The returned dictionary has the following keys:

    * system
    * node
    * release
    * version
    * processor
    * python
    * libc
    * linux
    * mac_os
    * win32
    * status

    The linux/mac_os/win32 entries are "empty" if they are not applicable.

    If the processor information returned by `platform` is "empty", a query of
    `lscpu` is attempted in order to provide the necessary information.

    The status entry informs about the success of the queries. It has one of
    the follwing values:

    * 'All OK' (everything seems to be OK)
    * 'Used lscpu in processor query' (`lscpu` was used)
    * 'Processor query failed' (failed to get processor information)

    """

    platform_info = {'system': json.dumps(platform.system()),
                     'node': json.dumps(platform.node()),
                     'release': json.dumps(platform.release()),
                     'version': json.dumps(platform.version()),
                     'machine': json.dumps(platform.machine()),
                     'processor': json.dumps(platform.processor()),
                     'python': json.dumps(platform.python_version()),
                     'libc': json.dumps(platform.libc_ver()),
                     'linux': json.dumps(platform.linux_distribution()),
                     'mac_os': json.dumps(platform.mac_ver()),
                     'win32': json.dumps(platform.win32_ver()),
                     'status': 'All OK'}

    if platform_info['processor'] == '':
        try:
            platform_info['processor'] = str(
                subprocess.check_output(['lscpu']).decode())
            platform_info['status'] = 'Used lscpu in processor query'

        except (subprocess.CalledProcessError, OSError):
            platform_info['status'] = 'Processor query failed'

    return platform_info


@contextlib.contextmanager
def _catch_stderr(catcher):
    """
    A context manager for catching stderr.

    Parameters
    ----------
    catcher : file-like
        The file-like object which is to catch stderr.

    """

    _stderr = sys.stderr
    sys.stderr = catcher
    yield

    sys.stderr = _stderr
