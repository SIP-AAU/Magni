"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing functions that may be used to annotate data.

Routine Listings
----------------
get_conda_info()
    Function that returns information about a Continnum Anaconda install.
get_datetime()
    Function that returns information about the current date and time.
get_git_revision()
    Function that returns information about the `magni` git revision.
get_magni_config()
    Function that returns information about the current configuration of Magni.
get_magni_info()
    Function that returns genral information about Magni.
get_platform_info()
    Function that returns information about the platform used to run the code.

Notes
-----
The return annotations are any nested level of dicts of dicts of strings.

"""

from __future__ import division
import datetime
import json
import os
import pkgutil
import platform
import pydoc
import re
import subprocess

import magni


def get_conda_info():
    """
    Return a dictionary contianing information from Conda.

    Conda is the package manager for the `Continuum Anaconda
    <https://store.continuum.io/cshop/anaconda/>`_ scientific Python
    distribution. This function will return various information about the
    Anaconda installation on the system by quering the Conda package database.

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

    Additionally, the returned dictionary has a key named *status*, which can
    have either of the following values:

    * 'Succeeded' (Everything seems to be OK)
    * 'Failed' (Import of conda failed - nothing else is returned)

    """

    try:
        import conda
        import conda.config
        import conda.install
    except ImportError:
        return {'status': 'Failed'}

    conda_info = {'platform': conda.config.subdir,
                  'conda_version': conda.__version__,
                  'root_prefix': conda.config.root_dir,
                  'default_prefix': conda.config.default_prefix,
                  'envs_dirs': json.dumps(conda.config.envs_dirs),
                  'package_cache': json.dumps(conda.config.pkgs_dirs),
                  'channels': json.dumps(conda.config.get_channel_urls()),
                  'config_file': json.dumps(conda.config.rc_path),
                  'is_foreign_system': json.dumps(bool(conda.config.foreign)),
                  'linked_modules': sorted(
                      conda.install.linked(conda.config.root_dir))}

    modules_info = {module:
                    conda.install.is_linked(conda_info['root_prefix'], module)
                    for module in conda_info['linked_modules']}
    conda_info['modules_info'] = modules_info
    conda_info['linked_modules'] = json.dumps(conda_info['linked_modules'])

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


def get_git_revision():
    """
    Return a dictionary containing information about the current git revision.

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

    If the git revision extract failed, the returned dictionary has the
    following keys:

    * status (with value 'Failed')
    * returncode (returncode from failing git command)
    * output (output from failing git command)

    The "git describe" commands are run in the directory in which `magni` is
    loaded from.

    """

    cur_dir = os.getcwd()
    os.chdir(os.path.split(magni.__path__[0])[0])

    try:
        git_revision = {'tag': str(subprocess.check_output(
            ['git', 'describe'], stderr=subprocess.STDOUT)[:-1].decode()),
            'branch': str(subprocess.check_output(
                ['git', 'describe', '--all'],
                stderr=subprocess.STDOUT)[:-1].decode()),
            'status': 'Succeeded'}

    except subprocess.CalledProcessError as e:
        git_revision = {'status': 'Failed: CallProcessError',
                        'returncode': e.returncode,
                        'output': e.output}

    except OSError as e:
        git_revision = {'status': 'Failed: OSError',
                        'errno': e.errno,
                        'strrror': e.strerror}

    os.chdir(cur_dir)

    return git_revision


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

    packages = pkgutil.walk_packages(path=magni.__path__,
                                     prefix=magni.__name__ + '.')

    magni_config = {'status': 'Succeeded'}
    try:
        for importer, modname, ispkg in packages:
            if modname[-7:] == '.config' and modname != 'magni.utils.config':
                settings = eval(modname + '.get()')
                for setting in settings:
                    if not isinstance(settings[setting], str):
                        settings[setting] = repr(settings[setting])

                magni_config[modname] = settings

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

    magni_info = pydoc.render_doc(magni)
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
