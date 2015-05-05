"""
..
    Copyright (c) 2014-2015, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

A setup.py script for Magni.

More about python packaging:
https://python-packaging-user-guide.readthedocs.org
https://pythonhosted.org/setuptools/setuptools.html#developer-s-guide
https://github.com/pypa/sampleproject

"""

import io
import re

from setuptools import setup, find_packages


def get_version(file_path):
    """
    Return the project version tag.

    Parameters
    ----------
    file_path : str
        The path to the project __init__.py file.

    Returns
    -------
    version : str
        The project version tag.

    """

    with open(file_path, mode='r') as init_file:
        init_cont = init_file.read()

    # PEP 440: [N:]N(.N)*[{a|b|c|rc}N][.postN][.devN]
    ver_spec = (r'__version__ = ["\'](\d+:)?\d+(\.\d+){1,2}' +
                r'((a|b|c|rc)\d+)?(\.post\d+)?(\.dev\d+)?["\']')
    ver_match = re.search(ver_spec, init_cont)

    if not ver_match:
        raise RuntimeError('Unable to find (valid) version specifier in {0!r}'
                           .format(file_path))

    return ver_match.group(0)[15:-1]


with io.open('README.rst', mode='r', encoding='utf-8') as desc_file:
    long_description = desc_file.read()


setup(
    name='magni',
    version=get_version('magni/__init__.py'),
    description=('A Python Package for Compressive Sampling ' +
                 'and Reconstruction of Atomic Force Microscopy Images'),
    long_description=long_description,
    url='https://github.com/SIP-AAU/Magni',
    download_url='https://pypi.python.org/pypi/magni',
    author='Magni Developers',
    author_email='magni@es.aau.dk',

    license='BSD 2-Clause',

    # Classifiers
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization'],

    # Keywords
    keywords=('Atomic Force Microscopy; Compressive Sensing; Python; ' +
              'Image Reconstruction; Reproducible Research'),

    # Project packages
    packages=find_packages(),

    # Required runtime dependencies
    install_requires=['numpy>=1.8',
                      'scipy>=0.13',
                      'tables>=3.1',
                      'matplotlib>=1.3'],

    # Data files that are part of package
    package_data={},
    data_files=[('', ['LICENSE.rst', 'CHANGELOG.rst'])]

)
