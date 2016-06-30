=====
Magni
=====

:Primary developers:
    Christian Schou Oxvig,
    Patrick Steffen Pedersen

:Additional developers:
   Jan Østergaard,
   Thomas Arildsen,
   Tobias L. Jensen,
   Torben Larsen

:Institution:
   Aalborg University,
   Department of Electronic Systems,
   Signal and Information Processing

:Version:
   1.5.0

:Travis CI test status:
   .. image:: https://travis-ci.org/SIP-AAU/Magni.svg?branch=master
       :target: https://travis-ci.org/SIP-AAU/Magni


Introduction
------------

Magni is a Python package which provides functionality for increasing the speed
of image acquisition using Atomic Force Microscopy (AFM). The image
acquisition algorithms of Magni are based on the Compressed Sensing (CS) signal
acquisition paradigm and include both sensing and reconstruction. The sensing
part of the acquisition generates sensed data from regular images possibly
acquired using AFM. This is done by AFM hardware simulation. The reconstruction
part of the acquisition reconstructs images from sensed data. This is done by
CS reconstruction using well-known CS reconstruction algorithms modified for
the purpose. The Python implementation of the above functionality uses the
standard library, a number of third-party libraries, and additional utility
functionality designed and implemented specifically for Magni. The
functionality provided by Magni can thus be divided into five groups:

- **Atomic Force Microscopy**: AFM specific functionality including AFM image
  acquisition, AFM hardware simulation, and AFM data file handling.
- **Compressed Sensing**: General CS functionality including signal
  reconstruction and phase transition determination.
- **Imaging**: General imaging functionality including measurement matrix and
  dictionary construction in addition to visualisation and evaluation.
- **Reproducibility**: Tools that may aid in increasing the reproducibility of
  results obtained using Magni.
- **Utilities**: General Python utilities including multiprocessing, tracing,
  and validation.


Downloading
-----------

Magni can be downloaded in a number of ways:

- All official releases of Magni are available for download at 
  'http://dx.doi.org/10.5278/VBN/MISC/Magni'.
- The source code is hosted on GitHub at 'https://github.com/SIP-AAU/Magni'
  where every release of Magni is available for download and where **known
  issues** are tracked.

Furthermore, all official releases of the magni package (**without** examples
and documentation) are made available through PyPI and binstar. Both of these
are considered unofficial channels and provided solely for your convenience.

- Using PyPI located at 'https://pypi.python.org/pypi/magni/'.
- Using Anaconda.org located at 'https://anaconda.org/chroxvi/magni/'.


Installation
------------

To use Magni, extract the downloaded archive and include the extracted Magni
folder in your PYTHONPATH.

Magni has been designed for use with Python 2 >= 2.7 or Python 3 >= 3.3.

The required thid party dependencies and optional third party dependencies for
Magni are listed in the compiled documentation which can be found in the Magni
folder under '/doc/build/html/index.html'. You may use the 'dep_check.py'
script found in the Magni folder under '/magni/tests/' to check for missing
dependencies for Magni. Simply run the script to print a dependency report.


Documentation
-------------

The included subpackages, modules, classes and functions are documented through
Python docstrings using the same format as the third-party library, numpy, i.e.
using the numpydoc standard. A description of any entity can thus be found in
the source code of Magni in the docstring of that entity. For readability, the
documentation has been compiled using Sphinx to produce an HTML page which can
be found in the Magni folder under '/doc/build/html/index.html'. The entire
documentation is also available as a PDF file in the Magni folder under
'doc/pdf/index.pdf'. Note, that neither the HTML version nor the PDF version of
the documentation is provided through PyPI and Anaconda.org.


Tests
-----

A test suite consisting of unittests, doctests, IPython notebook examples, and
several style checks is included in Magni. The tests are organised in python
modules found in the Magni folder under '/magni/tests/'. These may be invoked
using any test runner that supports the unittest.TestCase classes. The entire
test suite may be run by executing the convenience script 'run_tests.py'. The
code is this repository is continuously tested by `Travis CI
<https://travis-ci.org/SIP-AAU/Magni>`_.


Examples
--------

A large number of examples showing the capabilities of Magni are included. Some
of these examples are in the form of doctests whereas the more elaborated
examples are in the form of IPython notebook examples which can be found in the
Magni folder under '/examples/'.


Citing Magni
------------

A Magni software metapaper has been published:

Oxvig, C.S., Pedersen, P.S., Arildsen, T, Østergaard, J and Larsen, T 2014.
Magni: A Python Package for Compressive Sampling and Reconstruction of Atomic
Force Microscopy Images. Journal of Open Research Software 2(1):e29,
DOI: http://dx.doi.org/10.5334/jors.bk

Please consider citing this paper if you publish any work in which you have
used Magni.
