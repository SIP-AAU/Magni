===================
Magni Documentation
===================

:py:mod:`magni` is a Python package developed by Christian Schou Oxvig and Patrick Steffen Pedersen in collaboration with Jan Østergaard, Thomas Arildsen, Tobias L. Jensen, and Torben Larsen. The work was supported by 1) the Danish Council for Independent Research | Technology and Production Sciences - via grant DFF-1335-00278 for the project `Enabling Fast Image Acquisition for Atomic Force Microscopy using Compressed Sensing <http://sip-aau.github.io/FastAFM/>`_ and 2) the Danish e-Infrastructure Cooperation - via a grant for a high performance computing system for the project "High Performance Computing SMP Server for Signal Processing".

This page gives an :ref:`sec_introduction` to the package, briefly describes :ref:`sec_how_to_read_the_documentation`, and explains how to actually use :ref:`sec_the_package`.


.. _sec_introduction:

Introduction
------------

:py:mod:`magni` [#magni]_ is a `Python <https://www.python.org>`_ package which provides functionality for increasing the speed of image acquisition using `Atomic Force Microscopy (AFM) <https://en.wikipedia.org/wiki/Atomic_force_microscopy>`_ (see e.g. [1]_ for an introduction).
The image acquisition algorithms of :py:mod:`magni` are based on the `Compressed Sensing <https://en.wikipedia.org/wiki/Compressed_sensing>`_ (CS) signal acquisition paradigm (see e.g. [2]_ or [3]_ for an introduction) and include both sensing and reconstruction.
The sensing part of the acquisition generates sensed data from regular images possibly acquired using AFM. This is done by AFM hardware simulation. The reconstruction part of the acquisition reconstructs images from sensed data. This is done by CS reconstruction using well-known CS reconstruction algorithms modified for the purpose. The Python implementation of the above functionality uses the standard library, a number of third-party libraries, and additional utility functionality designed and implemented specifically for :py:mod:`magni`. The functionality provided by :py:mod:`magni` can thus be divided into five groups:

- **Atomic Force Microscopy** (:py:mod:`magni.afm`): AFM specific functionality including AFM image acquisition, AFM hardware simulation, and AFM data file handling.
- **Compressed Sensing** (:py:mod:`magni.cs`): General CS functionality including signal reconstruction and phase transition determination.
- **Imaging** (:py:mod:`magni.imaging`): General imaging functionality including measurement matrix and dictionary construction in addition to visualisation and evaluation.
- **Reproducibility** (:py:mod:`magni.reproducibility`): Tools that may aid in increasing the reproducibility of result obtained using :py:mod:`magni`.
- **Utilities** (:py:mod:`magni.utils`): General Python utilities including multiprocessing, tracing, and validation.

See :ref:`sec_other_resources` as well as :ref:`sec_notation` for further documentation related to the project and the :ref:`sec_tests` and :ref:`sec_examples` to draw inspiration from.

.. rubric:: References

.. [1] \B. Bhushan and O. Marti , "Scanning Probe Microscopy – Principle of Operation, Instrumentation, and Probes", *in Springer Handbook of Nanotechnology*, pp. 573-617, 2010.
.. [2] D.L. Donoho, "Compressed Sensing", *IEEE Transactions on Information Theory*, vol. 52, no. 4, pp. 1289-1306, Apr. 2006.
.. [3] E.J. Candès, J. Romberg, and T. Tao, "Robust Uncertainty Principles: Exact Signal Reconstruction From Highly Incomplete Frequency Information", *IEEE Transactions on Information Theory*, vol. 52, no.2, pp. 489-509, Feb. 2010.

.. rubric:: Footnotes

.. [#magni] In Norse mythology, Magni is son of Thor and the god of strength. However, the word MAGNI could as well be an acronym for almost anything including "Making AFM Grind the Normal Impatience".


.. _sec_how_to_read_the_documentation:

How to Read the Documentation
-----------------------------

The included subpackages, modules, classes and functions are documented through Python docstrings using the same format as the third-party library, numpy, i.e. using the `numpydoc standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_. A description of any entity can thus be found in the source code of :py:mod:`magni` in the docstring of that entity. For readability, the documentation has been compiled using |sphinx|_ to produce this HTML page which can be found in the magni folder under '/doc/build/html/index.html'. The entire documentation is also available as a PDF file in the magni folder under '/doc/build/pdf/index.pdf'.

.. _sec_building_the_documentation:

Building the Documentation
==========================

The HTML documentation may be built from source using the supplied Makefile in the magni folder under '/doc/'. Make sure the required :ref:`sec_dependencies` for building the documentation are installed. The build process consists of running three commands:

.. code:: bash

    $ make sourceclean
    $ make docapi
    $ make html

.. note:: 
   In the *make docapi* command it is assumed that the python interpreter is available on the PATH under the name *python*. If the python interpreter is available under another name, the PYTHONINT variable may be set, e.g. "make PYTHONINT=python2 docapi" if the python interpreter is named python2.

Run :code:`make clean` to remove all builds created by |sphinx|_ under '/doc/build'.


.. _sec_the_package:

The Package
-----------

The source code of :py:mod:`magni` is released under the `BSD 2-Clause <http://opensource.org/licenses/BSD-2-Clause>`_ license, see the :ref:`sec_license` section. To install :py:mod:`magni`, follow the instructions given under :ref:`sec_installation`.

:py:mod:`magni` has been tested with `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ (64-bit) on Linux. It may or may not work with other Python distributions and/or operating systems. See also the list of :ref:`sec_dependencies` for :py:mod:`magni`.


.. _sec_license:

License
=======

.. include:: ../../LICENSE.rst


.. _sec_installation:

Download and Installation
=========================

All official releases of :py:mod:`magni` are available for download at :doi:`10.5278/VBN/MISC/Magni`. The source code is hosted on GitHub at https://github.com/SIP-AAU/Magni.

To use Magni, extract the downloaded archive and include the extracted magni folder in your `PYTHONPATH <https://docs.python.org/2/tutorial/modules.html#the-module-search-path>`_. 

.. note::

   The :py:mod:`magni` package (excluding examples and documentation) is also available on an "as is" basis in source form at `PyPi <https://pypi.python.org/pypi/magni>`_ and as a `conda <http://conda.pydata.org/docs/index.html>`_ package at `Binstar <http://binstar.org/chroxvi/magni>`_.

.. _sec_dependencies:

Dependencies
============

:py:mod:`magni` has been designed for use with |python2|_ >= 2.7 or |python3|_ >= 3.3

**Required** third party dependencies for :py:mod:`magni` are:


- |matplotlib|_ (Tested on version >= 1.3)
- |numpy|_ (Tested on version >= 1.8)
- |pytables|_  (Tested on version >= 3.1)
- |scipy|_ (Tested on version >= 0.14)

**Optional** third party dependencies for :py:mod:`magni` are:

- |bottleneck|_ (Tested on version >=1.0.0) (For speed-up of some algorithms)
- |coverage|_ (Tested on version >= 3.7) (For running the test suite script)
- |ipython|_ (Tested on version >= 2.1) (For running the IPython notebook examples)
- |mkl|_ (Tested on version >= 11.1) (For accelerated vector operations)
- |nose|_ (Tested on version >= 1.3) (For running unittests and doctests)
- |pep8|_ (Tested on version >= 1.5) (For running style check tests)
- |pil|_ (Tested on version >= 1.1.7) (For running the IPython notebook examples as tests)
- |pyflakes|_ (Tested on version >= 0.8) (For running style check tests)
- |radon|_ (Tested on version >= 1.2) (For running style check tests)
- |sphinx|_ (Tested on version >= 1.3.1) (For building the documentation from source)


.. note:: 

   When using the :py:mod:`magni.utils.multiprocessing` subpackage, it is generally a good idea to restrict acceleration libraries like MKL or OpenBLAS to use a single thread. If MKL is installed, this is done automatically at runtime in the :py:mod:`magni.utils.multiprocessing` subpackage. If other libraries than MKL are used, the user has to manually set an appropriate evironmental variable, e.g. OMP_NUM_THREADS.

.. |python2| replace:: ``Python 2``
.. _Python2: https://www.python.org/
.. |python3| replace:: ``Python 3``
.. _Python3: https://www.python.org/
.. |pytables| replace:: ``PyTables``
.. _pytables: http://www.pytables.org/
.. |numpy| replace:: ``Numpy``
.. _numpy: http://www.numpy.org/
.. |scipy| replace:: ``Scipy``
.. _scipy: http://scipy.org/scipylib/index.html
.. |matplotlib| replace:: ``Matplotlib``
.. _matplotlib: http://matplotlib.org
.. |ipython| replace:: ``IPython``
.. _ipython: http://ipython.org/
.. |mkl| replace:: ``Math Kernel Library (MKL)``
.. _mkl: https://software.intel.com/en-us/intel-mkl
.. |sphinx| replace:: ``Sphinx``
.. _sphinx: http://sphinx-doc.org/
.. |nose| replace:: ``Nose``
.. _nose: https://nose.readthedocs.org/en/latest/
.. |pep8| replace:: ``PEP8``
.. _pep8: http://pep8.readthedocs.org/en/latest/
.. |pyflakes| replace:: ``Pyflakes``
.. _pyflakes: https://launchpad.net/pyflakes
.. |radon| replace:: ``Radon``
.. _radon: https://radon.readthedocs.org/en/latest/
.. |coverage| replace:: ``Coverage``
.. _coverage: http://nedbatchelder.com/code/coverage/
.. |pil| replace:: ``PIL (or Pillow)``
.. _pil: http://www.pythonware.com/products/pil/
.. |bottleneck| replace:: ``Bottleneck``
.. _bottleneck: http://berkeleyanalytics.com/bottleneck/

You may use the *dep_check.py* script found in the Magni folder under '/magni/tests/' to check for missing dependencies for Magni. Simply run the script to print a dependency report, e.g.:

.. code:: bash

  $ python dep_check.py


.. _sec_tests:

Tests
-----

A test suite consisting of unittests, doctests, the IPython notebook examples, and several style checks is included in :py:mod:`magni`. The tests are organized in python modules found in the Magni folder under '/magni/tests/'. Each module features one or more :py:mod:`unittest.TestCase` classes containing the tests. Thus, the tests may be invoked using any test runner that supports the :py:mod:`unittest.TestCase`. E.g. running the wrapper for the doctests using |nose|_ is done by issuing:

.. code:: bash

  $ nosetests magni/tests/wrap_doctests.py

The entire test suite may be run by executing the convenience script :code:`run_tests.py`:

.. code:: bash

   $ magni/tests/run_tests.py

.. note::

   This convenience script assumes that :py:mod:`magni` is available on the PYTHONPATH as explained under :ref:`sec_installation`.

.. _sec_bug_reports:

Bug Reports
-----------

Found a bug? Bug report may be submitted using the magni `GitHub issue tracker <https://github.com/SIP-AAU/Magni/issues>`_. Please include all relevant details in the bug report, e.g. version of Magni, input/output data, stack traces, etc. If the supplied information does not entail reproducibility of the problem, there is no way we can fix it. 

.. note::
   **Due to limited funds, we are unfortunately unable make any guarantees, whatsoever, that reported bugs will be fixed.**

.. _sec_other_resources:

Other Resources
---------------

Papers published in relation to the `Enabling Fast Image Acquisition for Atomic Force Microscopy using Compressed Sensing <http://sip-aau.github.io/FastAFM/>`_ project:

- \C. S. Oxvig, P. S. Pedersen, T. Arildsen, J. Østergaard, and T. Larsen, "Magni: A Python Package for Compressive Sampling and Reconstruction of Atomic Force Microscopy Images", *Journal of Open Research Software*, vol. 2, no. 1, p. e29, Oct. 2014, :doi:`10.5334/jors.bk`.

- \T. L. Jensen, T. Arildsen, J. Østergaard, and T. Larsen, "Reconstruction of Undersampled Atomic Force Microscopy Images : Interpolation versus Basis Pursuit", in *International Conference on Signal-Image Technology & Internet-Based Systems (SITIS)*, Kyoto, Japan, December 2 - 5, 2013, pp. 130-135, :doi:`10.1109/SITIS.2013.32`.


.. _sec_notation:

========
Notation
========

To the extent possible, a consistent notation has been used in the documentation and implementation of algorithms that are part of :py:mod:`magni`. All the details are described in :ref:`file_notation`.

.. toctree::
   :hidden:

   notation


.. _sec_examples:

========
Examples
========

The :py:mod:`magni` package includes a large number of examples showing its capabilities. See the dedicated :ref:`file_examples` page for all the details.

.. toctree::
   :hidden:

   examples

============
API Overview
============
An overview of the high level :py:mod:`magni` API is given below:

.. toctree::
   :maxdepth: 5

   magni
