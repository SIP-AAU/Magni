.. _file_examples:

========
Examples
========

All the examples are available as `IPython Notebooks <http://ipython.org/notebook.html>`_ in the magni folder under '/examples/'. For an introduction to getting started with IPython Notebook see the `official documentation <http://ipython.org/ipython-doc/stable/notebook/notebook.html>`_.

Starting the IPython Notebook
-----------------------------

Starting the IPython Notebook basically boils down to running:

.. code:: bash

    ipython notebook

from a shell with the working directory set to the Magni '/examples/' folder. Remember to make sure that :py:mod:`magni` is available as described in :ref:`sec_installation` prior to starting the IPython Notebook.

Examples overview
-----------------

An overview of the available examples is given in the below table:

===========================    ======================================================    =========================================
**IPython Notebook Name**      **Example illustrates**                                   **Magni functionality used**
===========================    ======================================================    =========================================
afm-io                         * Reading data from a mi-file.                            * `magni.afm.io.read_mi_file`
                               * Handling the resulting buffers and images.

cs-phase_transition-config     * Using Magni configuration modules including setting     * `magni.cs.phase_transition.config`
                                 and getting configuration values.                       * `magni.utils.config`
								 
cs-phase_transition            * Estimating phase transitions using simulations.         * `magni.cs.phase_transition._util.determine`
                               * Plotting phase transitions.                             * `magni.cs.phase_transition.io`
                               * Plotting phase transition probability colormaps.        * `magni.cs.phase_transition.plotting`

cs-reconstruction              * Reconstruction of compressively sampled 1D signals.     * `magni.cs.reconstruction`

imaging-dictionaries           * Handling compressed sensing dictionaries using          * `magni.imaging.dictionaries`
                                 Magni.

imaging-domains                * Easy handling of an image in the three domains:         * `magni.imaging.domains.MultiDomainImage`
                                 image, measurement and sparse (dictionary).

imaging-measurements           * Handling sampling/measurement patterns using Magni.     * `magni.imaging.measurements`
                               * Sampling a surface.
                               * Sampling an image.
                               * Illustrating sampling patterns.

imaging-preprocessing          * Pre-processing an image prior to sampling               * `magni.imaging.preprocessing`
                               * De-tilting AFM images.

magni                          * The typical work flow in compressively sampling and     * `magni.afm`
                                 reconstructing AFM images using Magni.                  * `magni.imaging`

reporducibility-data           * Obtaining various platform and runtime data for         * `magni.reproducibility.data`
                                 annotations and chases.
								 
reporducibility-io             * Annotating an HDF5 database to help in improving        * `magni.reproducibility.io`
                                 the reproducibility of the results it contains.

util-matrices                  * Using the special Magni Matrix and MatrixCollection     * `magni.utils.matrices.Matrix`
                                 classes.                                                * `magni.utils.matrices.MatrixCollection`

utils-multiprocessing          * Doing multiprocessing using Magni.                      * `magni.utils.multiprocessing`

utils-plotting                 * Using the predefined plotting options in Magni to       * `magni.utils.plotting`
                                 create clearer and more visually pleasing plots.

utils-validation               * Validation of function parameters.                      * `magni.utils.validation`
                               * Disabling input validation to reduce computation
                                 overhead.
===========================    ======================================================    =========================================

