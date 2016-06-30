==================
1.5.0 (2016-06-30)
==================

Version 1.5.0 is primarily an addition of extensive unittests of multiple parts
of magni and a redo of magni.utils.multiprocessing.process. Furthermore, this
version includes minor improvements and minor bug fixes.


Additions
---------

- Added extensive unittests of the following modules and subpackages:

  * magni.reproducibility
  * magni.utils.matrices
  * magni.utils.multiprocessing
  * magni.utils.validation


Improvements
------------

- Improved magni.reproducibility with the following changes:

  * Added env_export to conda_info in get_conda_info to hold information about
	the active environment in order to include information about pip packages.
  * Added an annotations_sub_group argument to write_custom_annotation to allow
	grouping custom annotations in a tree structure.

- Improved magni.utils.validation with the following changes:

  * Added a superclass argument to validate_generic to allow validating classes
	based on inheritance.
  * Added the validate_once function to allow validating the arguments of a
	function only the first time the function is called. Furthermore, added the
	enable_validate_once function to enable validate_once which is disabled by
	default.

- Improved magni.utils.multiprocessing.process by adding the option to use
  concurrent.futures instead of multiprocessing each having some advantages
  over the other. See the documentation for more information.


Bug Fixes
---------

- Fixed a bug where importing magni raised an exception even when the package
  dependencies were met. This was done by making package version checking more
  robust.
- Fixed a mathematically incorrect behaviour as, previously, Matrix.T actually
  implied Matrix.conj().T, and MatrixCollection.T actually implied
  MatrixCollection.conj().T.
- Fixed a few holes in the checks of the validation schemes of the validation
  functions to catch invalid validation schemes.
- Made all doctests pass even if the example.mi is not available - this is only
  relevant when retrieving magni from secondary sources.
- Fixed a number of minor bugs.
- Various documentation clean-ups.



==================
1.4.0 (2016-01-29)
==================

Version 1.4.0 is primarily an addition of the cs.indicators package, an
extension of the imaging.dictionaries package, an extension of the
imaging.measurements package, and an extension of the reproducibility package.
Furthermore, unittests have been added, examples have been updated and added,
and this version includes minor improvements and minor bug fixes.


Additions
---------

- Added the cs.indicators module which currently provides functions to
  calculate the values of the performance indicators; coherence, mutual
  coherence, and relative energy in relation to measurement matrices and
  dictionary matrices.
- Added functionality to the imaging.measurements package.

  * imaging.measurements is now a package rather than a module.
  * A number of functions have been added: construct_pixel_mask,
	lissajous_sample_image, lissajous_sample_surface,
	uniform_rotated_line_sample_image, uniform_rotated_line_sample_surface,
	zigzag_sample_image, and zigzag_sample_surface.

- Added functionality to the reproducibility package:

  * Added the reproducibility.data module which currently provides access to
	the following already existing functions: get_conda_info, get_datetime,
	get_git_revision, get_magni_config, get_magni_info, get_main_file_name,
	get_main_file_source, get_main_source, get_platform_info, get_stack_trace.
  * Added a function to reproducibility.data: get_file_hashes.
  * Added a function to reproducibility.io: write_custom_annotation.

- Added unittests for the cs.indicators module, the imaging.dictionaries
  package, and the utils.plotting module.
- Added an iPython Notebook example of the usage of the reproducibility.data
  module.


Improvements
------------

- Improved the dictionary matrix generating functions of the
  imaging.dictionaries package to accept overcomplete DCT and DFT matrices.
- Rewritten the following functions of the imaging.measurements package for
  readability and reduced computation time: random_line_sample_image,
  random_line_sample_surface, uniform_line_sample_image, and
  uniform_line_sample_surface.
- Improved the imsubplot function of the imaging.visualisation module to accept
  more versatile x_ticklabels and y_ticklabels as well as to accept a
  fixed_clim.
- Improved the get_git_revision function of the reproducibility.data module to
  allow for other git root directories and to include git remote -v
  information.
- Updated tests.ipynb_examples to support Jupyter in addition to iPython 2 and
  iPython 3.
- Updated the following examples to reflect the current state of the package:
  imaging-dictionaries, imaging-measurements, magni, and reproducibility-io.


Bug Fixes
---------

- Fixed a few bugs in magni.utils.validation.validate_numeric.

  * The exception for non-numeric variables is now raised by the module with a
	sensical message rather than as a side-effect.
  * The function call now accepts precisions which exist only for some of the
	specified types.

- Fixed a number of minor bugs.
- Various documentation clean-ups.



==================
1.3.0 (2015-05-05)
==================

Version 1.3.0 is primarily an extension of the afm package, a generalisation
and restructuring of the cs.reconstruction package, and an extension of the
imaging package. Furthermore, this version includes minor improvements and
minor bug fixes.


Additions
---------

- Added functionality to the imaging package.

  * A number of functions have been added: double_mirror, get_inscribed_masks,
	visualisation.imsubplot, and visualisation.mask_img_from_coords.
  * imaging.dictionaries is now a package rather than a module. In addition to
	the existing functionality, the package has an added analysis module for
	analysing dictionaries.

- Added the utils.types module which currently provides the following
  general-purpose classes: ClassProperty and ReadOnlyDict.
- Added functionality to chase data for reproducibility purposes. A number of
  functions have been added to the reproducibility.io module: chase_database,
  create_database, read_chases, and remove_chases.
- Added tests of the cs.phase_transition package and of the cs.reconstruction
  package.
- Sphinx >= 1.3 is now required to build the documentation.


Improvements
------------

- Added functionality for reading and representing all known .mi files.

  * afm.io and afm.types are now packages rather than modules.
  * afm.io.read_mi_file supports all known .mi files.
  * afm.types contains the following classes (of which some are rewrites of
	previous classes): BaseClass, File, FileCollection, image.Buffer,
	image.Image, spectroscopy.Buffer, spectroscopy.Chunk, spectroscopy.Grid,
	spectroscopy.Point, and spectroscopy.Spectroscopy.

- Added functionality for customising the compressive sensing reconstruction
  algorithms of the cs.reconstruction package.

  * cs.reconstruction.it has been added for general iterative thresholding
	compressive sensing reconstruction functionality. Besides from being able
	to act as standard iterative hard thresholding and iterative soft
	thresholding, the threshold operator, the intial point and the step size
	can be configured.
  * cs.reconstruction.sl0 has been modified for general smoothed l0 compressive
	sensing reconstruction functionality. Besides from being able to act as
	original smoothed l0 and modified smoothed l0, the sigma, L, and mu
	parameters can be configured.


Bug Fixes
---------

- Fixed a number of minor bugs.
- Various documentation clean-ups.



==================
1.2.0 (2015-03-13)
==================

Version 1.2.0 is primarily a rewrite of the validation and configuration parts
of the package combined with the addition of automated testing capabilities.
Furthermore, this version includes minor improvements and bug fixes.


Additions
---------

- Added automated testing capabilities.

  * tests/run_tests.py runs all tests in the directory.
  * tests/wrap_doctests.py and tests/ipynb_examples.py check that all doctests
    and ipython notebook examples of the package produce the expected results.
  * tests/style_checks.py checks the code of the importable package for various
    errors using pyflakes, for PEP8 conformance, and for acceptable cyclomatic
    complexity using radon.
  * tests/build_docs.py checks that the documentation of the package can be
	automatically generated using sphinx.
  * tests/config.py, tests/imaging_evaluation.py, and tests/reproducibility.py
	test specific parts of the package.


Improvements
------------

- Rewritten validation functionality.

  * magni.utils.validation.validate_generic has been added for validation of
    generic (generally non-numeric) variables through an interface which is
    less error-prone and has a higher abstraction level than validate.
  * magni.utils.validation.validate_numeric has been added for validation of
    numeric variables through an interface which is less error-prone and has a
    higher abstraction level than validate and validate_ndarray.
  * magni.utils.validation.validate_levels has been added for validation of
	"nested" variables (sequences, sets, mappings, etc.) through an interface
	which is less error-prone and has a higher abstraction level than validate.

- Updated every validation call in the package to use the new validation
  functionality resulting in improved validation.
- Rewritten magni.utils.config.Configger to provide a subset of the interface
  of a dict in addition to the get and set methods.
- Updated every config module in the package to use the new Configger
  functionality resulting in increased readability.
- Changed some of the configuration parameter names which may cause the new
  version of the package to be incompatible with code written for a previous
  version (sorry, but this should not happen again).

  * In cs.phase_transition.config: renamed 'n' to 'problem_size'.
  * In cs.reconstruction.iht.config: renamed 'kappa' to 'kappa_fixed', and
    'threshold_rho' to 'threshold_fixed'.
  * In cs.reconstruction.sl0.config: replaced 'algorithm' by 'sigma_start',
    'L', and 'mu'; replaced 'L' by 'L_geometric_start' and 'L_fixed'; and
    renamed 'L_update' to 'L_geometric_ratio', 'mu' to 'mu_fixed', 'mu_end' to
    'mu_step_end', 'mu_start' to 'mu_step_start', 'sigma_min' to
    'sigma_stop_fixed', and 'sigma_update' to 'sigma_geometric'.

- Changed doctests to import required modules to allow nosetests and similar
  software to run the doctests of the package.
- Added a configuration option in magni.utils.multiprocessing.config,
  'silence_exceptions', to silence exceptions when using
  magni.utils.multiprocessing.process.
- Made minor improvements to selected parts of the package.


Bug Fixes
---------

- Fixed a number of minor bugs.



==================
1.1.0 (2014-11-25)
==================

Version 1.1.0 is primarily an improvement of the IPython Notebook examples and
the docstring examples. Furthermore, this version includes minor improvements
and bug fixes.


Additions
---------

- Added markdown comments and more visual output to the IPython Notebook
  examples.


Improvements
------------

- Changed docstring examples to yield more robust output and thus pass
  doctests in a wider variety of environments.
- Changed docstring examples relying on the provided example.mi file to
  unconditionally pass rather than fail if the example file is unavailable.
- Changed the default colormap from 'jet' to 'coolwarm'.
- Made minor improvements to selected parts of the package.


Bug Fixes
---------

- Fixed a number of minor bugs.



==================
1.0.0 (2014-05-23)
==================

Version 1.0.0 is the first public release of the Magni package. The present
version is essentially a rewrite of most of the code featured in version 0.1.0
alongside a lot of new code. The additions and improvements are reflected
directly in the extensive documentation of this version. The present entry in
the changelog is thus kept to a minimum whereas future versions will include
fewer additions and improvements and they will be accompanied by more detailed
changelog entries.

The public interface introduced is as follows:

- magni.afm.config.get
- magni.afm.config.set
- magni.afm.io.read_mi_file
- magni.afm.reconstruction.analyse
- magni.afm.reconstruction.reconstruct
- magni.afm.types.Buffer
- magni.afm.types.Image
- magni.cs.phase_transition.config.get
- magni.cs.phase_transition.config.set
- magni.cs.phase_transition.io.load_phase_transition
- magni.cs.phase_transition.plotting.plot_phase_transition_colormap
- magni.cs.phase_transition.plotting.plot_phase_transitions
- magni.cs.phase_transition.determine
- magni.cs.reconstruction.iht.config.get
- magni.cs.reconstruction.iht.config.set
- magni.cs.reconstruction.iht.run
- magni.cs.reconstruction.sl0.config.get
- magni.cs.reconstruction.sl0.config.set
- magni.cs.reconstruction.sl0.run
- magni.imaging.dictionaries.get_DCT
- magni.imaging.dictionaries.get_DFT
- magni.imaging.domains.MultiDomainImage
- magni.imaging.evaluation.calculate_mse
- magni.imaging.evaluation.calculate_psnr
- magni.imaging.evaluation.calculate_retained_energy
- magni.imaging.measurements.construct_measurement_matrix
- magni.imaging.measurements.plot_pattern
- magni.imaging.measurements.plot_pixel_mask
- magni.imaging.measurements.random_line_sample_image
- magni.imaging.measurements.random_line_sample_surface
- magni.imaging.measurements.spiral_sample_image
- magni.imaging.measurements.spiral_sample_surface
- magni.imaging.measurements.square_spiral_sample_image
- magni.imaging.measurements.square_spiral_sample_surface
- magni.imaging.measurements.uniform_line_sample_image
- magni.imaging.measurements.uniform_line_sample_surface
- magni.imaging.measurements.unique_pixels
- magni.imaging.preprocessing.detilt
- magni.imaging.visualisation.imshow
- magni.imaging.visualisation.shift_mean
- magni.imaging.visualisation.stretch_image
- magni.imaging.mat2vec
- magni.imaging.vec2mat
- magni.reproducibility.io.annotate_database
- magni.reproducibility.io.read_annotations
- magni.reproducibility.io.remove_annotations
- magni.utils.multiprocessing.config.get
- magni.utils.multiprocessing.config.set
- magni.utils.multiprocessing.File
- magni.utils.multiprocessing.process
- magni.utils.config.Configger
- magni.utils.matrices.Matrix
- magni.utils.matrices.MatrixCollection
- magni.utils.plotting.setup_matplotlib
- magni.utils.plotting.colour_collections
- magni.utils.plotting.div_cmaps
- magni.utils.plotting.linestyles
- magni.utils.plotting.markers
- magni.utils.plotting.seq_cmaps
- magni.utils.validation.decorate_validation
- magni.utils.validation.disable_validation
- magni.utils.validation.validate
- magni.utils.validation.validate_ndarray
- magni.utils.split_path


Improvements
------------

- Rewrote 'magni.cs.phase_transition' to use 'magni.utils' functionality and
  simplify the code significantly.
- Rewrote 'magni.cs.phase_transition' to use pytables instead of h5py by using
  'magni.utils.multiprocessing.File' to increase the abstraction level.
- Refactored 'magni.cs.reconstruction' to use a consistent naming convention
  for the modules of a reconstruction algorithm.
- Added validation options to the functions of the 'magni.utils.validation'
  module.
- Reformatted the packages, modules, and functions in the present package to be
  PEP8 compliant.
- Documented the packages, modules, and functions in the present package in a
  format compatible with the sphinx numpydoc plugin according to
  https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt



==================
0.1.0 (2013-10-28)
==================

Version 0.1.0 is basically the merge of selected functionality from two
previous Python packages, the Compressive Sensing Simulation Framework ('cssf')
and the Wind Analysis Framework ('waf'). A few essential improvements and a
single bug fix are included in this version but everything else is postponed to
be included in the next version.


Additions
---------

- Copied a number of subpackages from the Compressive Sensing Simulation
  Framework ('cssf') package into the present package with minor changes:

  * The 'cssf.iht' subpackage as 'magni.cs.reconstruction.iht'.
  * The 'cssf.sl0' subpackage as 'magni.cs.reconstruction.sl0'.
  * The 'cssf.test' subpackage as 'magni.cs.phase_transition'.

- Copied a number of subpackages from the Wind Analysis Framework ('waf')
  package into the present package with minor changes:

  * The 'waf.multiprocessing' subpackage as 'magni.utils.multiprocessing'.
  * Elements ('_util.split_path', '_validation.decorate_validation', and
    '_validation.validate') of the 'waf.utils' subpackage as 'magni.utils'.


Improvements
------------

- Changed 'magni.cs.phase_transition' to run simulations in parallel to reduce
  the time spent on simulating reconstruction algorithms.
- Changed 'magni.utils.validation' to include the function 'disable_validation'
  which globally disables validation to reduce the time spent on computations.


Bug Fixes
---------

- Fixed a bug with multiprocessing and mkl competing for CPU cores.
