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
