.. _file_notation:

==================
A Note on Notation
==================

As much as possible, a consistent notation is used in the :py:mod:`magni` package. This implies that variable names are shared between functions that are related. Furthermore a consistent coordinate system is used for the description of related surfaces.


The Compressed Sensing Reconstruction Problem
---------------------------------------------

In the :py:mod:`magni.cs` subpackage, a consistent naming scheme is used for variables, i.e., vectors and matrices. This section gives an overview of the chosen notation. For the purpose of illustration, consider the Basis Pursuit CS reconstruction problem [1]_:

.. math::

   &\text{minimize}\qquad    ||\alpha||_1 \\
   &\text{subject to}\qquad  \mathbf{z} = \mathbf{A}\mathbf{\alpha}

Here :math:`\mathbf{A} \in \mathbb{C}^{m \times n}` is the matrix product of a sampling matrix :math:`\mathbf{\Phi} \in \mathbb{C}^{m \times p}` and a dictionary matrix :math:`\mathbf{\Psi} \in \mathbb{C}^{p \times n}`. The dictionary coefficients are denoted :math:`\mathbf{\alpha} \in \mathbb{C}^{n \times 1}` whereas the (noiseless) measurements are denoted by :math:`\mathbf{z} \in \mathbb{C}^{m \times 1}`.

Thus, the following relations are used:

.. math::

   \mathbf{A} &= \mathbf{\Phi}\mathbf{\Psi} \\
              & \\
   \mathbf{x} &= \mathbf{\Psi}\mathbf{\alpha} \\
              & \\
   \mathbf{z} &= \mathbf{\Phi}\mathbf{x} \\
              &= \mathbf{\Phi}\mathbf{\Psi}\mathbf{\alpha} \\
			  &= \mathbf{A}\mathbf{\alpha} \\

Here the vector :math:`\mathbf{x} \in \mathbb{C}^{p \times 1}` represents the signal of interrest. That is, it is the signal that is assumed to have a sparse representation in the dictionary :math:`\mathbf{\Psi}`. The sparsity of the coefficient vector :math:`\mathbf{\alpha}`, that is the size of the support set, is denoted :math:`k=|\text{supp}(\mathbf{\alpha})|`. Furthermore, we have the undersampling ratio :math:`\delta=m/n`, the sparsity level is :math:`\rho=k/m`, and the signal density is :math:`\tau=k/n`.

Oftentimes, one considers noisy measurements, e.g. measurements corrupted by Additive White Gaussian Noise (AWGN) :math:`\mathbf{e} \in \mathbb{R}^{m \times 1}` with

.. math::

   \mathbf{y} &= \mathbf{z} + \mathbf{e} \\
              &= \mathbf{A}\mathbf{\alpha} + \mathbf{e} \\

Here the (noisy) measurements are denoted :math:`\mathbf{y} \in \mathbb{C}^{m \times 1}`.

.. note::

   All vectors in :py:mod:`magni` are considered to be column vectors, i.e. they explicitly have a second dimension of length 1 when defined as :py:mod:`numpy` arrays.

.. note::

   Even though the above example involves complex vectors and matrices, the algorithms provided in :py:mod:`magni.cs` may be restricted to inputs and outputs that are real.

.. rubric:: References

.. [1] S. Chen, D. L. Donoho, and M. A. Saunders, "Atomic Decomposition by Basis Pursuit", *Siam Review*, vol. 43, no. 1, pp. 129-159, Mar. 2001.


Handling Images as Matrices and Vectors
---------------------------------------

In parts of :py:mod:`magni.imaging`, an image is considered a matrix :math:`\mathbf{M} \in \mathbb{R}^{h \times w}`. That is, the image height is :math:`h` whereas the width is :math:`w`. In the :py:mod:`magni.cs` subpackage, the image must be represented as a vector. This is done by stacking the columns of :math:`\mathbf{M}` to form the vector :math:`\mathbf{x}`. Thus, the dimension of the image vector representation is :math:`n = h \cdot w`. The :py:func:`magni.imaging._util.vec2mat` (available as :py:func:`magni.imaging.vec2mat`) and :py:func:`magni.imaging._util.mat2vec` (available as :py:func:`magni.imaging.mat2vec`) may be use to convert between the matrix and vector notations.

When the matrix representation is used, the following coordinate system is used for its visual representation:

| \
|  -----------------------------------> x (first axis - width :math:`w`)
|  \|
|  \|
|  \|
|  \|
|  \|
|  \|
|  \|
|  \|
|  \|
|  \|
|  v
|  y (second axis - height :math:`h`)


This way, a position on an AFM sample of size :math:`w \times h` is specified by a :math:`(x, y)` coordinate pair.
