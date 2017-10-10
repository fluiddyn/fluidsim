Installation
============

FluidSim is part of the FluidDyn project.  Some issues regarding the
installation of Python packages are discussed in `the main
documentation of the project
<http://fluiddyn.readthedocs.org/en/latest/install.html>`_.

Dependencies
------------

- Python 2.7 or >= 3.4

- Numpy

- `fluidfft <http://fluidfft.readthedocs.io>`_

  fluidsim needs fluidfft. If you don't install it before carefully, it will be
  installed automatically and you won't be able to use fancy FFT libraries
  (using for example MPI with 2D decomposition or CUDA). If you are not too
  concerned about performance, no problem. Otherwise, install fluidfft as
  explained `here <http://fluidfft.readthedocs.io/en/latest/install.html>`__

- A C++11 compiler (for example GCC 4.9)

- `Pythran <https://github.com/serge-sans-paille/pythran>`_

  We choose to use the new static Python compiler `Pythran
  <https://github.com/serge-sans-paille/pythran>`_ for some functions. Our
  microbenchmarks show that the performances are as good as what we are able to
  get with Fortran or C++!

.. warning::

  To reach good performance, we advice to try to put in the file `~/.pythranrc`
  the lines (see the `Pythran documentation
  <https://pythonhosted.org/pythran/MANUAL.html>`_):

  .. code:: bash

     [pythran]
     complex_hook = True

- h5py (optionally, with MPI support)

.. warning::

  Prebuilt installations (for eg. via h5py wheels) may lack MPI support. It may
  be useful to install from source, as follows:

  .. code:: bash

     $ CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-deps --no-binary=h5py h5py
     $ python -c 'import h5py; h5py.run_tests()'

  See the `h5py documentation
  <http://docs.h5py.org/en/latest/build.html>`_ for more details.

- Optionally, mpi4py (which depends on a MPI implementation).

Basic installation with pip
---------------------------

If you are in a hurry and that you are not really concerned about performance,
you can use pip::

  pip install fluidsim

or::

  pip install fluidsim --user


Install from the repository (recommended)
-----------------------------------------

Get the source code
~~~~~~~~~~~~~~~~~~~

For fluidsim, we use the revision control software Mercurial and the main
repository is hosted `here <https://bitbucket.org/fluiddyn/fluidsim>`_ in
Bitbucket. Download the source with something like::

  hg clone https://bitbucket.org/fluiddyn/fluidsim

If you are new with Mercurial and Bitbucket, you can also read `this short
tutorial
<http://fluiddyn.readthedocs.org/en/latest/mercurial_bitbucket.html>`_.

If you don't want to use Mercurial, you can also just manually download the
package from `the Bitbucket page <https://bitbucket.org/fluiddyn/fluidsim>`_ or
from `the PyPI page <https://pypi.python.org/pypi/fluidsim>`_.


Build/install
~~~~~~~~~~~~~

Build/install in development mode (with a virtualenv)::

  python setup.py develop

or (without virtualenv)::

  python setup.py develop --user

Of course you can also install FluidDyn with the install command ``python
setup.py install``.


Run the tests!
--------------

You can run some unit tests by running ``make tests`` or ``make tests_mpi``
from the root directory or ``python -m unittest discover`` from the root
directory or from any of the "test" directories.


