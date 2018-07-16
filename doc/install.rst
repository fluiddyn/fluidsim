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

- `fluiddyn <http://fluiddyn.readthedocs.io>`_

  The base package of the FluidDyn project is a pure Python package that will be
  installed automatically so you should not need to worry about this dependency.

- `fluidfft <http://fluidfft.readthedocs.io>`_

  fluidsim needs fluidfft. If you don't install it before carefully, it will be
  installed automatically and you won't be able to use fancy FFT libraries
  (using for example MPI with 2D decomposition or GPU with CUDA). If you are
  not too concerned about performance, no problem. Otherwise, install fluidfft
  as explained `here <http://fluidfft.readthedocs.io/en/latest/install.html>`__.

- h5py (optionally, with MPI support, but only if you know what you do)

  .. warning::

    Prebuilt installations (for eg. via h5py wheels) may lack MPI support.
    Most of the time, this is what you want.  However, you can install h5py
    from source and link it to a hdf5 built with MPI support, as follows:

    .. code:: bash

       $ CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-deps --no-binary=h5py h5py
       $ python -c 'import h5py; h5py.run_tests()'

    See the `h5py documentation
    <http://docs.h5py.org/en/latest/build.html>`_ for more details.

- A C++11 compiler (for example GCC 4.9 or clang)

- `Pythran <https://github.com/serge-sans-paille/pythran>`_

  We choose to use the new static Python compiler `Pythran
  <https://github.com/serge-sans-paille/pythran>`_ for some functions. Our
  microbenchmarks show that the performances are as good as what we are able to
  get with Fortran or C++!

  .. warning::

     To reach good performance, we advice to try to put in the file
     `~/.pythranrc` the lines (it seems to work well on Linux, see the `Pythran
     documentation <https://pythonhosted.org/pythran/MANUAL.html>`_):

     .. code:: bash

        [pythran]
        complex_hook = True

  .. warning::

     The compilation of C++ files produced by Pythran can be long and can
     consume a lot of memory. If you encounter any problems, you can try to use
     clang (for example with ``conda install clangdev``) and to enable its use
     in the file `~/.pythranrc` with:

     .. code:: bash

        [compiler]
        CXX=clang++
        CC=clang

- Optionally (for MPI runs), `mpi4py <http://mpi4py.scipy.org>`_ (which depends
  on a MPI implementation).

- Optionally (for some command-line tools), `Pandas
  <https://pandas.pydata.org/>`_.

Basic installation with pip
---------------------------

If you are in a hurry and that you are not really concerned about performance,
you can use pip::

  pip install fluidsim

or::

  pip install fluidsim --user

You can also configure the installation of fluidsim by creating the file
``~/.fluidsim-site.cfg`` and modify it to fit your requirements before the
installation with pip::

  wget https://bitbucket.org/fluiddyn/fluidsim/raw/default/site.cfg.default -O ~/.fluidsim-site.cfg


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
from `the PyPI page <https://pypi.org/project/fluidsim>`_.

Configuration file
~~~~~~~~~~~~~~~~~~

For particular installation setup, copy the default configuration file::

  cp site.cfg.default site.cfg

and modify it to fit your requirements.

.. warning::

   If you care about performance, correctly set up a configuration file. By
   default, some Pythran files are not Pythranized!

Build/install
~~~~~~~~~~~~~

Build/install in development mode (with a virtualenv or with conda)::

  python setup.py develop

or (without virtualenv)::

  python setup.py develop --user

Of course you can also install FluidDyn with the install command ``python
setup.py install``.


Run the tests!
--------------

You can run some unit tests by running ``make tests`` (shortcut for
``fluidsim-test -v``) or ``make tests_mpi`` (shortcut for ``mpirun -np 2
fluidsim-test -v``). Alternatively, you can also run ``python -m unittest
discover`` from the root directory or from any of the "test" directories.


Environment variables
---------------------

Fluidsim builds its binaries in parallel. It speedups the build process a lot on
most computers. However, it can be a very bad idea on computers with not enough
memory. If you encounter problems, you can force the number of processes used
during the build using the environment variable ``FLUIDDYN_NUM_PROCS_BUILD``::

   export FLUIDDYN_NUM_PROCS_BUILD=2

Fluidsim is also sensible to the environment variables:

- ``FLUIDSIM_PATH``: path where the simulation results are saved.

  In Unix systems, you can for example put this line in your ``~/.bashrc``::

    export FLUIDSIM_PATH=$HOME/Data

- ``FLUIDDYN_PATH_SCRATCH``: working directory (can be useful on some clusters).
