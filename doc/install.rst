Installation
============

FluidSim is part of the FluidDyn project and requires Python >= 3.6. Some
issues regarding the installation of Python and Python packages are discussed
in `the main documentation of the project
<http://fluiddyn.readthedocs.org/en/latest/install.html>`_.

We don't upload wheels on PyPI, so the simplest and fastest procedure is to
use ``conda`` (no compilation needed). Alternatively, one can compile fluidsim
and fluidfft using ``pip``.

Installing the conda-forge packages with conda or mamba
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fluidsim packages are in the conda-forge channels, so if it is not already
done, one needs to add it::

  conda config --add channels conda-forge

If you just want to run sequential simulations and/or analyze the results of
simulations, you can just install the fluidsim package::

  conda install fluidsim

For parallel simulations using MPI, let's create a dedicated environment::

  conda create -n env_fluidsim ipython fluidsim "fluidfft[build=mpi*]" "h5py[build=mpi*]"

The environment can then be activated with ``conda activate env_fluidsim``.

Build/install using pip
~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing with a recent version of pip so you might want to
run ``pip install pip -U`` before anything (if you use conda, ``conda update
pip``). Then, fluidsim can be installed with::

  pip install fluidsim

However, one have to note that (i) fluidsim builds are sensible to environment
variables (see below) and (ii) fluidsim can optionally use
`fluidfft <http://fluidfft.readthedocs.io>`_ for pseudospectral solvers.
Fluidsim and fluidfft can be both installed with the command::

  pip install fluidsim[fft]

Moreover, fluidfft builds can also be tweaked so you could have a look at
`fluidfft documentation
<http://fluidfft.readthedocs.io/en/latest/install.html>`_.

.. _env_vars:

Environment variables and build configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build time configuration
........................

- ``FLUIDDYN_NUM_PROCS_BUILD``

Fluidsim binaries are builds in parallel. This speedups the build process a lot
on most computers. However, it can be a very bad idea on computers with not
enough memory. If you encounter problems, you can force the number of processes
used during the build using the environment variable
``FLUIDDYN_NUM_PROCS_BUILD``::

   export FLUIDDYN_NUM_PROCS_BUILD=2

- Customize compilers to build extensions, if the defaults do not work for you,
  either using the environment variables:

  - ``MPICXX``: for Cython extensions in ``fluidfft`` (default: ``mpicxx``)
  - ``CC``: command to generate object files in ``fluidsim``
  - ``LDSHARED``: command to link and generate shared libraries in ``fluidsim``
  - ``CARCH``: to cross compile (default: ``native``)

- ``DISABLE_PYTHRAN`` disables compilation with Pythran at build time.

- ``FLUIDSIM_TRANSONIC_BACKEND``

  "pythran" by default, it can be set to "python", "numba" or "cython".

You can also configure the installation of fluidsim by creating the file
``~/.fluidsim-site.cfg`` (or ``site.cfg`` in the repo directory) and modify it
to fit your requirements before the installation with pip::

  wget https://foss.heptapod.net/fluiddyn/fluidsim/raw/default/site.cfg.default -O ~/.fluidsim-site.cfg

Runtime configuration
.....................

Fluidsim is also sensitive to the environment variables:

- ``FLUIDSIM_PATH``: path where the simulation results are saved.

  In Unix systems, you can for example put this line in your ``~/.bashrc``::

    export FLUIDSIM_PATH=$HOME/Data

- ``FLUIDDYN_PATH_SCRATCH``: working directory (can be useful on some clusters).

- ``TRANSONIC_COMPILE_JIT``: set this variable to force JIT compilation using
  ``transonic`` while running tests. This is not necessary, but could be useful
  for troubleshooting if simulations freeze. For example::

     TRANSONIC_COMPILE_JIT=1 fluidsim-test -m fluidsim.solvers.sw1l

Warning about re-installing fluidsim and fluidfft with new build options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If fluidsim has already been installed and you want to recompile with new
configuration values, you need to really recompile fluidsim and not just
reinstall an already produced wheel. To do this, use::

  pip install fluidsim --no-binary fluidsim -v

``-v`` toggles the verbose mode of pip so that we see the compilation log and
can check that everything goes well.

.. _pythranrc:

About using Pythran to compile functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We choose to use the Python compiler `Pythran
<https://github.com/serge-sans-paille/pythran>`_ for some functions of the
operators. Our microbenchmarks show that the performances are as good as what
we are able to get with Fortran or C++!

But it implies that building Fluidsim requires a C++11 compiler (for example
GCC 4.9 or clang).

.. note::

  If you don't want to use Pythran and C++ to speedup Fluidsim, you can use the
  environment variable ``DISABLE_PYTHRAN``.

.. warning::

  To reach good performance, we advice to try to put in the file
  ``~/.pythranrc`` the lines (it seems to work well on Linux, see the `Pythran
  documentation <https://pythran.readthedocs.io>`_):

  .. code:: bash

    [pythran]
    complex_hook = True

.. warning::

  The compilation of C++ files produced by Pythran can be long and can consume
  a lot of memory. If you encounter any problems, you can try to use clang (for
  example with ``conda install clangdev``) and to enable its use in the file
  ``~/.pythranrc`` with:

  .. code:: bash

    [compiler]
    CXX=clang++
    CC=clang

MPI simulations and mpi4py!
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fluidsim can use `mpi4py <http://mpi4py.scipy.org>`_ (which depends on a MPI
implementation) for MPI simulations.

.. warning::

    If the system has multiple MPI libraries, it is adviced to explicitly
    mention the MPI command. For instance to use Intel MPI::

      CC=mpiicc pip install mpi4py --no-binary mpi4py

About h5py and HDF5_MPI
~~~~~~~~~~~~~~~~~~~~~~~

FluidSim is able to use h5py built with MPI support.

.. warning::

  Prebuilt installations (for e.g. via h5py wheels) lacks MPI support.
  Most of the time, this is what you want.  However, you can install h5py
  from source and link it to a hdf5 built with MPI support, as follows:

  .. code:: bash

      $ CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-deps --no-binary=h5py h5py
      $ python -c 'import h5py; h5py.run_tests()'

  In some cases you need to set C_INCLUDE_PATH variable before h5py
  installation. For example on Debian stretch:

  .. code:: bash

      $ export C_INCLUDE_PATH=/usr/include/openmpi/
      $ CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-deps --no-binary=h5py h5py

  See the `h5py documentation
  <http://docs.h5py.org/en/latest/build.html>`_ for more details.

Installing from the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  A good base to install Fluidsim from source can be to create and activate a
  conda environment with::

    conda create -y -n env-fluidsim -c conda-forge "fluidfft=*=*openmpi*" pythran clangdev mako
    conda activate env-fluidsim

For fluidsim, we use the revision control software Mercurial and the main
repository is hosted `here <https://foss.heptapod.net/fluiddyn/fluidsim>`_ in
Heptapod. Download the source with something like::

  hg clone https://foss.heptapod.net/fluiddyn/fluidsim

If you are new with Mercurial and Heptapod, you can also read `this short
tutorial
<http://fluiddyn.readthedocs.org/en/latest/mercurial_heptapod.html>`_.

For particular installation setup, copy the default configuration file to
``site.cfg``::

  cp site.cfg.default site.cfg

and modify it to fit your requirements.

Build/install in development mode, by running from the top-level directory::

  cd lib && pip install -e .; cd ..
  pip install -e .

.. note::

  To install from source in a conda environment, it is actually necessary to
  disable the isolated build by running the command ``pip install -e .
  --no-build-isolation``.

To install Fluidsim with all optional dependencies and all capacities::

  pip install -e .[full]

Run the tests!
..............

You can run some unit tests by running ``make tests`` (shortcut for
``fluidsim-test -v``) or ``make tests_mpi`` (shortcut for ``mpirun -np 2
fluidsim-test -v``). Alternatively, you can also run ``pytest`` from the root
directory or from any of the source directories.
