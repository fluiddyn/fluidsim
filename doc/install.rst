Installation
============

FluidSim is part of the FluidDyn project.  Some issues regarding the
installation of Python packages are discussed in `the main
documentation of the project
<https://pythonhosted.org/fluiddyn/install.html>`_.

Dependencies
------------

- FFTW3 (and some modules take advantage of libfftw3_mpi, but this one
  is optional) and on the python package pyfftw,

The first thing to do before installing FluidSim is to installed these
libraries (in contrast, the Python packages should automatically be
installed by the installer)!

Optional dependencies
---------------------

- optionally, mpi4py (which depends on a MPI implementation).


Install commands
----------------
  
FluidSim can be installed by running the following commands::

  hg clone https://bitbucket.org/fluiddyn/fluidsim
  cd fluidsim
  python setup.py develop
 
Installation with Pip should also work::

  pip install fluidsim

Run the tests
-------------

You can run some unit tests by running ``make tests`` from the root
directory or ``python -m unittest discover`` from the root directory
or from any of the "test" directories.


