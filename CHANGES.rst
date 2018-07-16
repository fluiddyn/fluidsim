
0.2.2 (2018-07-01)
------------------

- Let fluidfft decides which FFT class to use (dependency fluidfft >= 0.2.4)

0.2.1 (2018-05-24)
------------------

- IPython magic commands (by Ashwin Vishnu).
- Bugfix divergence-free flow and time_stepping in ns3d solvers.

0.2.0 (2018-05-04)
------------------

- Many bugfixes and nicer code (using the Python code formatter Black).
- Faster ns3d solver.
- ns2d.strat + anisotropic forcing (by Miguel Calpe Linares).
- Nicer forcing parameters.

0.1.1
-----

- Better ``phys_fields.plot`` and ``phys_fields.animate`` (by Ashwin Vishnu and
  Miguel Calpe Linares).
- Faster installation (with configuration file).
- Installation without mpi4py.
- Faster time stepping with less memory allocation.
- Much faster ns3d solvers.

0.1.0
-----

- Uses fluidfft and Pythran

0.0.5
-----

- Compatible fluiddyn 0.1.2

0.0.4
-----

- 0D models (predaprey, lorenz)
- Continuous integration, unittests with bitbucket-pipelines

0.0.3a0
-------

Merge with geofluidsim (Ashwin Vishnu Mohanan repository)

- Movies.
- Preprocessing of parameters.
- Less bugs.

0.0.2a1
-------

- Use a cleaner parameter container class (fluiddyn 0.0.8a1).

0.0.2a0
-------

- SetOfVariables inherits from numpy.ndarray.

- The creation of default parameter has been simplified and is done
  by a class function Simul.create_default_params.

0.0.1a
------

- Split the package fluiddyn between one base package and specialized
  packages.
