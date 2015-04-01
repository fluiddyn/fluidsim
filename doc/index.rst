.. FluidDyn documentation master file, created by
   sphinx-quickstart on Sun Mar  2 12:15:31 2014.

Welcome to FluidSim's documentation!
====================================

FluidSim is a framework for studying fluid dynamics with numerical
simulations using Python. It is part of the wider project `FluidDyn
<https://pypi.python.org/pypi/fluiddyn/>`_.

The package is still in a planning stage so it is still pretty
unstable and many of its planned features have not yet been
implemented.

FluidSim provides object-oriented libraries to develop quite simple
solvers (mainly using pseudo-spectral methods) by writing mainly
Python code. The result should be quite efficient compared to a pure
Fortran or C++ code (to be better quantify). An advantage is that to run
simulations and analyze the results, the users communicate with the
machine through Python (possibly interactively), which is nowadays
among the best languages to do this.  Moreover, it should be much
simpler than with pure Fortran or C++ codes to add any complicate
analysis. For example, writing a solver for adjoin equations should
really be a matter of hours.

At this stage, just few solvers have been written, but at least
FluidSim can solve these equations:

- Incompressible Navier-Stokes equations in a two-dimensional periodic space,

- One-layer shallow-water equations in a two-dimensional periodic space,

- ...

User Guide
----------

.. toctree::
   :maxdepth: 2

   install
   tutorials


Modules Reference
-----------------

.. autosummary::
   :toctree: generated/

   fluidsim.base
   fluidsim.operators
   fluidsim.solvers
   fluidsim.util

Scripts and examples
--------------------

FluidSim also comes with scripts and examples. They are organised in
the following directories:

.. autosummary::
   :toctree: generated/

   examples.launch
   examples.plot_results
   examples.util

More
----

.. toctree::
   :maxdepth: 1

   FluidSim forge in Bitbucket <https://bitbucket.org/fluiddyn/fluidsim>
   FluidSim in PyPI  <https://pypi.python.org/pypi/fluidsim/>
   to_do
   changes


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

