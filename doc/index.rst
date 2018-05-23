.. FluidDyn documentation master file, created by
   sphinx-quickstart on Sun Mar  2 12:15:31 2014.

Fluidsim documentation
======================

Fluidsim is a framework for studying fluid dynamics with numerical
simulations using Python. It is part of the wider project `FluidDyn
<http://fluiddyn.readthedocs.io>`_.

Fluidsim is an object-oriented library to develop solvers (mainly using
pseudo-spectral methods) by writing mainly Python code. The result is **very
efficient** even compared to a pure Fortran or C++ code since the
time-consuming tasks are performed by optimized compiled functions.

Fluidsim is a `HPC <https://en.wikipedia.org/wiki/High-performance_computing>`_
code written mostly in Python. It uses the library `fluidfft
<http://fluidfft.readthedocs.io>`_ to use very efficient FFT
libraries. Fluidfft is written in C++, Cython and Python. Fluidfft and fluidsim
take advantage of `Pythran <https://github.com/serge-sans-paille/pythran>`_, a
static Python compiler which produces very efficient binaries by compiling
Python via C++11.

An advantage of a CFD code written mostly in Python is that to run simulations
and analyze the results, the users communicate (possibly interactively)
together and with the machine with Python, which is nowadays among the best
languages to do these tasks.  Moreover, it is much simpler and faster than with
pure Fortran or C++ codes to add any complicate analysis or to write a modified
solver.

We have created fluidsim to be **easy and nice to use and to develop**, highly
**efficient** and **robust**.

Fluidsim is a young package but the list of solvers already implemented (see
:mod:`fluidsim.solvers`) gives a good idea of the versatility of this framework.
However, currently, fluidsim excels in particular in solving equations over a
periodic space:

  * 2d and 3d incompressible Navier-Stokes equations,

  * 2d and 3d incompressible Navier-Stokes equations under the Boussinesq
    approximation (with a buoyancy variable),

  * 2d and 3d stratified Navier-Stokes equations under the Boussinesq
    approximation with constant Brunt-Väisälä frequency,

  * 2d one-layer shallow-water equations + modified versions of these equations,

  * 2d Föppl-von Kármán equations (elastic thin plate).


User Guide
----------

.. toctree::
   :maxdepth: 1

   install
   tutorials
   examples
   test_bench_profile

Modules Reference
-----------------

.. autosummary::
   :toctree: generated/

   fluidsim.base
   fluidsim.operators
   fluidsim.solvers
   fluidsim.util
   fluidsim.magic


More
----

.. toctree::
   :maxdepth: 1

   changes
   Advice for FluidDyn developers <http://fluiddyn.readthedocs.io/en/latest/advice_developers.html>
   to_do
   authors


Links
-----

.. |release| image:: https://img.shields.io/pypi/v/fluidsim.svg
   :target: https://pypi.org/project/fluidsim/
   :alt: Latest version

.. |coverage| image:: https://codecov.io/bb/fluiddyn/fluidsim/branch/default/graph/badge.svg
   :target: https://codecov.io/bb/fluiddyn/fluidsim/branch/default/
   :alt: Code coverage

.. |travis| image:: https://travis-ci.org/fluiddyn/fluidsim.svg?branch=master
    :target: https://travis-ci.org/fluiddyn/fluidsim

.. |pipelines| image:: https://img.shields.io/bitbucket/pipelines/fluiddyn/fluidsim.svg
   :target: https://bitbucket.org/fluiddyn/fluidsim/addon/pipelines/home#!/


- `FluidDyn documentation <http://fluiddyn.readthedocs.io>`_
- `Fluidsim forge on Bitbucket <https://bitbucket.org/fluiddyn/fluidsim>`_
- Fluidsim in PyPI |release|
- Unittest coverage |coverage|
- Continuous integration with travis |travis|
- Continuous integration with Bitbucket Pipelines |pipelines|
- `FluidDyn user chat room
  <https://riot.im/app/#/room/#fluiddyn-users:matrix.org>`_
- `FluidDyn mailing list <https://www.freelists.org/list/fluiddyn>`_
- `FluidDyn on Twitter <https://twitter.com/pyfluiddyn>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

