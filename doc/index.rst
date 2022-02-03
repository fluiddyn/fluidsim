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
code written mostly in Python. It uses the library `Fluidfft
<http://fluidfft.readthedocs.io>`_ to use very efficient FFT libraries.
Fluidfft is written in C++, Cython and Python. Fluidfft and fluidsim take
advantage of `Pythran <https://github.com/serge-sans-paille/pythran>`_, a
static Python compiler which produces very efficient binaries by compiling
Python via C++11. Pythran is actually used in Fluidsim through `Transonic
<http://transonic.readthedocs.io>`_, which is a new and cool project for HPC
with Python.

An advantage of a CFD code written mostly in Python is that to run simulations
and analyze the results, the users communicate (possibly interactively)
together and with the machine with Python, which is nowadays among the best
languages to do these tasks.  Moreover, it is much simpler and faster than with
pure Fortran or C++ codes to add any complicate analysis or to write a modified
solver.

We have created fluidsim to be **easy and nice to use and to develop**, highly
**efficient** and **robust**.

Fluidsim is a young package but the list of solvers already implemented (see
:mod:`fluidsim.solvers`) gives a good idea of the versatility of this
framework. However, currently, Fluidsim excels in particular in solving
equations over a periodic space:

* 2d and 3d incompressible Navier-Stokes equations,

* 2d and 3d incompressible Navier-Stokes equations under the Boussinesq
  approximation (with a buoyancy variable),

* 2d and 3d stratified Navier-Stokes equations under the Boussinesq
  approximation with constant Brunt-Väisälä frequency,

* 2d one-layer shallow-water equations + modified versions of these
  equations,

* 2d Föppl-von Kármán equations (elastic thin plate).

Being a framework, Fluidsim can easily be extended in other packages to develop
other solvers (see for example the packages `snek5000
<https://github.com/exabl/snek5000/>`_ and `fluidsim_ocean
<https://foss.heptapod.net/fluiddyn/fluidsim_ocean>`_).


**Metapapers and citations**

If you use FluidSim to produce scientific articles, please cite our metapapers
presenting the `FluidDyn project
<https://openresearchsoftware.metajnl.com/articles/10.5334/jors.237/>`__,
`FluidFFT
<https://openresearchsoftware.metajnl.com/articles/10.5334/jors.238/>`__, and
`FluidSim
<https://openresearchsoftware.metajnl.com/articles/10.5334/jors.239/>`__:


.. code-block :: bibtex

    @article{fluiddyn,
    doi = {10.5334/jors.237},
    year = {2019},
    publisher = {Ubiquity Press,  Ltd.},
    volume = {7},
    author = {Pierre Augier and Ashwin Vishnu Mohanan and Cyrille Bonamy},
    title = {{FluidDyn}: A Python Open-Source Framework for Research and Teaching in Fluid Dynamics
        by Simulations,  Experiments and Data Processing},
    journal = {Journal of Open Research Software}
    }

    @article{fluidfft,
    doi = {10.5334/jors.238},
    year = {2019},
    publisher = {Ubiquity Press,  Ltd.},
    volume = {7},
    author = {Ashwin Vishnu Mohanan and Cyrille Bonamy and Pierre Augier},
    title = {{FluidFFT}: Common {API} (C$\mathplus\mathplus$ and Python)
        for Fast Fourier Transform {HPC} Libraries},
    journal = {Journal of Open Research Software}
    }

    @article{fluidsim,
    doi = {10.5334/jors.239},
    year = {2019},
    publisher = {Ubiquity Press,  Ltd.},
    volume = {7},
    author = {Mohanan, Ashwin Vishnu and Bonamy, Cyrille and Linares, Miguel
        Calpe and Augier, Pierre},
    title = {{FluidSim}: {Modular}, {Object}-{Oriented} {Python} {Package} for
        {High}-{Performance} {CFD} {Simulations}},
    journal = {Journal of Open Research Software}
    }

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   install
   tutorials
   examples
   test_bench_profile
   ipynb/restart_modif_resol
   faq

Modules Reference
-----------------

A pure-Python package ``fluidsim-core`` houses all the abstraction necessary to
define solvers.

.. autosummary::
   :toctree: generated/

   fluidsim_core

The package ``fluidsim`` provides a set of specialized solvers solvers,
supporting classes and functions.

.. autosummary::
   :toctree: generated/

   fluidsim.base
   fluidsim.operators
   fluidsim.solvers
   fluidsim.util
   fluidsim.magic
   fluidsim.extend_simul


.. toctree::
   :maxdepth: 1
   :caption: More

   changes
   Advice for FluidDyn developers <http://fluiddyn.readthedocs.io/en/latest/advice_developers.html>
   to_do
   authors

Links
-----

.. |release| image:: https://img.shields.io/pypi/v/fluidsim.svg
   :target: https://pypi.org/project/fluidsim/
   :alt: Latest version

.. |coverage| image:: https://codecov.io/gh/fluiddyn/fluidsim/branch/default/graph/badge.svg
   :target: https://codecov.io/gh/fluiddyn/fluidsim/
   :alt: Code coverage

- `FluidDyn documentation <http://fluiddyn.readthedocs.io>`_
- `Fluidsim forge on Heptapod <https://foss.heptapod.net/fluiddyn/fluidsim>`_
- Fluidsim in PyPI |release|
- Unittest coverage |coverage|
- FluidDyn user chat room in
  `Riot <https://riot.im/app/#/room/#fluiddyn-users:matrix.org>`_ or
  `Slack <https://fluiddyn.slack.com>`_
- `FluidDyn mailing list <https://www.freelists.org/list/fluiddyn>`_
- `FluidDyn on Twitter <https://twitter.com/pyfluiddyn>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
