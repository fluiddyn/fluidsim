======
|logo|
======

|release| |pyversions| |docs| |chat| |coverage| |heptapod_ci| |github_actions|

.. |logo| image:: https://foss.heptapod.net/fluiddyn/fluidsim/raw/branch/default/doc/logo.svg
   :alt: FluidSim

.. |release| image:: https://badge.fury.io/py/fluidsim.svg
   :target: https://pypi.python.org/pypi/fluidsim/
   :alt: Latest version

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/fluidsim.svg
   :alt: Supported Python versions

.. |docs| image:: https://readthedocs.org/projects/fluidsim/badge/?version=latest
   :target: http://fluidsim.readthedocs.org
   :alt: Documentation status

.. |chat| image:: https://img.shields.io/matrix/fluiddyn-users:matrix.org.svg
   :target: https://matrix.to/#/#fluiddyn-users:matrix.org
   :alt: Chat room

.. |coverage| image:: https://codecov.io/gh/fluiddyn/fluidsim/branch/branch%2Fdefault/graph/badge.svg
   :target: https://codecov.io/gh/fluiddyn/fluidsim
   :alt: Code coverage

.. |heptapod_ci| image:: https://foss.heptapod.net/fluiddyn/fluidsim/badges/branch/default/pipeline.svg
   :target: https://foss.heptapod.net/fluiddyn/fluidsim/-/pipelines
   :alt: Heptapod CI

.. |github_actions| image:: https://github.com/fluiddyn/fluidsim/actions/workflows/ci.yml/badge.svg?branch=branch/default
   :target: https://github.com/fluiddyn/fluidsim/actions
   :alt: Github Actions

.. description

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/fluiddyn/fluidsim/branch%2Fdefault?urlpath=lab/tree/doc/ipynb
   :alt: Binder notebook

Fluidsim is an extensible framework for studying fluid dynamics with numerical
simulations using Python. Fluidsim is an object-oriented library to develop
solvers (mainly using pseudo-spectral methods) by writing mainly Python code.
The result is **very efficient** even compared to a pure Fortran or C++ code
since the time-consuming tasks are performed by optimized compiled functions.

**Documentation**: https://fluidsim.readthedocs.io

Getting started
---------------

To try fluidsim without installation: |binder|

For a **basic installation** it should be sufficient to run::

  pip install fluidsim

or with conda::

  conda install -c conda-forge fluidsim

Much more detailed instructions are given in `the documentation
<https://fluidsim.readthedocs.io/en/latest/install.html>`__.

How does it work?
-----------------

Fluidsim is a `HPC <https://en.wikipedia.org/wiki/High-performance_computing>`_
code. It is part of the wider project `FluidDyn
<https://pypi.python.org/pypi/fluiddyn/>`_ and its pseudospectral solvers rely
on the library `fluidfft <http://fluidfft.readthedocs.io>`_ to use very
efficient FFT libraries. Fluidfft is written in C++, Cython and Python.
Fluidfft and fluidsim take advantage of `Pythran
<https://github.com/serge-sans-paille/pythran>`_, an ahead-of-time compiler
which produces very efficient binaries by compiling Python via C++11.

An advantage of a CFD code written mostly in Python is that, to run simulations
and analyze the results, the users communicate (possibly interactively)
together and with the machine with Python, which is nowadays among the best
languages to do these tasks. Moreover, it is much simpler and faster than with
pure Fortran or C++ codes to add any complicated analysis or to write a
modified solver. Fluidsim can also be used to extend existing solvers with
Python interfaces such as `Basilisk <http://basilisk.fr>`__.

We have created fluidsim to be **easy and nice to use and to develop**,
**efficient** and **robust**.

*Keywords and ambitions*: fluid dynamics research with Python (>=3.6);
modular, object-oriented, collaborative, tested and documented, free and
open-source software.

License
-------

FluidSim is distributed under the CeCILL_ License, a GPL compatible french
license.

.. _CeCILL: http://www.cecill.info/index.en.html

Metapapers and citations
------------------------

If you use FluidSim to produce scientific articles, please cite our metapapers
presenting the `FluidDyn project
<https://openresearchsoftware.metajnl.com/articles/10.5334/jors.237/>`__,
`FluidFFT
<https://openresearchsoftware.metajnl.com/articles/10.5334/jors.238/>`__, and
`FluidSim
<https://openresearchsoftware.metajnl.com/articles/10.5334/jors.239/>`__:

.. code ::

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
