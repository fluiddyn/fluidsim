========
FluidSim
========

|release| |docs| |coverage| |travis|

.. |release| image:: https://img.shields.io/pypi/v/fluidsim.svg
   :target: https://pypi.python.org/pypi/fluidsim/
   :alt: Latest version

.. |docs| image:: https://readthedocs.org/projects/fluidsim/badge/?version=latest
   :target: http://fluidsim.readthedocs.org
   :alt: Documentation status

.. |coverage| image:: https://codecov.io/gh/fluiddyn/fluidsim/graph/badge.svg
   :target: https://codecov.io/gh/fluiddyn/fluidsim/
   :alt: Code coverage

.. |travis| image:: https://travis-ci.org/fluiddyn/fluidsim.svg?branch=master
    :target: https://travis-ci.org/fluiddyn/fluidsim

Fluidsim is a framework for studying fluid dynamics with numerical
simulations using Python. It is part of the wider project `FluidDyn
<https://pypi.python.org/pypi/fluiddyn/>`_.

Fluidsim has first been developed by `Pierre Augier
<http://www.legi.grenoble-inp.fr/people/Pierre.Augier/>`_ (CNRS researcher at
`LEGI <http://www.legi.grenoble-inp.fr>`_, Grenoble) at KTH (Stockholm) as a
numerical code to solve fluid equations in a periodic two-dimensional space
with pseudo-spectral methods.

Now, Fluidsim is an object-oriented library to develop solvers (mainly using
pseudo-spectral methods) by writing mainly Python code. The result is really
efficient compared to a pure Fortran or C++ code since the time-consuming tasks
are performed by optimized compiled functions.

Fluidsim is a `HPC <https://en.wikipedia.org/wiki/High-performance_computing>`_
code written mostly in Python. It uses the library `fluidfft
<http://fluidfft.readthedocs.io>`_ to use very efficient FFT
libraries. Fluidfft is written in C++, Cython and python. Fluidfft and fluidsim
take advantage of `Pythran <https://github.com/serge-sans-paille/pythran>`_, a
new static Python compiler which produces very efficient binaries by compiling
Python via C++11.

An advantage of a CFD code written mostly in Python is that to run simulations
and analyze the results, the users communicate (possibly interactively)
together and with the machine with Python, which is nowadays among the best
languages to do these tasks.  Moreover, it is much simpler and faster than with
pure Fortran or C++ codes to add any complicate analysis or to write a modified
solver.

We have created fluidsim to be easy and nice to use and to develop, efficient
and robust.

*Key words and ambitions*: fluid dynamics research with Python (2.7 or
>= 3.4); modular, object-oriented, collaborative, tested and
documented, free and open-source software.

License
-------

FluidDyn is distributed under the CeCILL_ License, a GPL compatible
french license.

.. _CeCILL: http://www.cecill.info/index.en.html

Installation
------------

You can get the source code from `Bitbucket
<https://bitbucket.org/fluiddyn/fluidsim>`__ or from `the Python
Package Index <https://pypi.python.org/pypi/fluidsim/>`__.

The development mode is often useful. From the root directory::

  python setup.py develop

Tests
-----

From the root directory::

  make tests
  make tests_mpi

Or, from the root directory or from any of the "test" directories::

  python -m unittest discover

Alternatively, if you have installed FluidSim using `pip` or `easy_install`::

  fluidsim-test
  mpirun -np 2 fluidsim-test
