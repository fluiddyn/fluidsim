---
myst:
  substitutions:
    coverage: |-
      ```{image} https://codecov.io/gh/fluiddyn/fluidsim/graph/badge.svg?token=dVfssLBgF2
      :alt: Code coverage
      :target: https://codecov.io/gh/fluiddyn/fluidsim/
      ```
    release: |-
      ```{image} https://img.shields.io/pypi/v/fluidsim.svg
      :alt: Latest version
      :target: https://pypi.org/project/fluidsim/
      ```
---

# Fluidsim documentation

## Overview

Fluidsim is a framework for studying fluid dynamics with numerical simulations using
Python. It is part of the wider project [FluidDyn](http://fluiddyn.readthedocs.io).

Fluidsim is an object-oriented library to develop "solvers" (i.e. Python packages solving
equations) by writing mainly Python code. The result is **very efficient** even compared
to a pure Fortran or C++ code since the time-consuming tasks are performed by optimized
compiled functions.

Fluidsim is a [HPC](https://en.wikipedia.org/wiki/High-performance_computing) code
written mostly in Python. It uses the library [Fluidfft](http://fluidfft.readthedocs.io)
to use very efficient FFT libraries. Fluidfft is written in C++, Cython and Python.
Fluidfft and fluidsim take advantage of
[Pythran](https://github.com/serge-sans-paille/pythran), a static Python compiler which
produces very efficient binaries by compiling Python via C++11. Pythran is actually used
in Fluidsim through [Transonic](http://transonic.readthedocs.io), which is a new and cool
project for HPC with Python.

An advantage of a CFD code written mostly in Python is that to run simulations and
analyze the results, the users communicate (possibly interactively) together and with the
machine with Python, which is nowadays among the best languages to do these tasks.
Moreover, it is much simpler and faster than with pure Fortran or C++ codes to add any
complicate analysis or to write a modified solver.

We have created fluidsim to be **easy and nice to use and to develop**, highly
**efficient** and **robust**.

Being a framework, Fluidsim can easily be extended in other packages to develop other
solvers (see for example the packages [snek5000] and [fluidsimfoam]).

The list of solvers implemented using Fluidsim (see {mod}`fluidsim.solvers`, [snek5000]
and [fluidsimfoam]) gives a good idea of the versatility of this framework. The main
Fluidsim package contains mostly solvers solving equations over a periodic space:

- 2d and 3d incompressible Navier-Stokes equations,
- 2d and 3d incompressible Navier-Stokes equations under the Boussinesq approximation
  (with a buoyancy variable),
- 2d and 3d stratified Navier-Stokes equations under the Boussinesq approximation with
  constant Brunt-Väisälä frequency,
- 2d one-layer shallow-water equations + modified versions of these equations,
- 2d Föppl-von Kármán equations (elastic thin plate).

### Metapapers and citations

If you use FluidSim to produce scientific articles, please cite our metapapers presenting
the
[FluidDyn project](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.237/),
[FluidFFT](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.238/), and
[FluidSim](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.239/):

```bibtex
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
```

```{toctree}
---
caption: Get started
maxdepth: 1
---
install
tutorials
examples
build-from-source
install-clusters
faq
```

```{toctree}
---
caption: Utilities
maxdepth: 1
---
test_bench_profile
ipynb/restart_modif_resol
```

## API Reference

A pure-Python package `fluidsim-core` houses all the abstraction necessary to define
solvers.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :caption: API reference fluidsim-core

   fluidsim_core
```

The package `fluidsim` provides a set of specialized solvers solvers, supporting classes
and functions.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :caption: API reference fluidsim

   fluidsim.base
   fluidsim.operators
   fluidsim.solvers
   fluidsim.util
   fluidsim.magic
   fluidsim.extend_simul

```

```{toctree}
---
caption: Fluidsim development
maxdepth: 1
---
changes
authors
Advice for FluidDyn developers <http://fluiddyn.readthedocs.io/en/latest/advice_developers.html>
to_do
roadmap
release_process
```

## Links

- [FluidDyn documentation](http://fluiddyn.readthedocs.io)
- [Fluidsim forge on Heptapod](https://foss.heptapod.net/fluiddyn/fluidsim)
- Fluidsim in PyPI {{ release }}
- Unittest coverage {{ coverage }}
- FluidDyn user chat room in
  [Riot](https://riot.im/app/#/room/#fluiddyn-users:matrix.org) or
  [Slack](https://fluiddyn.slack.com)
- [FluidDyn mailing list](https://www.freelists.org/list/fluiddyn)
- [FluidDyn on Twitter](https://twitter.com/pyfluiddyn)

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

[fluidsimfoam]: https://foss.heptapod.net/fluiddyn/fluidsimfoam
[snek5000]: https://github.com/exabl/snek5000/
