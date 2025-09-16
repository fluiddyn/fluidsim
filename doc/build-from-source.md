# Build from source

## Requirements

To build Fluidsim from source, ones needs:

- A modern Python (>=3.9) with Python headers and `pip`

- A decent amount of RAM (at least few GB available).

- A C++ compiler fully compliant with the C++-11 standard (currently not Intel compilers)

## Get the source

Fluidsim development uses the revision control software [Mercurial] with modern Mercurial
extensions like [Evolve] and [Topic]. The main repository is hosted
[here](https://foss.heptapod.net/fluiddyn/fluidsim) in [Heptapod](https://heptapod.net/).

There are other ways to get the source but we are going here to assume that you can
install [Mercurial]. It can be useful when working with Fluidsim source to
[fully setup Mercurial with these extensions and learn a bit of Mercurial](https://fluidhowto.readthedocs.io/en/latest/mercurial.html).
Then, the Fluidsim repository can be cloned with

```sh

hg clone https://foss.heptapod.net/fluiddyn/fluidsim

```

```{admonition} Why Mercurial/Heptapod and not simply Git/Github?

We consider that modern Mercurial is really great, even better in some aspects
than Git. Moreover, we do not think that it is a good thing that the whole
open-source ecosystem depends on Github, a close-source project owned by
Microsoft.

Thanks to [Octobus](https://octobus.net/) and [Clever
Cloud](https://www.clever-cloud.com) for providing <https://foss.heptapod.net>!

```

## Installing from the repository

### Simple installation from source

We recommend to create a clean virtual environment, for example with:

```sh
cd fluidsim
python3 -m venv .venv
. .venv/bin/activate
pip install pip -U
```

Then, let us use `pip` to install the local project:

```sh
pip install . -v
```

```{note}

`-v` toggles `pip` verbose mode so that we see the compilation log and
can check that everything goes well.

```

Moreover, the build (which uses [Meson]) can be controlled through environment variables
(for the C++ compilation) and options. The particular build options for Fluidsim are
defined in the file `meson.options` which contains:

```{literalinclude} ../meson.options
```

To choose a value different from the default value, one can use this ugly syntax:

```sh
pip install . -v --config-settings=setup-args=-Dtransonic-backend=python
# or
pip install . -v -C setup-args=-Dtransonic-backend=python
```

```{admonition} Let's decompose this syntax!

There are 3 levels:

- `--config-settings` / `-C` is a `pip` option to pass configuration to the PEP
  517 build backend (for Fluidsim [meson-python]).

- `setup-args` is [understood by
  meson-python](https://meson-python.readthedocs.io/en/latest/reference/config-settings.html)

- `transonic-backend` is a Fluidsim build option. But one needs to add the `-D`
  for [Meson]!

```

````{important}

One can activate a performance oriented and not portable build using:

```sh
pip install . -v -C setup-args=-Dnative=true
```

````

````{note}

Recent versions of `pip` allow one to specify different options like so:

```sh
pip install . -v \
  -C setup-args=-Dtransonic-backend=python \
  -C setup-args=-Duse-xsimd=false
```

````

Of course, one can also change values of
[other buildin Meson options](https://meson-python.readthedocs.io/en/latest/reference/config-settings.html).

````{warning}
(compile-args-j2)=

[Meson] builds Fluidsim binaries in parallel. This speedups the build process a
lot on most computers. However, it can be a very bad idea on computers with not
enough memory. One can control the number of processes launched in parallel
with:

```sh
pip install . -v -C compile-args=-j2
```

````

````{Admonition} Another example to set the optimization level

The default optimization level is `-O3`. One can change that with:

```sh
pip install . -v -C setup-args=-Doptimization=2
```

````

```{admonition} Advanced compilation configuration

The environment variables `CC`, `CXX`, `CFLAGS`, `CXXFLAGS` and `LDFLAGS`
are honored.

Note that Fluidsim builds are not sensible to the [`~/.pythranrc` file](pythranrc)!

```

```{admonition} FAQ

- How to know which compilers and compilation flags are used?
  How to check if XSIMD was indeed used?

  One can study the file `build/cp39/compile_commands.json`.

- How to differentiate a native build from a
  regular build to produce binaries usable on other computers?

  By default the produced wheels should be portable. There is the `native`
  build option to target the exact CPU used for compilation.

- How to produce a wheel for other architectures (cross-compilation)?

  ???

```

### Setup a full developer environment with editable installation

Let us first present the tools used for Fluidsim development.

- [PDM] is a modern Python package and dependency manager,

- [Meson] is an open source build system (in particular used by Scipy),

- [Nox] is a command-line tool that automates testing in multiple Python environments,

- [Pytest] is the most popular testing framework for Python,

- [pip] is the official package installer for Python,

- [Pythran] is an ahead of time compiler for a subset of the Python language, with a
  focus on scientific computing,

- [Transonic] is a pure Python package to accelerate modern Python-Numpy code with
  different accelerators (in particular Pythran).

Fluidsim is built with [Meson]. We use [PDM] for Fluidsim development. [Pytest] and [Nox]
are used for testing. We use [Pythran] through [Transonic] to accelerate some numerical
kernels written in Python.

#### Standard Python from Python.org

We present here how one can build Fluidsim from source like the main developers and
users.

##### Install PDM

A first step is to install [PDM] as an external independant application. I (Pierre
Augier) usually use [pipx] for that but
[there are other methods](https://pdm-project.org/latest/#installation).

```sh
python3 -m pip install pipx
pipx install pdm -U
```

Installing in editable mode is a bit particular with Meson, since editable installations
are incompatible with isolated builds, meaning that all build dependencies have to be
installed in the main virtual environment! Fortunatelly, it's not too difficult with
[PDM]. From the root directory of the repository, just run:

```sh
pdm install --no-self
```

This command creates a virtual environment and installs all build and runtime
dependencies. You can then activate this environment and build/install Fluidsim with:

```sh
. .venv/bin/activate
pip install -e . -v --no-build-isolation --no-deps
```

### Conda-based Python with conda-forge and Pixi

One can use [Pixi] to setup a developer environment based on [conda-forge] and compile
from source. From the root directory of Fluidsim repository, just run:

```sh
# TODO: remove this clone after Transonic release
hg clone https://foss.heptapod.net/fluiddyn/transonic/ ../transonic
pixi run install-editable
pixi run fluidsim-test -v
```

Then, `pip` is available and previous commands should work.

## Advice for developers

### Run the tests

You can run some unit tests by running `make tests` (shortcut for `fluidsim-test -v`) or
`make tests_mpi` (shortcut for `mpirun -np 2 fluidsim-test -v`). Alternatively, you can
also run `pytest` from the root directory or from any of the source directories.

(pythranrc)=

### About using Pythran to compile functions

When developing with Pythran, it can be useful to have a `~/.pythranrc` file, with for
example something like (see
[the dedicated section in Pythran documentation](https://pythran.readthedocs.io/en/latest/MANUAL.html#customizing-your-pythranrc)):

```sh

[pythran]
complex_hook = True

[compiler]
CXX=clang++
CC=clang

```

Note however, that Fluidsim build does not take into account this file! Instead there is
a build option `pythran-complex-hook` and one can use environment variables to change the
C++ compilation (performed with [Meson]).

### Set the MESONPY_EDITABLE_VERBOSE mode

It can be useful to set this environment variable when using the editable mode.

```sh
export MESONPY_EDITABLE_VERBOSE=1
```

See
https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html#verbose-mode

## Installation with Apptainer

```{toctree}
---
maxdepth: 1
---
apptainer/README.md
```

[conda-forge]: https://conda-forge.org/
[evolve]: https://www.mercurial-scm.org/doc/evolution/
[mercurial]: https://www.mercurial-scm.org/
[meson]: https://mesonbuild.com
[nox]: https://nox.thea.codes
[pdm]: https://pdm-project.org
[pip]: https://pip.pypa.io
[pipx]: https://github.com/pypa/pipx
[pixi]: https://pixi.sh/
[pytest]: https://docs.pytest.org
[pythran]: https://pythran.readthedocs.io
[topic]: https://www.mercurial-scm.org/doc/evolution/tutorials/topic-tutorial.html
[transonic]: https://transonic.readthedocs.io
