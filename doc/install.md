# Install and configure

First, ensure that you have a recent Python installed, since Fluidsim requires
Python >= 3.9. Some issues regarding the installation of Python and Python
packages are discussed in
[the main documentation of the project](http://fluiddyn.readthedocs.org/en/latest/install.html).

## Installation methods without compilation

### Install using pip without compilation

```{note}

We strongly advice to install Fluidsim in a virtual environment. See the
official guide [Install packages in a virtual environment using pip and
venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

```

Fluidsim can be installed without compilation with `pip`:

```sh
pip install pip -U
pip install fluidsim
```

However, fluidsim requires [fluidfft](http://fluidfft.readthedocs.io) for
pseudospectral solvers. Fluidsim and fluidfft can be both installed without
compilation with the command:

```sh
pip install fluidsim[fft]
```

Moreover, fluidfft works with pluggins to compute FFTs with different methods
(see the [fluidfft
documentation](http://fluidfft.readthedocs.io/en/latest/install.html)). For
example, to install Fluidfft plugins using the FFTW library, one can run (but
it will trigger compilation):

```sh
pip install fluidfft[fftw,fftwmpi,mpi_with_fftw]
```

### Installing the conda-forge packages with conda or mamba

We recommend installing `conda` and `mamba` (using the [conda-forge] channel) with
the [miniforge installer](https://github.com/conda-forge/miniforge).

If you just want to run sequential simulations and/or analyze the results of
simulations, you can just install the fluidsim package:

```sh
mamba install fluidsim
```

For parallel simulations using MPI, let's create a dedicated environment:

```sh
mamba create -n env_fluidsim ipython fluidsim "fluidfft[build=mpi*]" "h5py[build=mpi*]"
```

The environment can then be activated with `conda activate env_fluidsim`.

### Optional dependencies

Fluidsim has 4 sets of optional dependencies, which can be installed with commands
like `pip install fluidsim[fft]` or `pip install fluidsim[fft, mpi]`:

- `fft`: mainly for pseudo spectral solvers using the Fourier basis.

- `mpi`: for parallel computing using [MPI]. `pip install fluidsim[mpi]` installs
  [mpi4py], which requires a local compilation.

- `test`: for testing Fluidsim (can be done without the repository).

- `scipy`

## Environment variables and runtime configuration

Fluidsim is sensitive to environment variables:

- `FLUIDSIM_PATH`: path where the simulation results are saved.

  In Unix systems, you can for example put this line in your `~/.bashrc`:

  ```sh
  export FLUIDSIM_PATH=$HOME/data_fluidsim
  ```

- `FLUIDDYN_PATH_SCRATCH`: working directory (can be useful on some clusters).

## Dependencies with different flavours

## Fluidfft

Fluidsim uses Fluidfft to compute Fourier transforms. Have a look at the
[fluidfft documentation](http://fluidfft.readthedocs.io/en/latest/install.html).

## MPI simulations and mpi4py

Fluidsim can use [mpi4py] (which depends on a [MPI] implementation) for parallel
simulations.

````{warning}

If the system has multiple MPI libraries, it is adviced to explicitly mention the
MPI command. For instance to use Intel MPI:

```sh
CC=mpiicc pip install mpi4py --no-binary mpi4py
```

````

## About h5py and HDF5_MPI

FluidSim is able to use h5py built with MPI support.

````{warning}

Prebuilt installations (for e.g. via h5py wheels) lacks MPI support. Most of the
time, this is what you want. However, you can install h5py from source and link it
to a hdf5 built with MPI support, as follows:

```bash
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-deps --no-binary=h5py h5py
python -c 'import h5py; h5py.run_tests()'
```

In some cases you need to set C_INCLUDE_PATH variable before h5py installation.
For example on Debian stretch:

```bash
export C_INCLUDE_PATH=/usr/include/openmpi/
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-deps --no-binary=h5py h5py
```

See the [h5py documentation](http://docs.h5py.org/en/latest/build.html) for more
details.

````

[conda-forge]: https://conda-forge.org/
[mpi]: https://fr.wikipedia.org/wiki/Message_Passing_Interface
[mpi4py]: https://mpi4py.readthedocs.io/
