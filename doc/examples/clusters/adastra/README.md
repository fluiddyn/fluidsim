# Using Fluidsim on Adastra (CINES)

We show in this directory
(<https://foss.heptapod.net/fluiddyn/fluidsim/-/tree/branch/default/doc/examples/clusters/adastra>)
how to use Fluidsim on Adastra. The main documentation for this HPC platform is
[here](https://dci.dci-gitlab.cines.fr/webextranet/index.html). We use modules produced
by [Spack](https://spack.io/).

## Get a login and setup ssh

Get an account on <https://www.edari.fr/>.

Set the alias

```sh
alias sshadastra='ssh -X augier@adastra.cines.fr'
```

## Setup Mercurial and clone fluidsim

Ask authorization to be able to clone the Fluidsim repository from
<https://foss.heptapod.net> as explained
[here](https://dci.dci-gitlab.cines.fr/webextranet/data_storage_and_transfers/index.html#authorizing-an-outbound-connection).

Install and setup Mercurial as explained
[here](https://fluidhowto.readthedocs.io/en/latest/mercurial/install-setup.html). Clone
the Fluidsim repository in `$HOME/dev`.

```{warning}
The file `.bashrc` is not sourced at login so the user should do it
to use pipx-installed applications.
```

```sh
mkdir ~/dev
cd ~/dev
. ~/.bashrc
hg clone https://foss.heptapod.net/fluiddyn/fluidsim
cd ~/dev/fluidsim/doc/examples/clusters/adastra

```

## Setup a virtual environment

Execute the script `setup_venv.sh`.

```sh
./setup_venv.sh
```

```{literalinclude} ./setup_venv.sh
```

Due to a bug in Meson (the build system used by few fluidfft pluggins, see
https://github.com/mesonbuild/meson/pull/13619), we need to complete the installation:

```sh
module purge
module load cpe/23.12
module load craype-x86-genoa
module load PrgEnv-gnu
module load gcc/13.2.0
module load cray-hdf5-parallel cray-fftw
module load cray-python

export LIBRARY_PATH=/opt/cray/pe/fftw/3.3.10.6/x86_genoa/lib
export CFLAGS="-I/opt/cray/pe/fftw/3.3.10.6/x86_genoa/include"

. ~/venv-fluidsim/bin/activate

# --no-build-isolation because of the Meson bug

# because of --no-build-isolation
pip install meson-python ninja fluidfft-builder cython
cd ~/dev
hg clone https://github.com/paugier/meson.git
cd ~/dev/meson
hg up mpi-detection
pip install -e .
cd
#

pip install fluidfft-fftwmpi --no-binary fluidfft-fftwmpi --no-build-isolation --force-reinstall --no-cache-dir --no-deps -v
```

## Install Fluidsim from source

```sh
module purge
module load cpe/23.12
module load craype-x86-genoa
module load PrgEnv-gnu
module load gcc/13.2.0
module load cray-hdf5-parallel cray-fftw
module load cray-python

. ~/venv-fluidsim/bin/activate

cd ~/dev/fluidsim
# update to the wanted commit
pip install . -v -C setup-args=-Dnative=true
```
