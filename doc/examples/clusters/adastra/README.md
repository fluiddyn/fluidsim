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

Install UV with:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

````{warning}
No file are sourced at login so the user should source `$HOME/.local/bin/env` with

```sh
. $HOME/.local/bin/env
```

to use uv-installed applications.
````

Install and setup Mercurial with:

```sh
uv tool install mercurial --with hg-git --with hg-evolve
uvx hg-setup init
```

Clone the Fluidsim repository in `$HOME/dev`.

```sh
mkdir ~/dev
cd ~/dev
. ~/.bashrc
hg clone https://foss.heptapod.net/fluiddyn/fluidsim
cd ~/dev/fluidsim/doc/examples/clusters/adastra
```

## Create a Python environment and install Fluidsim from source

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
export PYFFTW_LIB_DIR="/opt/cray/pe/fftw/3.3.10.6/x86_genoa/lib"
export PYFFTW_INCLUDE="/opt/cray/pe/fftw/3.3.10.6/x86_genoa/include"

python -m venv ~/venv-fluidsim
. ~/venv-fluidsim/bin/activate
pip install pip -U
python ~/dev/fluidsim/scripts/install-fluidsim-stack-from-source.py --fftw-openmp -v
```
