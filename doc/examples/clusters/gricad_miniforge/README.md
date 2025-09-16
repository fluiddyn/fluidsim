# Install fluidsim with Miniforge on Dahu

We show in this directory
(<https://foss.heptapod.net/fluiddyn/fluidsim/-/tree/branch/default/doc/examples/clusters/gricad_miniforge>)
how to use Fluidsim with Miniforge on Gricad clusters. The main documentation for this
HPC platform is [here](https://gricad-doc.univ-grenoble-alpes.fr/hpc/).

## Install Miniforge

```sh
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
$HOME/miniforge3/bin/conda init
. .bashrc
conda config --set auto_activate_base false
```

## Install Fluidsim in a conda env

```sh
conda env create --file https://foss.heptapod.net/fluiddyn/fluidsim/-/raw/branch/default/doc/examples/clusters/gricad_miniforge/env-fluidsim-mpi.yml
```

## Submit jobs

```sh
./submit_bench_fluidfft.py
./submit_bench_fluidsim.py
```

For the devel script `submit_devel_bench_fluidsim.py`, one needs to be connected to a
devel login node (`ssh dahu-oar3`).

```sh
./submit_devel_bench_fluidsim.py
```
