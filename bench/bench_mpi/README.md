# Benchmark MPI with and without infiniband network

It appears that the value of the environment variable `OMPI_MCA_pml` influences how mpi4py is built.
Therefore, to use the infiniband network, mpi4py has to be rebuilt with something like:

```sh
export OMPI_MCA_pml=ucx
pip install mpi4py --force-reinstall --no-binary mpi4py
```

At LEGI, there is a module `env/ib4openmpi` that sets this variable.

The script `submit_legi.py` can use 2 virtual environments that should be built with something like this.

First, get an interactive session on cores with the infiniband network:

```sh
oarsub -I -l "{net='ib' and os='bullseye'}/core=2,walltime=0:10:0"
```

Then,

```sh
rm -rf venv_mpi_ib venv_mpi_noib
# module with infiniband
module purge
/usr/bin/python3 -m venv venv_mpi_ib
. venv_mpi_ib/bin/activate
module load env/ib4openmpi
pip install mpi4py --no-binary mpi4py
deactivate
# module without infiniband
module purge
/usr/bin/python3 -m venv venv_mpi_noib
. venv_mpi_noib/bin/activate
pip install mpi4py --no-binary mpi4py
```
