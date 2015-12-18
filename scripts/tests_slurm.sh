#!/bin/bash
source /etc/profile
module load anaconda/py27/2.3
export CRAY_ROOTFS=DSL
source $LOCAL_ANACONDA/bin/activate $LOCAL_ANACONDA
export ANACONDA_HOME=$LOCAL_ANACONDA
source activate_python
echo 'PYTHONPATH=',$PYTHONPATH
which python
which conda
echo 'ANACONDA_HOME=',$ANACONDA_HOME
echo 'HOME=',$HOME
echo 'FLUIDSIM=',$FLUIDSIM_PATH,$FLUIDDYN_PATH_SCRATCH
aprun -n 2 python -m unittest discover
#mpirun -np 2 python -m unittest discover
#srun -n 2 python -m unittest discover
source deactivate_python
