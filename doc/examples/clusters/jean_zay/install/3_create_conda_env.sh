#!/bin/bash

source ../setup_env_base.sh

set -e

#conda env remove --name env_fluidsim
#conda create -y -n env_fluidsim -c python clangdev "blas-devel[build=*openblas]" \
#    cython matplotlib pandas psutil ipython scipy pillow scikit-image \
#    pythran colorlog

conda create -y -n env_fluidsim -c conda-forge pythran clangdev mako

conda activate env_fluidsim

conda install pyfftw

#module load python/3.8.8 gcc/8.3.1 openmpi/4.1.1 hdf5/1.12.0-mpi         # Not sure that these two lines are usefull because the modules are loaded at the begining of the file.
#module load fftw/3.3.8-mpi pfft/1.0.8-alpha-mpi # p3dfft/2.7.9-mpi # For now, p3dfft/2.7.9-mpi cannot be us

pip install setuptools -U
pip install Cython

pip install mpi4py --no-binary mpi4py

pip install pytest

# to install hdf5 and h5py parallel
export HDF5_DIR=/gpfslocalsup/spack_soft/hdf5/1.12.0/gcc-8.3.1-qj43pa5rathksrgn4sx2ici42tg75nun

CC="mpicc" HDF5_MPI="ON" pip install --no-deps --no-binary=h5py h5py
python -c "import h5py; print(h5py.h5.get_config().mpi)"   # Should return True

