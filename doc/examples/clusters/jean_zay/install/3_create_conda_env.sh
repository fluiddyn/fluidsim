#!/bin/bash

source ../setup_env_base.sh

set -e

conda create -y -n env_fluidsim python clangdev "blas-devel[build=*openblas]" \
    cython matplotlib pandas psutil ipython scipy pillow scikit-image \
    pythran colorlog

conda activate env_fluidsim

module load mercurial/6.0 python/3.8.8 gcc/8.3.1 openmpi/4.1.1 hdf5/1.12.0-mpi
module load fftw/3.3.8-mpi pfft/1.0.8-alpha-mpi p3dfft/2.7.9-mpi 

pip install mpi4py

pip install pyfftw
pip install pytest

# to install h5py parallel
export CC=mpicc
export HDF5_MPI="ON"
unset HDF5_LIBDIR
export LDFLAGS="-Wl,--no-as-needed"
pip install h5py --no-binary h5py
unset CC
