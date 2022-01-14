#!/bin/bash

source ../setup_env_base.sh

set -e

conda create -y -n env_$USER python clangdev "blas-devel[build=*openblas]" \
    cython matplotlib pandas psutil ipython scipy pillow scikit-image \
    pythran colorlog

conda activate env_$USER

module load intel openmpi hdf5

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
