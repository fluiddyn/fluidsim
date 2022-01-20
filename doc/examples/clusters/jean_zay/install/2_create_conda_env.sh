#!/bin/bash

source ../setup_env_base.sh

set -e

conda env remove --name env_fluidsim -y

#conda install mamba -y    # It takes too much time... try with conda-app
pip install conda-app

# replace conda by mambda
conda create -y -n env_fluidsim -c conda-forge \
    ipython scipy "blas-devel[build=*openblas]" \
    matplotlib pandas psutil pillow scikit-image \
    mako pythran clangdev \
    mercurial hg-git hg-evolve

conda activate env_fluidsim

pip install hg-fluiddyn transonic setuptools cython pytest

# Better than with conda install because we don't want the fftw conda package
pip install pyfftw

pip install mpi4py --no-binary mpi4py

# to install hdf5 and h5py parallel
export HDF5_DIR=/gpfslocalsup/spack_soft/hdf5/1.12.0/gcc-8.3.1-qj43pa5rathksrgn4sx2ici42tg75nun

CC="mpicc" HDF5_MPI="ON" pip install --no-deps --no-binary=h5py h5py
python -c "import h5py; print(h5py.h5.get_config().mpi)"   # Should return True
