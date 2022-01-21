#!/bin/bash
source ../setup_env_base.sh

set -e

conda env remove --name env_fluidsim -y

conda create -y -n env_fluidsim mamba
conda activate env_fluidsim

mamba install -y \
    ipython scipy "blas-devel[build=*openblas]" \
    matplotlib pandas psutil pillow scikit-image \
    mako clangdev \
    mercurial hg-git hg-evolve

# Using Pythran master > 0.11.0 (possible performance boost)
pip install git+https://github.com/serge-sans-paille/pythran#egg=pythran

pip install hg-fluiddyn transonic setuptools cython pytest

# Better than with conda install because we don't want the fftw conda package
pip install pyfftw

pip install mpi4py --no-binary mpi4py

# to install hdf5 and h5py parallel
export HDF5_DIR=/gpfslocalsup/spack_soft/hdf5/1.12.0/gcc-8.3.1-qj43pa5rathksrgn4sx2ici42tg75nun
CC="mpicc" HDF5_MPI="ON" pip install --no-deps --no-binary=h5py h5py
python -c "import h5py; assert h5py.h5.get_config().mpi, 'h5py not built with MPI support'"
