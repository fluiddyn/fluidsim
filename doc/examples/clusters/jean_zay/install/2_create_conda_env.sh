#!/bin/bash

source ../setup_env_base.sh

set -e

conda env remove --name env_fluidsim #-y

# Note: The python version is important for packages compatibility
conda create -n env_fluidsim # python=3.8 #mamba

conda activate env_fluidsim

# you can also use mamba
conda install ipython python=3.8.8 scipy "blas-devel[build=*openblas]" \
    matplotlib pandas psutil pillow scikit-image mako clangdev \
    mercurial hg-git hg-evolve

# Using Pythran master > 0.11.0 (possible performance boost)
#pip install git+https://github.com/serge-sans-paille/pythran#egg=pythran
#pip install pythran
conda install pythran # (Vincent) I don't know why, but it is important to install with conda (and not pip)

#pip install hg-fluiddyn transonic setuptools cython pytest
pip install hg-fluiddyn transonic setuptools cython pytest

# Better than with conda install because we don't want the fftw conda package
pip install pyfftw


pip install mpi4py --no-binary mpi4py

# Install hdf5 and h5py parallel
# (Check the HDF5_DIR with module show)
HDF5_DIR=/gpfslocalsup/spack_soft/hdf5/1.12.0/gcc-8.3.1-qj43pa5rathksrgn4sx2ici42tg75nun
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/gpfslocalsup/spack_soft/hdf5/1.12.0/gcc-8.3.1-qj43pa5rathksrgn4sx2ici42tg75nun pip install --no-deps --no-binary=h5py h5py

python -c "import h5py; assert h5py.h5.get_config().mpi, 'h5py not built with MPI support'"


