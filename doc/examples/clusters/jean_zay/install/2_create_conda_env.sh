#!/bin/bash

source ../setup_env_base.sh

set -e

conda env remove --name env_fluidsim

conda create -n env_fluidsim \
    ipython python=3.8.8 scipy "blas-devel[build=*openblas]" \
    matplotlib pandas psutil pillow scikit-image mako \
    clangdev pythran \
    mercurial hg-git hg-evolve

conda activate env_fluidsim

# Using Pythran master > 0.11.0 (possible performance boost)
pip install git+https://github.com/serge-sans-paille/pythran#egg=pythran --force-reinstall

pip install hg-fluiddyn setuptools cython pytest

pip install pyfftw  # Better than with conda install because we don't want the fftw conda package

pip install mpi4py --no-binary mpi4py

# Install hdf5 and h5py parallel (giving the hdf5 library path obtained with module show)
HDF5_DIR=/gpfslocalsup/spack_soft/hdf5/1.12.0/gcc-8.3.1-qj43pa5rathksrgn4sx2ici42tg75nun
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/gpfslocalsup/spack_soft/hdf5/1.12.0/gcc-8.3.1-qj43pa5rathksrgn4sx2ici42tg75nun pip install --no-deps --no-binary=h5py --force-reinstall h5py
python -c "import h5py; assert h5py.h5.get_config().mpi, 'h5py not built with MPI support'"

pip install hg+https://foss.heptapod.net/fluiddyn/fluiddyn#egg=fluiddyn
pip install hg+https://foss.heptapod.net/fluiddyn/transonic#egg=transonic
