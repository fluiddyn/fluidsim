#!/bin/bash

source ../setup_env_base.sh

set -e

# We first remove the environement
conda env remove --name env_fluidsim

# Create and install the usefull modules for fluidsim
conda create --name env_fluidsim python ipython scipy "blas-devel[build=*openblas]" \
    matplotlib pandas psutil pillow scikit-image mako clangdev

# Activate the environement
source activate $HOME/miniconda3/envs/env_fluidsim

# Using Pythran master > 0.11.0 (possible performance boost)
pip install git+https://github.com/serge-sans-paille/pythran#egg=pythran --force-reinstall

pip install setuptools cython pytest black

# Install pyfftw with pip and not conda because we don't want the fftw conda package
pip install pyfftw

pip install mpi4py --no-deps --no-binary mpi4py

# Install hdf5 and h5py parallel (giving the hdf5 library path obtained with module show)
CC="mpiicc" HDF5_MPI="ON" HDF5_DIR=/opt/ohpc/pub/oca/apps/intel2020u2-gnu8/impi/phdf5/1.12.0/ \
    pip install --no-binary=h5py h5py --no-build-isolation --no-deps

# Test if h5py has been installed correctly
python -c "import h5py; assert h5py.h5.get_config().mpi, 'h5py not built with MPI support'"

pip install hg+https://foss.heptapod.net/fluiddyn/fluiddyn#egg=fluiddyn
pip install hg+https://foss.heptapod.net/fluiddyn/transonic#egg=transonic
