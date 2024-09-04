#!/usr/bin/env bash
set -e

module purge
module load cpe/23.12
module load craype-x86-genoa
module load PrgEnv-gnu
module load gcc/13.2.0
module load cray-hdf5-parallel cray-fftw
module load cray-python

cd $HOME
python -m venv venv-fluidsim
.  ~/venv-fluidsim/bin/activate
pip install --upgrade pip

# install fluidsim and all dependencies from wheels!
pip install "fluidsim[fft,test]"

# fix/improve few packages (force recompilation)
pip install fluidfft --no-binary fluidfft -C setup-args=-Dnative=true --force-reinstall --no-cache-dir --no-deps -v

CC=mpicc pip install mpi4py --no-binary mpi4py --force-reinstall --no-cache-dir --no-deps -v
CC="mpicc" HDF5_MPI="ON" pip install h5py --no-binary=h5py --force-reinstall --no-cache-dir --no-deps -v

export LIBRARY_PATH=/opt/cray/pe/fftw/3.3.10.6/x86_genoa/lib
export CFLAGS="-I/opt/cray/pe/fftw/3.3.10.6/x86_genoa/include"
export PYFFTW_LIB_DIR="/opt/cray/pe/fftw/3.3.10.6/x86_genoa/lib"

pip install pyfftw --no-binary pyfftw --force-reinstall --no-cache-dir --no-deps -v

# install fluidfft pluggins
pip install fluidfft-fftw --no-binary fluidfft-fftw --force-reinstall --no-cache-dir --no-deps -v
