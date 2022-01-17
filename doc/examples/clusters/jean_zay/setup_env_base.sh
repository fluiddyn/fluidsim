
alias duh1='du -h --max-depth=1'

module load mercurial/6.0 python/3.8.8 gcc/8.3.1 openmpi/4.1.1 hdf5/1.12.0-mpi
module load fftw/3.3.8-mpi pfft/1.0.8-alpha-mpi p3dfft/2.7.9-mpi

conda init bash

# needed to use clang for Pythran
unset CC
unset CXX

export FLUIDSIM_PATH=$WORK/Fluidsim
