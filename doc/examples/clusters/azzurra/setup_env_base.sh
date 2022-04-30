conda deactivate
module purge

module load gnu8
module load openmpi
module load hdf5/1.12.1-ompi412
module load fftw/3.3.8

# needed to use clang for Pythran
unset CC
unset CXX

export LD_LIBRARY_PATH=$HOME/opt/p3dfft/2.7.5/lib:$HOME/opt/pfft/lib:$LD_LIBRARY_PATH
export FLUIDSIM_PATH=/workspace/$USER

export MPI4PY_RC_THREAD_LEVEL=single
export TRANSONIC_MPI_TIMEOUT=300
