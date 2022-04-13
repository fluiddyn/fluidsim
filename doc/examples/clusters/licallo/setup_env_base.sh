module purge

module load intel-gnu8-runtime/19.1.2.254 
module load impi 
module load phdf5 
module load fftw3 

conda activate base

export LD_LIBRARY_PATH=$HOME/opt/p3dfft/2.7.5/lib:$HOME/opt/pfft/lib:$LD_LIBRARY_PATH
export FLUIDSIM_PATH=/scratch/$USER

export TRANSONIC_MPI_TIMEOUT=300
