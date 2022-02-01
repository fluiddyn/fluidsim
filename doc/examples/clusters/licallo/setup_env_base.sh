module purge

module load intel-gnu8-runtime/19.1.2.254 
module load impi 
module load phdf5 
module load fftw3 

conda activate base

export FLUIDSIM_PATH=/scratch/$USER


