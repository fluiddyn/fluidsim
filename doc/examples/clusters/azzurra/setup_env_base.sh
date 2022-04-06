conda deactivate
module purge

module load gnu8
module load openmpi
module load hdf5/1.12.1-ompi412
module load fftw/3.3.8

conda activate base

#export LD_LIBRARY_PATH=$HOME/opt/p3dfft/2.7.5/lib:$HOME/opt/pfft/lib:$LD_LIBRARY_PATH
export FLUIDSIM_PATH=/workspace/$USER







