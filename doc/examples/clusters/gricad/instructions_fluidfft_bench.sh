
source $HOME/venv-fluidsim-guix/bin/activate

mpirun -np `cat machinefile | wc -l` \
    --machinefile machinefile \
    --prefix $1 \
    fluidfft-bench 256 -d 3
