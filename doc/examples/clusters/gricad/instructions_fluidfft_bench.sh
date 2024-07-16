
mpirun -np `cat machinefile | wc -l` \
    --machinefile machinefile \
    --prefix $1 \
    fluidfft-bench 256 -d 3
