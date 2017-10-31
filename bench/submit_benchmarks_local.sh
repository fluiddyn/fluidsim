#!/bin/bash
set -e
cmd='fluidsim bench'
options='-d 2 -s ns2d'  # Navier Stokes 2D benchmarks

for np in 2 4 8 12 16 32  # No. of processes
do
  [[ "$np" > $(nproc) ]] && break
  for nh in 32 64 128 256 512 1024  # No. of gridpoints
  do
    let nk_loc=$nh/$np/2
    if [[ nk_loc -gt 1 ]]
    then
        full_cmd="mpirun -np $np $cmd $options $nh"
        echo $full_cmd
        $full_cmd > /dev/null
    fi
  done
done
