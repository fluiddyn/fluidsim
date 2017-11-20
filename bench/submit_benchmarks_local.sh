#!/bin/bash
set -e
cmd='fluidsim bench'
# options='-d 2 -s ns2d'  # Navier Stokes 2D benchmarks
# gridpoints="32 64 128 256 512 1024" # No. of gridpoints

options='-d 3 -s ns3d -t fft3d.mpi_with_p3dfft'  # Navier Stokes 3D benchmarks
gridpoints="32 64" # No. of gridpoints

for np in 2 4 8 12 16 32  # No. of processes
do
  [[ $np -gt $(nproc) ]] && break
  for nh in $gridpoints
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
