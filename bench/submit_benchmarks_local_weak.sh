#!/bin/bash
set -e
cmd='fluidsim bench'
options='-d 2 -s ns2d -t all'  # Navier Stokes 2D benchmarks

n0=128
fluidsim bench --estimate-shapes $n0 -d 2 |
  awk -v cmd="$cmd" -v options="$options" '
    /^ - /{
      np=$2
      n0=$4
      n1=$5
      full_cmd="mpirun -np "np" "cmd" "options" "n0" "n1
      print full_cmd
    }
    ' > ./launcher_local.sh

nohup bash ./launcher_local.sh > ~/.fluiddyn/fluidsim_bench.log 2>&1 &
