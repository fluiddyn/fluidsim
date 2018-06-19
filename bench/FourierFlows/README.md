# FourierFlows benchmarking and profiling

See https://github.com/FourierFlows/FourierFlows.jl

- Install Julia: https://julialang.org/downloads/

- In the Julia interpreter:

  ```
  Pkg.add("FourierFlows")
  Pkg.add("StatProfilerHTML")
  ```

- Run:

  ```
  export OMP_NUM_THREADS=1
  julia benchtestFourierFlows.jl
  julia profilingFourierFlows.jl
  ```

  To be compared to:
  ```
  export OMP_NUM_THREADS=1
  fluidsim-bench 1024 -d 2 -s ns2d -it 10
  mpirun -np 4 fluidsim-bench 1024 -d 2 -s ns2d -it 10

  fluidsim-profile 1024 -d 2 -s ns2d -it 10 -v

  ```


## About the profiling

"From what I understand the code spends ~59% of it’s time doing FFT’s (that’s what
A_mul_B! is called for) and 21% doing in-place array multiplications (that’s what
broadcast.jl is called for)."