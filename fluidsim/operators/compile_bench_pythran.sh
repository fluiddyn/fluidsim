#! /bin/bash

name=util2d_pythran

pythran $name.py -o ${name}_bench_simple.so

pythran $name.py -march=native -o ${name}_bench_native.so

pythran $name.py -march=native -fopenmp -o ${name}_bench_native_openmp.so

pythran $name.py -fopenmp -o ${name}_bench_openmp.so

pythran $name.py -march=native -DUSE_BOOST_SIMD -o ${name}_bench_simd.so
