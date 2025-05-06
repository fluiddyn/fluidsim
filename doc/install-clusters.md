# Fluidsim on clusters

Computing clusters are sets of computers used for HPC. Installing on such machines in
order to run very large simulations is particular since

- Performance is key. With very large simulations, differences of few percents in
  performance can lead to important differences of electricity consumption and CO₂
  production.

  With large simulations, a large proportion of elapsed time is spent in crushing numbers
  (concentrated in few functions) and MPI communications. For pseudo-spectral simulations
  based on Fourier transform, the FFT functions and few other numerical kernels have to
  be very efficient. This is achieved by using advanced FFT libraries and by compiling
  with special options like `-march=native` and `-Ofast`.

- Parallelism is done trough MPI with advanced hardware so it's important to use the
  right MPI implementation compiled with the right options.

- The software environment is usually quite different than on more standard machines,
  with quite old operative systems and particular systems to use other software (modules,
  Guix, Spack, ...).

- Computations are launched through a schedulers (like Slurm, OAR, ...) with a launching
  script. In the Fluiddyn project, we tend to avoid writting manually the launching
  scripts (which is IMHO error prone and slow) and prefer to use the `fluiddyn.clusters`
  API, which allows users to launch simulations with simple Python scripts.

We present here few examples of installation methods and launching scripts on different
kinds of clusters:

```{toctree}
---
caption: Examples
maxdepth: 1
---
./examples/clusters/adastra/README.md
./examples/clusters/gricad_guix/README.md
./examples/clusters/gricad_miniforge/README.md
```
