# Roadmap

## Clean up code and increase [code coverage](https://app.codecov.io/gh/fluiddyn/fluidsim)

Code not covered should be removed, except for very good reasons.

- fluidsim-core should be at >95%, with proper unittests.

- code for using third-party programs (Dedalus and Basilisk) should be
  removed from the fluidsim package.

- scripts should be moved outside of the package (in particular
  `fluidsim/solvers/ns2d/strat/find_dissipation.py`)

## Fluidsim-core and fluidsim-... packages for using third-party programs

- snek5000 stable

- fluidsim-core should grow, in particular by moving code from fluidsim and
  snek5000 to fluidsim-core when needed for the implementation of another package
  (in particular fluidsim-openfoam).

- fluidsim-... (OpenFoam, Dedalus, Basilisk, ...)

## Fluidsim

Specialized in pseudo-spectral Fourier.

- Questions about focus:

  - What about cos/sin transforms?

  - What about `fluidsim.solvers.sphere`?

- SVV (Spectral Vanishing Viscosity)

- cos/sin transform (how? options operators?)

- More 3D solvers:

  - MHD
  - [Gross–Pitaevskii equation](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation)
  - ... (?)

- Linear NS (as in [NS3D](NS3D-2.13 (26/05/2014).))

- FFT accelerated with GPU and MPI+GPU (fluidfft)

- Particles and Lagrangian dynamics

- Shear as in [Snoopy](https://ipag.osug.fr/~lesurg/snoopy.html) (see
  [#99](https://foss.heptapod.net/fluiddyn/fluidsim/-/issues/99))
