# Roadmap

## Clean up code and increase [code coverage](https://app.codecov.io/gh/fluiddyn/fluidsim)

Code not covered should be removed, except for very good reasons.

- fluidsim-core should be >~95%, with proper unittests.

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

- Linear stability (as in [NS3D](NS3D-2.13 (26/05/2014).))

- FFT accelerated with GPU and MPI+GPU (fluidfft)

- Particles and Lagrangian dynamics

- Shear as in [Snoopy](https://ipag.osug.fr/~lesurg/snoopy.html) (see
  [#99](https://foss.heptapod.net/fluiddyn/fluidsim/-/issues/99))

## Long term

- API to dynamically define a solver

"Ability to dynamically and concisely build a solver is what Dedalus is good
at. And performance and batteries-included approach is where FluidSim shines.
Our InfoSolver + Parameters approach is flexible but requires a lot of
boilerplate code. Even today I always need to refer to documentation while
creating a new solver. It must be possible to create intuitive factory classes
which dynamically generate InfoSolver, Parameters, Simul classes for us. We
could refer to some well known design patterns for inspiration." (Ashwin V.
Mohanan)

- Explore use of type hints

Inline or separate *.pyi files? Use MonkeyType or similar to autogenerate type
hints from tests? Some inspiration: [FOSDEM
talk](https://fosdem.org/2022/schedule/event/python_type_safety/) and [this
blog post](https://nskm.xyz/posts/stcmp2/)