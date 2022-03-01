# Roadmap

## Clean up code and increase [code coverage](https://app.codecov.io/gh/fluiddyn/fluidsim)

Code not covered should be removed, except for very good reasons.

fluidsim-core should be >~95%, with proper unittests.

See <https://foss.heptapod.net/fluiddyn/fluidsim/-/issues/100>

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

- Different diffusivity coefficients for different variables (Prandtl/Schmidt
  numbers), in particular for ns3d.strat.

- cos/sin transform (how? options operators?)

- More 3D solvers:

  - MHD
  - [Gross–Pitaevskii equation](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation)
  - ... (?)

- Linear stability (as in [NS3D](http://yakari.polytechnique.fr/people/deloncle/ns3d.html)

- FFT accelerated with GPU and MPI+GPU (fluidfft)

- Shear as in [Snoopy](https://ipag.osug.fr/~lesurg/snoopy.html) (see
  [#99](https://foss.heptapod.net/fluiddyn/fluidsim/-/issues/99))

- Simul class made of 2 (or n) interacting Simul classes. For example, ns2d +
passive scalar at higher resolution. Or fluid-structure interaction as in
[FLUSI](https://github.com/pseudospectators/FLUSI) (see
[!104](https://foss.heptapod.net/fluiddyn/fluidsim/-/issues/104))

- Particles and Lagrangian dynamics

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
