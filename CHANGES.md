# Release notes

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

See also the [unreleased changes].

% Type of changes

% ---------------

% Added      Added for new features.

% Changed    Changed for changes in existing functionality.

% Deprecated Deprecated for soon-to-be removed features.

% Removed    Removed for now removed features.

% Fixed      Fixed for any bug fixes.

% Security   Security in case of vulnerabilities.

## [0.8.5] (2025-10-23)

- [!427](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/427) Support for Python 3.14
- [!421](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/421) New time stepping method `RK2_phaseshift_random_split`
- [!415](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/415) `FLUIDSIM_TYPE_FFT2D` and `FLUIDSIM_TYPE_FFT3D` (see <https://fluidfft.readthedocs.io/en/latest/plugins.html>)
- [!408](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/408) `temporal_spectra.py`: improve plot_spectra
- [!407](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/407) Improve isotropic tcrandom forcing
- [!400](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/400) New script `scripts/install-fluidsim-stack-from-source.py`
- and various fixes and small improvements...

## [0.8.4] (2024-11-08)

- Python 3.13 compatibility

## [0.8.3] (2024-08-27)

- New build with Transonic 0.7.2 to fix Windows wheels

## [0.8.2] (2024-08-17)

- Compatibility with Python 3.12, Numpy 2.0 and mpi4py 4.0.
- Few bug fixes and small improvements.
- Installation methods with Apptainer and Guix.

## [0.8.1] (2024-02-12)

- Few fixes for Fluidfft 0.4.0

## [0.8.0] (2024-01-31)

- Build and upload wheels on PyPI with Github Actions!
- Much better CI in foss.heptapod.net and Github Actions.
- Use the [Meson build system](https://mesonbuild.com) via
  [meson-python](https://github.com/mesonbuild/meson-python).
- Development: use PDM, Nox and Pixi.

## [0.7.4] (2023-10-05)

- [!342](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/342) Refactoring
  and improvements spectra ns2d and ns3d.
- [!335](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/335) Improvement
  `fluidsim-ipy-load` which can now take a path as argument
- Code improvements, bug fixes (in particular for movies) and compatibility with new
  Matplotlib

## [0.7.3] (2023-05-24)

- Official support for only Python 3.9, 3.10 and 3.11.
- Few improvements for [Fluidsimfoam](https://foss.heptapod.net/fluiddyn/fluidsimfoam).

## [0.7.2] (2023-01-05)

- New module {mod}`fluidsim_core.output.remaining_clock_time`.

## [0.7.1] (2022-11-30)

- [!325](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/325) Small changes
  in restarts utilities for Snek5000 0.8.0.

## [0.7.0] (2022-11-23)

- [!316](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/316) Interactive
  movies

- [!317](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/317) and
  [!318](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/318)

  - Refactor movie code in fluidsim-core with several improvements and bugfixes
    ({mod}`fluidsim_core.output.movies` and {mod}`fluidsim_core.output.phys_fields`)
  - Movies for Snek5000

- [!319](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/319) Refactor
  restart code in fluidsim-core ({class}`fluidsim_core.scripts.restart.RestarterABC` and
  {class}`fluidsim.util.scripts.restart.Restarter`)

- [!320](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/320) Restart for
  Snek5000 in fluidsim-core

- [!321](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/321) command
  `fluidsim-ipy-load`.

## [0.6.1] (2022-09-07)

- Turbulence models with `extend_simul_class`
  ([!308](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/308), see
  {mod}`fluidsim.base.turb_model`)

- Kolmogorov forcing
  ([!307](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/307), see
  {mod}`fluidsim.base.forcing.kolmogorov`)

- Output {mod}`fluidsim.base.output.horiz_means`
  ([!309](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/309))

- Output {mod}`fluidsim.base.output.cross_corr3d`
  ([!295](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/295))

- Better support for 3d FFT libs based on pencil decompositions
  ([!283](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/283))

- [!289](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/289)

  - File `is_being_advanced.lock` in the result directory during the runs
  - Better handling of signals (`SIGINT`, `SIGTERM` and `SIGUSR2`)
  - `fluidsim-restart` supports idempotent jobs (OAR scheduler)
  - {func}`fluidsim.util.get_dataframe_from_paths` using `sim.output.get_mean_values`
  - {func}`fluidsim.util.get_last_estimated_remaining_duration`
  - `sim.output.spatiotemporal_spectra.get_spectra`

- CI also running on Github Actions
  ([!224](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/224))

- Various fixes (in particular energy steps with `fluidsim-restart`)

- Various plot improvements (in particular `plot_omega_emp` in
  {mod}`fluidsim.base.output.spatiotemporal_spectra`)

## [0.6.0] (2022-02-07)

- New subpackage {mod}`fluidsim.util.scripts` and module
  {mod}`fluidsim.util.scripts.turb_trandom_anisotropic`
  ([!255](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/255)).

- Entry points console_scripts `fluidsim-restart`
  ([!261](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/261)) and
  `fluidsim-modif-resolution`
  ([!263](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/263)).

- Forcing
  {class}`fluidsim.base.forcing.anisotropic.TimeCorrelatedRandomPseudoSpectralAnisotropic`
  (extension for 3d solvers + new parameter
  `params.forcing.tcrandom_anisotropic.delta_angle`)
  ([!247](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/247)).

- New projection functions (toroidal/poloidal) in {mod}`fluidsim.operators.operators3d`
  ([!247](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/247)).

- [! 250](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/250): New
  parameter `params.projection` for ns3d solvers.

  The equations (`ns3d`, `ns3d.strat` and `ns3d.bouss`) can be modified by projecting the
  solutions on the poloidal or toroidal manifolds.

- Faster loading at Python start
  ([!264](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/264))

- Various bugfixes, in particular related to restart.

## [0.5.1] (2021-11-05)

- [!244](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/244): Taylor Green
  forcing for ns3d solvers
- fluidsim-core: change order for the initialization of the parameters: Simul class
  before the subclasses.

## [0.5.0] (2021-09-29)

### Added

- [!200](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/200) : New
  mechanism to easily extend a Simul class (subpackage {mod}`fluidsim.extend_simul`).

- [!201](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/201) : Improve
  FluidSim Core API with a warning and a convenience function

  - Warnings added when `_set_attrib` is called instead of `_set_child` by a InfoSolver
    instance
  - New function `iter_complete_params`

- Output `spatial_means_regions_milestone.py` using {mod}`fluidsim.extend_simul`.

- New options `no_vz_kz0` and `NO_KY0`.

- Spatiotemporal spectra and many improvements for the temporal spectra for ns3d and ns2d
  solvers by Jason Reneuve
  ([!202](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/202), ...)

- Better Burgers1d solvers (by Ashwin Vishnu)

### Changed

- [!200](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/200) : (internal)
  {class}`fluidsim_core.info.InfoSolverCore`: `__init__` now fully initializes the
  instance (calling the method `complete_with_classes`). New keyword argument `only_root`
  to initialize only the root level.
- [!211](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/211) : Replace for
  ns2d solvers the output `frequency_spectra` (nearly not used) by the newer output
  `temporal_spectra` written for ns3d solvers.

### Fixed

- Many bugfixes!

## [0.4.1] (2021-02-02)

Few bugfixes and [!192](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/192)
(temporal spectra for ns3d solvers).

## [0.4.0] (2021-01-11)

- [!186](https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/186): Package split
  into `fluidsim-core` and `fluidsim`

  - Base classes and abstract base classes defined for `params`, `info_solver`, `sim`,
    `output` instances
  - Entry points as a *plugin framework* to register FluidSim solvers

- `base/output/print_stdout.py`: better regularity saving + method `plot_clock_times`

- Able to run bigger simulations (`2034x2034x384`) on the Occigen cluster (in particular
  new function `fluidsim.modif_resolution_from_dir_memory_efficient`)

## [0.3.3] (2020-10-15)

- Bugfixes and optimizations (in particular for ns3d solvers)
- Forcing WATU Coriolis and Milestone for ns3d.strat
- pyproject.toml and isolated build
- Timestepping using phase-shifting for dealiasing
- Improve regularity of saving for some outputs

## [0.3.2] (2019-11-14)

- Bug fixes and Transonic 0.4 compatibility

## [0.3.1] (2019-03-07)

- Windows compatibility
- Only Python code (stop using Cython)
- Improvements ns2d.strat

## [0.3.0] (2019-01-31)

- Drop support for Python 2.7!
- Accelerated by Transonic & Pythran (also time stepping)
- Better setup.py (by Ashwin Vishnu)
- Improvement ns2d.strat (by Miguel Calpe Linares)
- Much better testing (internal, CI, compatibility pytest, coverage 87%)
- Fix several bugs :-)
- New function load_for_restart

## [0.2.2] (2018-07-01)

- Let fluidfft decides which FFT class to use (dependency fluidfft >= 0.2.4)

## [0.2.1] (2018-05-24)

- IPython magic commands (by Ashwin Vishnu).
- Bugfix divergence-free flow and time_stepping in ns3d solvers.

## [0.2.0] (2018-05-04)

- Many bugfixes and nicer code (using the Python code formatter Black).
- Faster ns3d solver.
- ns2d.strat + anisotropic forcing (by Miguel Calpe Linares).
- Nicer forcing parameters.

## 0.1.1

- Better `phys_fields.plot` and `phys_fields.animate` (by Ashwin Vishnu and Miguel Calpe
  Linares).
- Faster installation (with configuration file).
- Installation without mpi4py.
- Faster time stepping with less memory allocation.
- Much faster ns3d solvers.

## 0.1.0

- Uses fluidfft and Pythran

## 0.0.5

- Compatible fluiddyn 0.1.2

## 0.0.4

- 0D models (predaprey, lorenz)
- Continuous integration, unittests with bitbucket-pipelines

## 0.0.3a0

Merge with geofluidsim (Ashwin Vishnu Mohanan repository)

- Movies.
- Preprocessing of parameters.
- Less bugs.

## 0.0.2a1

- Use a cleaner parameter container class (fluiddyn 0.0.8a1).

## 0.0.2a0

- SetOfVariables inherits from numpy.ndarray.
- The creation of default parameter has been simplified and is done by a class function
  Simul.create_default_params.

## 0.0.1a

- Split the package fluiddyn between one base package and specialized packages.

[0.2.0]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.1.1...0.2.0
[0.2.1]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.2.0...0.2.1
[0.2.2]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.2.1...0.2.2
[0.3.0]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.2.2...0.3.0
[0.3.1]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.3.0...0.3.1
[0.3.2]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.3.1...0.3.2
[0.3.3]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.3.2...0.3.3
[0.4.0]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.3.3...0.4.0
[0.4.1]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.4.0...0.4.1
[0.5.0]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.4.1...0.5.0
[0.5.1]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.5.0...0.5.1
[0.6.0]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.5.1...0.6.0
[0.6.1]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.6.0...0.6.1
[0.7.0]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.6.1...0.7.0
[0.7.1]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.7.0...0.7.1
[0.7.2]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.7.1...0.7.2
[0.7.3]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.7.2...0.7.3
[0.7.4]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.7.3...0.7.4
[0.8.0]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.7.4...0.8.0
[0.8.1]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.8.0...0.8.1
[0.8.2]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.8.1...0.8.2
[0.8.3]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.8.2...0.8.3
[0.8.4]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.8.3...0.8.4
[0.8.5]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.8.4...0.8.5
[unreleased changes]: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.8.5...branch%2Fdefault
