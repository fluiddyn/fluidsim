Changes
=======

All notable changes to this project will be documented in this file.

The format is based on `Keep a
Changelog <https://keepachangelog.com/en/1.0.0/>`__, and this project
adheres to `Semantic
Versioning <https://semver.org/spec/v2.0.0.html>`__.

.. Type of changes
.. ---------------
.. Added      Added for new features.
.. Changed    Changed for changes in existing functionality.
.. Deprecated Deprecated for soon-to-be removed features.
.. Removed    Removed for now removed features.
.. Fixed      Fixed for any bug fixes.
.. Security   Security in case of vulnerabilities.

..
  Unreleased_
  -----------

.. towncrier release notes start


0.7.2_ (2023-01-05)
-------------------

- New module :mod:`fluidsim_core.output.remaining_clock_time`.

0.7.1_ (2022-11-30)
-------------------

- `!325 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/325>`__
  Small changes in restarts utilities for Snek5000 0.8.0.

0.7.0_ (2022-11-23)
-------------------

- `!316 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/316>`__
  Interactive movies

- `!317 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/317>`__
  and `!318 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/318>`__

  - Refactor movie code in fluidsim-core with several improvements and bugfixes
    (:mod:`fluidsim_core.output.movies` and :mod:`fluidsim_core.output.phys_fields`)
  - Movies for Snek5000

- `!319 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/319>`__
  Refactor restart code in fluidsim-core
  (:class:`fluidsim_core.scripts.restart.RestarterABC` and
  :class:`fluidsim.util.scripts.restart.Restarter`)

- `!320 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/320>`__ Restart for Snek5000 in fluidsim-core

- `!321 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/321>`__ command ``fluidsim-ipy-load``.

0.6.1_ (2022-09-07)
-------------------

- Turbulence models with ``extend_simul_class`` (`!308
  <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/308>`__, see
  :mod:`fluidsim.base.turb_model`)

- Kolmogorov forcing (`!307
  <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/307>`__, see
  :mod:`fluidsim.base.forcing.kolmogorov`)

- Output :mod:`fluidsim.base.output.horiz_means` (`!309 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/309>`__)

- Output :mod:`fluidsim.base.output.cross_corr3d` (`!295 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/295>`__)

- Better support for 3d FFT libs based on pencil decompositions (`!283 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/283>`__)

- `!289 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/289>`__

  - File ``is_being_advanced.lock`` in the result directory during the runs
  - Better handling of signals (``SIGINT``, ``SIGTERM`` and ``SIGUSR2``)
  - ``fluidsim-restart`` supports idempotent jobs (OAR scheduler)
  - :func:`fluidsim.util.get_dataframe_from_paths` using ``sim.output.get_mean_values``
  - :func:`fluidsim.util.get_last_estimated_remaining_duration`
  - ``sim.output.spatiotemporal_spectra.get_spectra``

- CI also running on Github Actions (`!224 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/224>`__)

- Various fixes (in particular energy steps with ``fluidsim-restart``)

- Various plot improvements (in particular ``plot_omega_emp`` in :mod:`fluidsim.base.output.spatiotemporal_spectra`)

0.6.0_ (2022-02-07)
-------------------

- New subpackage :mod:`fluidsim.util.scripts` and module
  :mod:`fluidsim.util.scripts.turb_trandom_anisotropic` (`!255
  <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/255>`__).

- Entry points console_scripts ``fluidsim-restart`` (`!261
  <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/261>`__) and
  ``fluidsim-modif-resolution`` (`!263
  <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/263>`__).

- Forcing :class:`fluidsim.base.forcing.anisotropic.TimeCorrelatedRandomPseudoSpectralAnisotropic`
  (extension for 3d solvers + new parameter ``params.forcing.tcrandom_anisotropic.delta_angle``)
  (`!247 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/247>`__).

- New projection functions (toroidal/poloidal) in
  :mod:`fluidsim.operators.operators3d` (`!247
  <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/247>`__).

- `! 250 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/250>`__:
  New parameter ``params.projection`` for ns3d solvers.

  The equations (``ns3d``, ``ns3d.strat`` and ``ns3d.bouss``) can be modified by
  projecting the solutions on the poloidal or toroidal manifolds.

- Faster loading at Python start (`!264
  <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/264>`__)

- Various bugfixes, in particular related to restart.

0.5.1_ (2021-11-05)
-------------------

- `!244 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/244>`__:
  Taylor Green forcing for ns3d solvers
- fluidsim-core: change order for the initialization of the parameters: Simul
  class before the subclasses.

0.5.0_ (2021-09-29)
-------------------

Added
~~~~~

* `!200 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/200>`__ :
  New mechanism to easily extend a Simul class (subpackage
  :mod:`fluidsim.extend_simul`).

* `!201 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/201>`__ :
  Improve FluidSim Core API with a warning and a convenience function

  - Warnings added when ``_set_attrib`` is called instead of ``_set_child`` by
    a InfoSolver instance
  - New function ``iter_complete_params``

* Output ``spatial_means_regions_milestone.py`` using :mod:`fluidsim.extend_simul`.

* New options ``no_vz_kz0`` and ``NO_KY0``.

* Spatiotemporal spectra and many improvements for the temporal spectra for
  ns3d and ns2d solvers by Jason Reneuve (`!202
  <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/202>`__, ...)

* Better Burgers1d solvers (by Ashwin Vishnu)

Changed
~~~~~~~

* `!200 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/200>`__ :
  (internal) :class:`fluidsim_core.info.InfoSolverCore`: ``__init__`` now fully
  initializes the instance (calling the method ``complete_with_classes``). New
  keyword argument ``only_root`` to initialize only the root level.

* `!211 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/211>`__ :
  Replace for ns2d solvers the output ``frequency_spectra`` (nearly not used) by
  the newer output ``temporal_spectra`` written for ns3d solvers.

Fixed
~~~~~

* Many bugfixes!

0.4.1_ (2021-02-02)
-------------------

Few bugfixes and `!192 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/192>`__
(temporal spectra for ns3d solvers).

0.4.0_ (2021-01-11)
-------------------

* `!186 <https://foss.heptapod.net/fluiddyn/fluidsim/-/merge_requests/186>`__: Package split into ``fluidsim-core`` and ``fluidsim``

  - Base classes and abstract base classes defined for ``params``, ``info_solver``, ``sim``, ``output`` instances
  - Entry points as a *plugin framework* to register FluidSim solvers

* ``base/output/print_stdout.py``: better regularity saving + method ``plot_clock_times``

* Able to run bigger simulations (``2034x2034x384``) on the Occigen cluster (in
  particular new function ``fluidsim.modif_resolution_from_dir_memory_efficient``)

0.3.3_ (2020-10-15)
-------------------

- Bugfixes and optimizations (in particular for ns3d solvers)
- Forcing WATU Coriolis and Milestone for ns3d.strat
- pyproject.toml and isolated build
- Timestepping using phase-shifting for dealiasing
- Improve regularity of saving for some outputs

0.3.2_ (2019-11-14)
-------------------

- Bug fixes and Transonic 0.4 compatibility

0.3.1_ (2019-03-07)
-------------------

- Windows compatibility
- Only Python code (stop using Cython)
- Improvements ns2d.strat

0.3.0_ (2019-01-31)
-------------------

- Drop support for Python 2.7!
- Accelerated by Transonic & Pythran (also time stepping)
- Better setup.py (by Ashwin Vishnu)
- Improvement ns2d.strat (by Miguel Calpe Linares)
- Much better testing (internal, CI, compatibility pytest, coverage 87%)
- Fix several bugs :-)
- New function load_for_restart

0.2.2_ (2018-07-01)
-------------------

- Let fluidfft decides which FFT class to use (dependency fluidfft >= 0.2.4)

0.2.1_ (2018-05-24)
-------------------

- IPython magic commands (by Ashwin Vishnu).
- Bugfix divergence-free flow and time_stepping in ns3d solvers.

0.2.0_ (2018-05-04)
-------------------

- Many bugfixes and nicer code (using the Python code formatter Black).
- Faster ns3d solver.
- ns2d.strat + anisotropic forcing (by Miguel Calpe Linares).
- Nicer forcing parameters.

0.1.1
-----

- Better ``phys_fields.plot`` and ``phys_fields.animate`` (by Ashwin Vishnu and
  Miguel Calpe Linares).
- Faster installation (with configuration file).
- Installation without mpi4py.
- Faster time stepping with less memory allocation.
- Much faster ns3d solvers.

0.1.0
-----

- Uses fluidfft and Pythran

0.0.5
-----

- Compatible fluiddyn 0.1.2

0.0.4
-----

- 0D models (predaprey, lorenz)
- Continuous integration, unittests with bitbucket-pipelines

0.0.3a0
-------

Merge with geofluidsim (Ashwin Vishnu Mohanan repository)

- Movies.
- Preprocessing of parameters.
- Less bugs.

0.0.2a1
-------

- Use a cleaner parameter container class (fluiddyn 0.0.8a1).

0.0.2a0
-------

- SetOfVariables inherits from numpy.ndarray.

- The creation of default parameter has been simplified and is done
  by a class function Simul.create_default_params.

0.0.1a
------

- Split the package fluiddyn between one base package and specialized
  packages.

.. _Unreleased: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.7.2...branch%2Fdefault
.. _0.7.2: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.7.1...0.7.2
.. _0.7.1: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.7.0...0.7.1
.. _0.7.0: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.6.1...0.7.0
.. _0.6.1: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.6.0...0.6.1
.. _0.6.0: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.5.1...0.6.0
.. _0.5.1: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.5.0...0.5.1
.. _0.5.0: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.4.1...0.5.0
.. _0.4.1: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.4.0...0.4.1
.. _0.4.0: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.3.3...0.4.0
.. _0.3.3: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.3.2...0.3.3
.. _0.3.2: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.3.1...0.3.2
.. _0.3.1: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.3.0...0.3.1
.. _0.3.0: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.2.2...0.3.0
.. _0.2.2: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.2.1...0.2.2
.. _0.2.1: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.2.0...0.2.1
.. _0.2.0: https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.1.1...0.2.0
