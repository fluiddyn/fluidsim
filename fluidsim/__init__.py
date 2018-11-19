"""Numerical simulations (:mod:`fluidsim`)
==========================================

.. _simul:

The package :mod:`fluidsim` provides an object-oriented toolkit for doing
numerical simulations of different equations (incompressible Navier-Stokes,
shallow-water, primitive equations, with and without the quasi-geostrophic limit,
adjoin equations, ...)  with different simple methods (pseudo-spectral, finite
differences) and geometries (1D, 2D and 3D periodic, 1 inhomogeneous direction,
...).

The package is organised in four sub-packages:

.. autosummary::
   :toctree:

   util
   base
   operators
   solvers
   magic

"""

from pathlib import Path

from ._version import __version__, get_local_version

from fluiddyn.io import FLUIDSIM_PATH

# has to be done before importing util
path_dir_results = Path(FLUIDSIM_PATH)

from .util.util import (
    import_module_solver_from_key,
    import_simul_class_from_key,
    load_sim_for_plot,
    load_state_phys_file,
    modif_resolution_from_dir,
    modif_resolution_all_dir,
    load_for_restart,
)

from .base.params import load_params_simul

# clean up
from . import util

del util


__all__ = [
    "__version__",
    "get_local_version",
    "path_dir_results",
    "import_module_solver_from_key",
    "import_simul_class_from_key",
    "load_sim_for_plot",
    "load_state_phys_file",
    "modif_resolution_from_dir",
    "modif_resolution_all_dir",
    "load_params_simul",
    "load_for_restart",
]
