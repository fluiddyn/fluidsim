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
import os
import sys

if "FLUIDSIM_PATH" in os.environ:
    os.environ["TRANSONIC_DIR"] = str(
        Path(os.environ["FLUIDSIM_PATH"]) / ".transonic"
    )

_is_testing = False

if any(
    any(test_tool in arg for arg in sys.argv)
    for test_tool in ("pytest", "unittest", "fluidsim-test", "coverage")
):
    _is_testing = True
    from fluiddyn.util import mpi

    mpi.printby0(
        "Fluidsim guesses that it is tested so it "
        "loads the Agg Matplotlib backend."
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _show(*args, **kwargs):
        pass

    plt.show = _show

    if all(
        env_var not in os.environ
        for env_var in ("FLUID_COMPILE_CACHEDJIT", "TRANSONIC_COMPILE_JIT")
    ):
        mpi.printby0("Compilation of jit functions disabled.")
        from transonic import set_compile_jit

        set_compile_jit(False)
    elif "FLUID_COMPILE_CACHEDJIT" in os.environ:
        mpi.printby0(
            "WARNING: FLUID_COMPILE_CACHEDJIT is deprecated, use "
            "TRANSONIC_COMPILE_JIT instead."
        )

from ._version import __version__, get_local_version

from fluiddyn.io import FLUIDSIM_PATH

# has to be done before importing util
try:
    path_dir_results = Path(FLUIDSIM_PATH)
except TypeError:
    # to be able to import for transonic
    path_dir_results = None

from .util import (
    available_solver_keys,
    import_module_solver_from_key,
    import_simul_class_from_key,
    load_sim_for_plot,
    load_state_phys_file,
    modif_resolution_from_dir,
    modif_resolution_all_dir,
    modif_resolution_from_dir_memory_efficient,
    load_for_restart,
    load_params_simul,
)

from .base.params import load_params_simul


# clean up
from . import util

try:
    from .magic import load_ipython_extension
except ImportError:
    pass

del util

# useful alias
load = load_sim_for_plot

__citation__ = r"""
@article{fluiddyn,
doi = {10.5334/jors.237},
year = {2019},
publisher = {Ubiquity Press,  Ltd.},
volume = {7},
author = {Pierre Augier and Ashwin Vishnu Mohanan and Cyrille Bonamy},
title = {{FluidDyn}: A Python Open-Source Framework for Research and Teaching in Fluid Dynamics
    by Simulations,  Experiments and Data Processing},
journal = {Journal of Open Research Software}
}

@article{fluidfft,
doi = {10.5334/jors.238},
year = {2019},
publisher = {Ubiquity Press,  Ltd.},
volume = {7},
author = {Ashwin Vishnu Mohanan and Cyrille Bonamy and Pierre Augier},
title = {{FluidFFT}: Common {API} (C$\mathplus\mathplus$ and Python)
    for Fast Fourier Transform {HPC} Libraries},
journal = {Journal of Open Research Software}
}

@article{fluidsim,
doi = {10.5334/jors.239},
year = {2019},
publisher = {Ubiquity Press,  Ltd.},
volume = {7},
author = {Mohanan, Ashwin Vishnu and Bonamy, Cyrille and Linares, Miguel Calpe and Augier, Pierre},
title = {{FluidSim}: {Modular}, {Object}-{Oriented} {Python} {Package} for
    {High}-{Performance} {CFD} {Simulations}},
journal = {Journal of Open Research Software}
}
"""

__all__ = [
    "__version__",
    "available_solver_keys",
    "get_local_version",
    "path_dir_results",
    "import_module_solver_from_key",
    "import_simul_class_from_key",
    "load_sim_for_plot",
    "load_state_phys_file",
    "modif_resolution_from_dir",
    "modif_resolution_from_dir_memory_efficient",
    "modif_resolution_all_dir",
    "load_params_simul",
    "load_for_restart",
    "__citation__",
]
