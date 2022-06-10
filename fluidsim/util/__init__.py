"""Utilities for FluidSim
=========================

Provides:

.. autosummary::
   :toctree:

   util
   testing
   console
   scripts
   mini_oper_modif_resol

.. autofunction:: load_sim_for_plot

.. autofunction:: load_state_phys_file

.. autofunction:: load_for_restart

.. autofunction:: load_params_simul

.. autofunction:: modif_resolution_from_dir

.. autofunction:: modif_resolution_from_dir_memory_efficient

.. autofunction:: times_start_last_from_path

.. autofunction:: ensure_radians

.. autofunction:: get_mean_values_from_path

.. autofunction:: get_dataframe_from_paths

.. autofunction:: get_last_estimated_remaining_duration

.. autofunction:: open_patient

"""

from .util import (
    load_sim_for_plot,
    load_state_phys_file,
    load_for_restart,
    load_params_simul,
    times_start_last_from_path,
    ensure_radians,
    get_last_estimated_remaining_duration,
    get_mean_values_from_path,
    get_dataframe_from_paths,
    get_memory_usage,
    available_solver_keys,
    import_module_solver_from_key,
    import_simul_class_from_key,
    modif_resolution_from_dir,
    modif_resolution_all_dir,
    modif_resolution_from_dir_memory_efficient,
    open_patient,
)

__all__ = [
    "load_sim_for_plot",
    "load_state_phys_file",
    "load_for_restart",
    "load_params_simul",
    "times_start_last_from_path",
    "ensure_radians",
    "get_last_estimated_remaining_duration",
    "get_mean_values_from_path",
    "get_dataframe_from_paths",
    "get_memory_usage",
    "available_solver_keys",
    "import_module_solver_from_key",
    "import_simul_class_from_key",
    "modif_resolution_from_dir",
    "modif_resolution_all_dir",
    "modif_resolution_from_dir_memory_efficient",
    "open_patient",
]

# deprecated
from .util import times_start_end_from_path
