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

"""

from .util import (
    times_start_last_from_path,
    ensure_radians,
    get_last_estimated_remaining_duration,
)

# deprecated
from .util import times_start_end_from_path
