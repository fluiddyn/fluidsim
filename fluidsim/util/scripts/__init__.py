"""Utilities for fluidsim scripts (:mod:`fluidsim.util.scripts`)
================================================================

Provides:

.. autosummary::
   :toctree:

    restart
    modif_resolution
    turb_trandom_anisotropic
    ipy_load

.. autofunction:: parse_args

"""

from fluidsim_core.scripts.restart import parse_args

__all__ = ["parse_args"]
