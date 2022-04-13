"""Kolmogorov law 3d (:mod:`fluidsim.base.output.kolmo_law3d`)
==============================================================

Provides:

.. autoclass:: KolmoLaw
   :members:
   :private-members:

"""

# import numpy as np
# import h5py

# from fluiddyn.util import mpi

from .base import SpecificOutput


class KolmoLaw(SpecificOutput):
    """Kolmogorov law 3d."""

    _tag = "kolmo_law"

    @classmethod
    def _complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)

    def __init__(self, output):
        params = output.sim.params

        # dict containing rh and rz
        # TODO: complete arrays_1st_time
        arrays_1st_time = {}

        super().__init__(
            output,
            period_save=params.output.periods_save.kolmo_law,
            arrays_1st_time=arrays_1st_time,
        )

    def compute(self):
        """compute the values at one time."""
        # TODO: has to return a dictionnary containing the data for 1 instant
        return {}
