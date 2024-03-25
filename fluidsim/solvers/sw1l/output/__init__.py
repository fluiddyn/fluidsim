"""Output SW1L (:mod:`fluidsim.solvers.sw1l.output`)
====================================================

.. autosummary::
   :toctree:

   print_stdout
   spatial_means
   spect_energy_budget
   spectra
   normal_mode

"""

from .base import OutputBaseSW1L, OutputSW1L

__all__ = ["OutputBaseSW1L", "OutputSW1L"]
