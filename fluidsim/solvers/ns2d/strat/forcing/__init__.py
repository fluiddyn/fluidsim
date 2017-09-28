"""Forcing (:mod:`fluidsim.solvers.ns2d.strat.forcing`)
=======================================================

.. autoclass:: ForcingNS2DStrat
   :members:

"""
# from fluidsim.base.forcing import ForcingBasePseudoSpectral
from fluidsim.solvers.ns2d.strat.forcing.base import \
    ForcingBasePseudoSpectralAnisotrop

from fluidsim.base.forcing.specific import (
    Proportional, TimeCorrelatedRandomPseudoSpectral)

from fluidsim.solvers.ns2d.strat.forcing.specific import \
    TimeCorrelatedRandomPseudoSpectralAnisotrop


class ForcingNS2DStrat(ForcingBasePseudoSpectralAnisotrop):
    """Forcing class for the ns2d solver."""
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        classes = [Proportional, TimeCorrelatedRandomPseudoSpectral, \
                   TimeCorrelatedRandomPseudoSpectralAnisotrop]
        ForcingBasePseudoSpectralAnisotrop._complete_info_solver(info_solver,
                                                                 classes)
