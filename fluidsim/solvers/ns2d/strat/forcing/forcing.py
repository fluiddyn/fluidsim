"""Forcing (:mod:`fluidsim.solvers.ns2d.strat.forcing`)
=======================================================

.. autoclass:: ForcingNS2DStrat
   :members:

"""

from fluidsim.solvers.ns2d.forcing import ForcingNS2D

from fluidsim.solvers.ns2d.strat.forcing.specific import (
    TimeCorrelatedRandomPseudoSpectralAnisotrop)


class ForcingNS2DStrat(ForcingNS2D):
    """Forcing class for the ns2d.strat solver."""
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        classes = [TimeCorrelatedRandomPseudoSpectralAnisotrop]
        ForcingNS2D._complete_info_solver(info_solver, classes)
