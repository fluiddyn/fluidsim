"""Forcing (:mod:`fluidsim.solvers.ns2d.forcing`)
=================================================

.. autoclass:: ForcingNS2D
   :members:

"""


from fluidsim.base.forcing import ForcingBasePseudoSpectral

from fluidsim.base.forcing.specific import (
    InScriptForcingPseudoSpectral,
    Proportional, TimeCorrelatedRandomPseudoSpectral,
    TimeCorrelatedRandomPseudoSpectralAnisotropic)


class ForcingNS2D(ForcingBasePseudoSpectral):
    """Forcing class for the ns2d solver.

    .. inheritance-diagram:: ForcingNS2D

    """
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        classes = [InScriptForcingPseudoSpectral,
                   Proportional, TimeCorrelatedRandomPseudoSpectral,
                   TimeCorrelatedRandomPseudoSpectralAnisotropic]
        ForcingBasePseudoSpectral._complete_info_solver(info_solver, classes)
