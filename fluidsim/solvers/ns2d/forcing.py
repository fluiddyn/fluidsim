"""Forcing (:mod:`fluidsim.solvers.ns2d.forcing`)
=================================================

.. autoclass:: ForcingNS2D
   :members:

"""
from __future__ import division
from __future__ import print_function

from fluidsim.base.forcing import ForcingBasePseudoSpectral
from fluidsim.base.forcing.anisotropic import TimeCorrelatedRandomPseudoSpectralAnisotropic


class ForcingNS2D(ForcingBasePseudoSpectral):
    """Forcing class for the ns2d strat solver.

    .. inheritance-diagram:: ForcingNS2D

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        classes = [TimeCorrelatedRandomPseudoSpectralAnisotropic]
        ForcingBasePseudoSpectral._complete_info_solver(info_solver, classes)
