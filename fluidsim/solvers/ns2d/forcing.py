"""Forcing (:mod:`fluidsim.solvers.ns2d.forcing`)
=================================================

.. autoclass:: ForcingNS2D
   :members:

"""

from fluidsim.base.forcing import ForcingBasePseudoSpectral
from fluidsim.base.forcing.anisotropic import (
    TimeCorrelatedRandomPseudoSpectralAnisotropic,
)
from fluidsim.base.forcing.milestone import ForcingMilestone


class ForcingNS2D(ForcingBasePseudoSpectral):
    """Forcing class for the ns2d solver.

    .. inheritance-diagram:: ForcingNS2D

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        classes = [
            TimeCorrelatedRandomPseudoSpectralAnisotropic,
            ForcingMilestone,
        ]
        ForcingBasePseudoSpectral._complete_info_solver(info_solver, classes)

    def compute_coef_ab_normalize(
        self, constant_rate_of, key_forced, f_fft, var_fft, deltat
    ):

        if constant_rate_of not in ["energy", "energyK"]:
            raise ValueError

        if hasattr(self.forcing_maker, "oper_coarse"):
            oper = self.forcing_maker.oper_coarse
        else:
            oper = self.sim.oper

        if key_forced == "rot_fft":
            vx_fft, vy_fft = oper.vecfft_from_rotfft(var_fft)
            fx_fft, fy_fft = oper.vecfft_from_rotfft(f_fft)
        else:
            raise ValueError

        a = deltat / 2 * oper.sum_wavenumbers(abs(fx_fft) ** 2 + abs(fy_fft) ** 2)
        b = oper.sum_wavenumbers(
            (vx_fft.conj() * fx_fft).real + (vy_fft.conj() * fy_fft).real
        )

        return a, b
