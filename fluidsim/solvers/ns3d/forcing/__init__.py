"""Forcing (:mod:`fluidsim.solvers.ns3d.forcing`)
=================================================

.. autoclass:: ForcingTaylorGreen
   :members:

.. autoclass:: ForcingNS3D
   :members:

.. autosummary::
   :toctree:

   watu
   milestone
"""

from math import pi

import numpy as np

from fluidsim.base.forcing.base import ForcingBasePseudoSpectral
from fluidsim.base.forcing.specific import SpecificForcingPseudoSpectralSimple
from fluidsim.base.forcing.anisotropic import (
    TimeCorrelatedRandomPseudoSpectralAnisotropic3D,
)

from .milestone import ForcingMilestone3D
from .watu import ForcingInternalWavesWatuCoriolis


class ForcingTaylorGreen(SpecificForcingPseudoSpectralSimple):
    """Forcing proportional to a large scale Taylor-Green flow.

    fx = F0 * sin(X / Lx) * cos(Y / Ly) * cos(Z / Lz)
    fy = -F0 * cos(X / Lx) * sin(Y / Ly) * cos(Z / Lz)

    """

    tag = "taylor_green"

    @classmethod
    def _complete_params_with_default(cls, params):
        params.forcing.available_types.append(cls.tag)
        params.forcing._set_child(cls.tag, {"amplitude": 1.0})

    @classmethod
    def _modify_sim_repr_maker(cls, sim_repr_maker):
        params = sim_repr_maker.sim.params
        p_taylor_green = params.forcing[cls.tag]
        F0 = p_taylor_green.amplitude
        parameters = {"ampl": F0}
        nu_2 = params.nu_2
        if nu_2 != 0:
            lx = params.oper.Lx
            parameters["Re"] = np.sqrt(F0) * lx ** (3 / 2) / nu_2
        sim_repr_maker.add_parameters(parameters)

    def __init__(self, sim):
        super().__init__(sim)

        amplitude = sim.params.forcing.taylor_green.amplitude
        if not isinstance(amplitude, (float, int)):
            raise NotImplementedError

        X, Y, Z = sim.oper.get_XYZ_loc()
        p_oper = sim.params.oper
        lx, ly, lz = p_oper.Lx, p_oper.Ly, p_oper.Lz
        phase_x = 2 * pi * X / lx
        phase_y = 2 * pi * Y / ly
        phase_z = 2 * pi * Z / lz

        # Definition of the Taylor Green field which forces the flow
        fx = amplitude * np.sin(phase_x) * np.cos(phase_y) * np.cos(phase_z)
        fy = -amplitude * np.cos(phase_x) * np.sin(phase_y) * np.cos(phase_z)

        self.fstate.init_statespect_from(
            vx_fft=sim.oper.fft(fx), vy_fft=sim.oper.fft(fy)
        )

    def compute(self):
        # stationary forcing: nothing to do here!
        pass


class ForcingNS3D(ForcingBasePseudoSpectral):
    """Main forcing class for the ns3d solver."""

    @staticmethod
    def _complete_info_solver(info_solver, classes=None):
        ForcingBasePseudoSpectral._complete_info_solver(
            info_solver,
            [
                ForcingInternalWavesWatuCoriolis,
                ForcingTaylorGreen,
                ForcingMilestone3D,
                TimeCorrelatedRandomPseudoSpectralAnisotropic3D,
            ],
        )
