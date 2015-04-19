
"""InitFieldsSW1L"""

import numpy as np

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields

from fluidsim.solvers.ns2d.init_fields import (
    InitFieldsNoise as InitFieldsNoiseNS2D)

from fluidsim.solvers.ns2d.init_fields import InitFieldsJet, InitFieldsDipole


class InitFieldsNoise(InitFieldsNoiseNS2D):

    def __call__(self):
        rot_fft, ux_fft, uy_fft = self.compute_rotuxuy_fft()
        self.sim.state.init_from_uxuyfft(ux_fft, uy_fft)


class InitFieldsWave(SpecificInitFields):
    tag = 'wave'

    @classmethod
    def _complete_params_with_default(cls, params):
        super(InitFieldsWave, cls)._complete_params_with_default(params)
        params.init_fields._set_child(cls.tag, attribs={
            'eta_max': 1.,
            'ikx': 2})

    def __call__(self):
        oper = self.sim.oper

        ikx = self.sim.params.init_fields.wave.ikx
        eta_max = self.sim.params.init_fields.wave.eta_max

        kx = oper.deltakx * ikx
        eta_fft = np.zeros_like(self.sim.state('eta_fft'))
        cond = np.logical_and(oper.KX == kx, oper.KY == 0.)
        eta_fft[cond] = eta_max
        oper.project_fft_on_realX(eta_fft)

        self.sim.state.init_from_etafft(eta_fft)


class InitFieldsSW1L(InitFieldsBase):
    """Init the fields for the solver SW1L."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""

        InitFieldsBase._complete_info_solver(
            info_solver,
            classes=[InitFieldsNoise, InitFieldsJet,
                     InitFieldsDipole, InitFieldsWave])
