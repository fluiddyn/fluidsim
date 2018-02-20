from __future__ import print_function

import unittest
import numpy as np

from fluidsim.solvers.sw1l.output.test import BaseTestCase, mpi

class TestNS2DStrat(BaseTestCase):
    solver = 'NS2D.strat'
    _tag = 'spect_energy_budg'

    @classmethod
    def setUpClass(cls, init_fields='noise',
                   type_forcing='tcrandom_anisotropic'):
        nh = 32
        super(TestNS2DStrat, cls).setUpClass(nh=nh, init_fields=init_fields,
                                             type_forcing=type_forcing,
                                             HAS_TO_SAVE=True,
                                             forcing_enable=False)

    def test_nonlinear_transfer(self):
        """
        Check sum(transferEK_kx) == 0
        """
        oper = self.sim.oper
        state_phys = self.sim.state.state_phys
        state_spect = self.sim.state.state_spect

        ux = state_phys.get_var('ux')
        uy = state_phys.get_var('uy')
        rot_fft = state_spect.get_var('rot_fft')
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)

        px_ux_fft, py_ux_fft = oper.gradfft_from_fft(ux_fft)
        px_ux = oper.ifft2(px_ux_fft)
        py_ux = oper.ifft2(py_ux_fft)

        px_uy_fft, py_uy_fft = oper.gradfft_from_fft(uy_fft)
        px_uy = oper.ifft2(px_uy_fft)
        py_uy = oper.ifft2(py_uy_fft)

        Fx = -ux*px_ux - uy*(py_ux)
        Fx_fft = oper.fft2(Fx)
        oper.dealiasing(Fx_fft)

        Fy = -ux*px_uy - uy*(py_uy)
        Fy_fft = oper.fft2(Fy)
        oper.dealiasing(Fy_fft)

        transferEK_fft = np.real(ux_fft.conj() * Fx_fft +
                                 ux_fft * Fx_fft.conj() +
                                 uy_fft.conj() * Fy_fft +
                                 uy_fft * Fy_fft.conj()) / 2.

        transferEK_kx, transferEK_ky = oper.spectra1D_from_fft(
            transferEK_fft)
        
        sum_transferEK_kx = transferEK_kx.sum()
        self.assertAlmostEqual(sum_transferEK_kx, 0)


        
if __name__ == '__main__':
    unittest.main()        
