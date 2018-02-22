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
        nh = 480
        super(TestNS2DStrat, cls).setUpClass(nh=nh, init_fields=init_fields,
                                             type_forcing=type_forcing,
                                             HAS_TO_SAVE=False,
                                             forcing_enable=False)

    def test_nonlinear_transfer(self):
        """
        Check sum(transferEK_kx) == 0
        """
        sim = self.sim
        oper = sim.oper
        state_phys = sim.state.state_phys
        state_spect = sim.state.state_spect

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
                                 uy_fft.conj() * Fy_fft)
        
        transferEK_kx, transferEK_ky = oper.spectra1D_from_fft(
            transferEK_fft)
        
        self.assertAlmostEqual(transferEK_kx.sum(), 0)
        self.assertAlmostEqual(transferEK_ky.sum(), 0)

    def test_spectra(self):
        """
        Check sum(spectra) * deltak == energy.
        """
        sim = self.sim
        oper = sim.oper
        state_spect = sim.state.state_spect

        rot_fft = state_spect.get_var('rot_fft')
        b_fft = state_spect.get_var('b_fft')
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
        
        energyK_fft = (1. / 2) * (np.abs(ux_fft.conj() * ux_fft) +
                                  np.abs(uy_fft.conj() * uy_fft))
        spectrum1DEK_kx, spectrum1DEK_ky = oper.spectra1D_from_fft(energyK_fft)
        energyK = oper.sum_wavenumbers(energyK_fft)
        self.assertAlmostEqual(spectrum1DEK_kx.sum() * oper.deltakx, energyK)
        self.assertAlmostEqual(spectrum1DEK_ky.sum() * oper.deltaky, energyK)

        try:
            energyA_fft = (1. / 2) * ((
                np.abs(b_fft.conj() * b_fft) / sim.params.N**2))
        except ZeroDivisionError:
            pass
        else:
            spectrum1DEA_kx, spectrum1DEA_ky = oper.spectra1D_from_fft(
                energyA_fft)
            energyA = oper.sum_wavenumbers(energyA_fft)

            self.assertAlmostEqual(
                spectrum1DEA_kx.sum() * oper.deltakx, energyA)
            self.assertAlmostEqual(
                spectrum1DEA_ky.sum() * oper.deltaky, energyA)
        
if __name__ == '__main__':
    unittest.main()        
