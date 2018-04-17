from __future__ import print_function

import unittest
import numpy as np

from fluidsim.solvers.ns2d.strat.output.test import TestNS2DStrat, mpi


class TestSpectEnergyBudg(TestNS2DStrat):
    _tag = "spect_energy_budg"

    def test_nonlinear_transfer(self):
        """
        Check sum(transferEK_kx) == 0
        """
        sim = self.sim
        oper = sim.oper
        state_phys = sim.state.state_phys
        state_spect = sim.state.state_spect

        ux = state_phys.get_var("ux")
        uy = state_phys.get_var("uy")
        rot_fft = state_spect.get_var("rot_fft")
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)

        px_ux_fft, py_ux_fft = oper.gradfft_from_fft(ux_fft)
        px_ux = oper.ifft2(px_ux_fft)
        py_ux = oper.ifft2(py_ux_fft)

        px_uy_fft, py_uy_fft = oper.gradfft_from_fft(uy_fft)
        px_uy = oper.ifft2(px_uy_fft)
        py_uy = oper.ifft2(py_uy_fft)

        Fx = -ux * px_ux - uy * (py_ux)
        Fx_fft = oper.fft2(Fx)
        oper.dealiasing(Fx_fft)

        Fy = -ux * px_uy - uy * (py_uy)
        Fy_fft = oper.fft2(Fy)
        oper.dealiasing(Fy_fft)

        transferEK_fft = np.real(ux_fft.conj() * Fx_fft + uy_fft.conj() * Fy_fft)

        transferEK_kx, transferEK_ky = oper.spectra1D_from_fft(transferEK_fft)

        self.assertAlmostEqual(transferEK_kx.sum(), 0)
        self.assertAlmostEqual(transferEK_ky.sum(), 0)

    @unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
    def test_plot_spect_energy_budg(self):
        self.module.plot = self.module.plot
        self._plot()

    def test_online_plot_spect_energy_budg(self):
        self._online_plot_saving(self.dict_results)


if __name__ == "__main__":
    unittest.main()
