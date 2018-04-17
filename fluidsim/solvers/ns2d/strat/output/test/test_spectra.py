from __future__ import print_function

import unittest
import numpy as np

from fluidsim.solvers.ns2d.strat.output.test import TestNS2DStrat, mpi


class TestSpectra(TestNS2DStrat):
    _tag = "spectra"

    def test_spectra(self):
        """
        Check sum(spectra) * deltak == energy.
        """
        sim = self.sim
        oper = sim.oper
        state_spect = sim.state.state_spect

        rot_fft = state_spect.get_var("rot_fft")
        b_fft = state_spect.get_var("b_fft")
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)

        energyK_fft = (1. / 2) * (
            np.abs(ux_fft.conj() * ux_fft) + np.abs(uy_fft.conj() * uy_fft)
        )
        spectrum1DEK_kx, spectrum1DEK_ky = oper.spectra1D_from_fft(energyK_fft)
        energyK = oper.sum_wavenumbers(energyK_fft)
        self.assertAlmostEqual(spectrum1DEK_kx.sum() * oper.deltakx, energyK)
        self.assertAlmostEqual(spectrum1DEK_ky.sum() * oper.deltaky, energyK)

        try:
            energyA_fft = (1. / 2) * (
                (np.abs(b_fft.conj() * b_fft) / sim.params.N ** 2)
            )
        except ZeroDivisionError:
            pass
        else:
            spectrum1DEA_kx, spectrum1DEA_ky = oper.spectra1D_from_fft(
                energyA_fft
            )
            energyA = oper.sum_wavenumbers(energyA_fft)

            self.assertAlmostEqual(spectrum1DEA_kx.sum() * oper.deltakx, energyA)
            self.assertAlmostEqual(spectrum1DEA_ky.sum() * oper.deltaky, energyA)

    @unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
    def test_plot1d_spectra(self):
        self.module.plot = self.module.plot1d
        self._plot()

    @unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
    def test_plot2d_spectra(self):
        self.module.plot = self.module.plot2d
        self._plot()

    def test_online_plot_spectra(self):
        self._online_plot_saving(*self.dict_results)


if __name__ == "__main__":
    unittest.main()
