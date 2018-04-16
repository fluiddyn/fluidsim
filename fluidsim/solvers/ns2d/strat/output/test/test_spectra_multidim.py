from __future__ import print_function

import unittest
import numpy as np

from fluidsim.solvers.ns2d.strat.output.test import TestNS2DStrat, mpi


class TestSpectraMultiDim(TestNS2DStrat):
    _tag = "spectra_multidim"

    def test_sum_spectrumkykx_equals_energy(self):
        """Check if sum(spectra_multidim) * deltakx * deltaky == energy"""
        sim = self.sim
        oper = sim.oper
        state_spect = sim.state.state_spect

        deltakx = oper.deltakx
        deltaky = oper.deltaky

        rot_fft = state_spect.get_var("rot_fft")
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
        b_fft = state_spect.get_var("b_fft")
        energyK_fft = (1. / 2) * (
            np.abs(ux_fft.conj() * ux_fft) + np.abs(uy_fft.conj() * uy_fft)
        )

        # Check kinetic energy EK
        energyK = oper.sum_wavenumbers(energyK_fft)
        spectrumkykx_EK = oper.compute_spectrum_kykx(energyK_fft)
        self.assertAlmostEqual(spectrumkykx_EK.sum() * deltakx * deltaky, energyK)

        # Check potential energy EA
        try:
            energyA_fft = (1. / 2) * (
                (np.abs(b_fft.conj() * b_fft) / sim.params.N ** 2)
            )
        except ZeroDivisionError:
            pass
        else:
            energyA = oper.sum_wavenumbers(energyA_fft)
            spectrumkykx_EA = oper.compute_spectrum_kykx(energyA_fft)
            self.assertAlmostEqual(
                spectrumkykx_EA.sum() * deltakx * deltaky, energyA
            )

    def test_spectrumkykx_equals_spectrum1D(self, nb_k_compare=10):
        """
        Check:
        np.sum(spectrumkykx, axis=0) * deltaky = spectrum1Dkx
        np.sum(spectrumkykx, axis=1) * deltakx = spectrum1Dky

        Parameters:
        -----------
        nb_k_compare : int
          Number of wavenumbers to compare in the test.
        """
        sim = self.sim
        oper = sim.oper
        state_spect = sim.state.state_spect

        deltakx = oper.deltakx
        deltaky = oper.deltaky

        rot_fft = state_spect.get_var("rot_fft")
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
        b_fft = state_spect.get_var("b_fft")
        energyK_fft = (1. / 2) * (
            np.abs(ux_fft.conj() * ux_fft) + np.abs(uy_fft.conj() * uy_fft)
        )

        # Check kinetic energy EK
        spectrumkykx_EK = oper.compute_spectrum_kykx(energyK_fft)
        spectrumkx_EK_from = np.sum(spectrumkykx_EK * deltaky, axis=0)
        spectrumky_EK_from = np.sum(spectrumkykx_EK * deltakx, axis=1)

        spectrum1DEK_kx, spectrum1DEK_ky = oper.spectra1D_from_fft(energyK_fft)

        diff_kx = []
        diff_ky = []

        for i in range(nb_k_compare):
            diff_kx.append(
                abs(abs(spectrumkx_EK_from[i]) - abs(spectrum1DEK_kx[i]))
            )
            diff_ky.append(
                abs(abs(spectrumky_EK_from[i]) - abs(spectrum1DEK_ky[i]))
            )

        self.assertAlmostEqual(np.mean(diff_kx), 0)
        self.assertAlmostEqual(np.mean(diff_ky), 0)

        # Check potential energy EA
        try:
            energyA_fft = (1. / 2) * (
                (np.abs(b_fft.conj() * b_fft) / sim.params.N ** 2)
            )
        except ZeroDivisionError:
            pass
        else:
            spectrumkykx_EA = oper.compute_spectrum_kykx(energyA_fft)
            spectrumkx_EA_from = np.sum(spectrumkykx_EA * deltaky, axis=0)
            spectrumky_EA_from = np.sum(spectrumkykx_EA * deltakx, axis=1)

            spectrum1DEA_kx, spectrum1DEA_ky = oper.spectra1D_from_fft(
                energyA_fft
            )

            diff_kx = []
            diff_ky = []

            for i in range(nb_k_compare):
                diff_kx.append(
                    abs(abs(spectrumkx_EA_from[i]) - abs(spectrum1DEA_kx[i]))
                )
                diff_ky.append(
                    abs(abs(spectrumky_EA_from[i]) - abs(spectrum1DEA_ky[i]))
                )

            self.assertAlmostEqual(np.mean(diff_kx), 0)
            self.assertAlmostEqual(np.mean(diff_ky), 0)

    @unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
    def test_plot_spectra(self):
        self.module.plot = self.module.plot
        self._plot()


if __name__ == "__main__":
    unittest.main()
