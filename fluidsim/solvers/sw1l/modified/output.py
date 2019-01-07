""" """

import numpy as np

from fluidsim.solvers.sw1l.output import OutputBaseSW1L


class OutputSW1LModified(OutputBaseSW1L):
    """subclass :class:`OutputSW1L`"""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        OutputBaseSW1L._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        classes.SpatialMeans.class_name = "SpatialMeansMSW1L"
        classes.SpectralEnergyBudget.class_name = "SpectralEnergyBudgetMSW1L"

    def compute_energies_fft(self):
        ux_fft = self.sim.state.state_spect.get_var("ux_fft")
        uy_fft = self.sim.state.state_spect.get_var("uy_fft")
        eta_fft = self.sim.state.state_spect.get_var("eta_fft")
        energyA_fft = self.sim.params.c2 * np.abs(eta_fft) ** 2 / 2
        energyK_fft = np.abs(ux_fft) ** 2 / 2.0 + np.abs(uy_fft) ** 2 / 2.0
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        uxr_fft, uyr_fft = self.oper.vecfft_from_rotfft(rot_fft)
        energyKr_fft = np.abs(uxr_fft) ** 2 / 2.0 + np.abs(uyr_fft) ** 2 / 2.0
        return energyK_fft, energyA_fft, energyKr_fft

    def compute_energiesKA_fft(self):
        ux_fft = self.sim.state.state_spect.get_var("ux_fft")
        uy_fft = self.sim.state.state_spect.get_var("uy_fft")
        eta_fft = self.sim.state.state_spect.get_var("eta_fft")
        energyA_fft = self.sim.params.c2 * np.abs(eta_fft) ** 2 / 2
        energyK_fft = np.abs(ux_fft) ** 2 / 2.0 + np.abs(uy_fft) ** 2 / 2.0
        return energyK_fft, energyA_fft

    def compute_PV_fft(self):
        # compute Ertel and Charney (QG) potential vorticity
        rot = self.sim.state.get_var("rot")
        eta = self.sim.state.state_phys.get_var("eta")
        ErtelPV_fft = self.oper.fft2((self.sim.params.f + rot) / (1.0 + eta))
        ux_fft = self.sim.state.state_spect.get_var("ux_fft")
        uy_fft = self.sim.state.state_spect.get_var("uy_fft")
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        eta_fft = self.sim.state.state_spect.get_var("eta_fft")
        CharneyPV_fft = rot_fft - self.sim.params.f * eta_fft
        return ErtelPV_fft, CharneyPV_fft
