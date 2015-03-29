""" """

import numpy as np

from fluidsim.solvers.sw1l.output.output_base import OutputBaseSW1l


class OutputSW1lModified(OutputBaseSW1l):
    """subclass :class:`OutputSW1l`"""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ContainerXML info_solver.

        This is a static method!
        """
        OutputBaseSW1l._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        classes.SpatialMeans.class_name = 'SpatialMeansMSW1l'
        classes.SpectralEnergyBudget.class_name = 'SpectralEnergyBudgetMSW1l'

    def compute_energies_fft(self):
        ux_fft = self.sim.state.state_fft['ux_fft']
        uy_fft = self.sim.state.state_fft['uy_fft']
        eta_fft = self.sim.state.state_fft['eta_fft']
        energyA_fft = self.sim.params.c2 * np.abs(eta_fft)**2/2
        energyK_fft = np.abs(ux_fft)**2/2 + np.abs(uy_fft)**2/2
        rot_fft = self.rotfft_from_vecfft(ux_fft, uy_fft)
        uxr_fft, uyr_fft = self.vecfft_from_rotfft(rot_fft)
        energyKr_fft = np.abs(uxr_fft)**2/2 + np.abs(uyr_fft)**2/2
        return energyK_fft, energyA_fft, energyKr_fft

    def compute_energiesKA_fft(self):
        ux_fft = self.sim.state.state_fft['ux_fft']
        uy_fft = self.sim.state.state_fft['uy_fft']
        eta_fft = self.sim.state.state_fft['eta_fft']
        energyA_fft = self.sim.params.c2 * np.abs(eta_fft)**2/2
        energyK_fft = np.abs(ux_fft)**2/2 + np.abs(uy_fft)**2/2
        return energyK_fft, energyA_fft

    def compute_PV_fft(self):
        # compute Ertel and Charney (QG) potential vorticity
        rot = self.sim.state('rot')
        eta = self.sim.state.state_phys['eta']
        ErtelPV_fft = self.fft2((self.sim.params.f+rot)/(1.+eta))
        ux_fft = self.sim.state.state_fft['ux_fft']
        uy_fft = self.sim.state.state_fft['uy_fft']
        rot_fft = self.rotfft_from_vecfft(ux_fft, uy_fft)
        eta_fft = self.sim.state.state_fft['eta_fft']
        CharneyPV_fft = rot_fft - self.sim.params.f*eta_fft
        return ErtelPV_fft, CharneyPV_fft
