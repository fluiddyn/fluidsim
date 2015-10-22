""" """

import numpy as np

from fluidsim.base.output import OutputBasePseudoSpectral


class OutputBaseSW1L(OutputBasePseudoSpectral):

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.Output._set_child('classes')
        classes = info_solver.classes.Output.classes

        package = 'fluidsim.solvers.sw1l.output'

        classes._set_child(
            'PrintStdOut',
            attribs={'module_name': package + '.print_stdout',
                     'class_name': 'PrintStdOutSW1L'})

        classes._set_child(
            'PhysFields',
            attribs={'module_name': 'fluidsim.base.output.phys_fields',
                     'class_name': 'PhysFieldsBase2D'})

        classes._set_child(
            'Spectra',
            attribs={'module_name': package + '.spectra',
                     'class_name': 'SpectraSW1L'})

        classes._set_child(
            'SpatialMeans',
            attribs={'module_name': package + '.spatial_means',
                     'class_name': 'SpatialMeansSW1L'})

        attribs = {
            'module_name': package + '.spect_energy_budget',
            'class_name': 'SpectralEnergyBudgetSW1L'}
        classes._set_child('SpectralEnergyBudget', attribs=attribs)

        attribs = {
            'module_name': 'fluidsim.base.output.increments',
            'class_name': 'IncrementsSW1L'}
        classes._set_child('Increments', attribs=attribs)

        attribs = {
            'module_name': 'fluidsim.base.output.prob_dens_func',
            'class_name': 'ProbaDensityFunc'}
        classes._set_child('ProbaDensityFunc', attribs=attribs)

        attribs = {
            'module_name': 'fluidsim.base.output.time_signalsK',
            'class_name': 'TimeSignalsK'}
        classes._set_child('TimeSignalsK', attribs=attribs)

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        OutputBasePseudoSpectral._complete_params_with_default(
            params, info_solver)

        params.output.phys_fields.field_to_plot = 'rot'

    def linear_eigenmode_from_values_1k(self, ux_fft, uy_fft, eta_fft,
                                        kx, ky):
        div_fft = 1j*(kx*ux_fft + ky*uy_fft)
        rot_fft = 1j*(kx*uy_fft - ky*ux_fft)
        q_fft = rot_fft - self.sim.params.f*eta_fft
        k2 = kx**2+ky**2
        ageo_fft = self.sim.params.f*rot_fft/self.sim.params.c2 + k2*eta_fft
        return q_fft, div_fft, ageo_fft

    def omega_from_wavenumber(self, k):
        return np.sqrt(self.sim.params.f**2 + self.sim.params.c2*k**2)

    def compute_enstrophy_fft(self):
        rot_fft = self.sim.state('rot_fft')
        return np.abs(rot_fft)**2/2

    def compute_PV_fft(self):
        """Compute Ertel and Charney (QG) potential vorticity."""
        rot = self.sim.state('rot')
        eta = self.sim.state.state_phys.get_var('eta')
        ErtelPV_fft = self.oper.fft2((self.sim.params.f+rot)/(1.+eta))
        rot_fft = self.sim.state('rot_fft')
        eta_fft = self.sim.state('eta_fft')
        CharneyPV_fft = rot_fft - self.sim.params.f*eta_fft
        return ErtelPV_fft, CharneyPV_fft

    def compute_PE_fft(self):
        ErtelPV_fft, CharneyPV_fft = self.compute_PV_fft()
        return (abs(ErtelPV_fft)**2/2,
                abs(CharneyPV_fft)**2/2)

    def compute_CharneyPE_fft(self):
        # compute Charney (QG) potential vorticity
        rot_fft = self.sim.state('rot_fft')
        eta_fft = self.sim.state('eta_fft')
        CharneyPV_fft = rot_fft - self.sim.params.f*eta_fft
        return abs(CharneyPV_fft)**2/2

    def compute_energies(self):
        energyK_fft, energyA_fft, energyKr_fft = self.compute_energies_fft()
        return (self.sum_wavenumbers(energyK_fft),
                self.sum_wavenumbers(energyA_fft),
                self.sum_wavenumbers(energyKr_fft))

    def compute_energiesKA(self):
        energyK_fft, energyA_fft = self.compute_energiesKA_fft()
        return (self.sum_wavenumbers(energyK_fft),
                self.sum_wavenumbers(energyA_fft))

    def compute_energy(self):
        energyK_fft, energyA_fft = self.compute_energiesKA_fft()
        return (self.sum_wavenumbers(energyK_fft) +
                self.sum_wavenumbers(energyA_fft))

    def compute_enstrophy(self):
        enstrophy_fft = self.compute_enstrophy_fft()
        return self.sum_wavenumbers(enstrophy_fft)

    def compute_lin_energies_fft(self):
        """Compute quadratic energies."""

        ux_fft = self.sim.state('ux_fft')
        uy_fft = self.sim.state('uy_fft')
        eta_fft = self.sim.state('eta_fft')

        q_fft, div_fft, ageo_fft = \
            self.oper.qdafft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        udx_fft, udy_fft = self.oper.vecfft_from_divfft(div_fft)
        energy_dlin_fft = 0.5*(np.abs(udx_fft)**2 + np.abs(udy_fft)**2)

        ugx_fft, ugy_fft, etag_fft = self.oper.uxuyetafft_from_qfft(q_fft)
        energy_glin_fft = 0.5*(np.abs(ugx_fft)**2 + np.abs(ugy_fft)**2 +
                               self.sim.params.c2*np.abs(etag_fft)**2)

        uax_fft, uay_fft, etaa_fft = self.oper.uxuyetafft_from_afft(ageo_fft)
        energy_alin_fft = 0.5*(np.abs(uax_fft)**2 + np.abs(uay_fft)**2 +
                               self.sim.params.c2*np.abs(etaa_fft)**2)

        return energy_glin_fft, energy_dlin_fft, energy_alin_fft


class OutputSW1L(OutputBaseSW1L):

    def compute_energies_fft(self):
        state = self.sim.state
        eta_fft = state('eta_fft')
        energyA_fft = self.sim.params.c2 * np.abs(eta_fft)**2/2
        Jx_fft = state('Jx_fft')
        Jy_fft = state('Jy_fft')
        ux_fft = state('ux_fft')
        uy_fft = state('uy_fft')
        energyK_fft = np.real(Jx_fft.conj()*ux_fft +
                              Jy_fft.conj()*uy_fft)/2

        rot_fft = state('rot_fft')
        uxr_fft, uyr_fft = self.oper.vecfft_from_rotfft(rot_fft)
        rotJ_fft = self.oper.rotfft_from_vecfft(Jx_fft, Jy_fft)
        Jxr_fft, Jyr_fft = self.oper.vecfft_from_rotfft(rotJ_fft)
        energyKr_fft = np.real(Jxr_fft.conj()*uxr_fft +
                               Jyr_fft.conj()*uyr_fft)/2
        return energyK_fft, energyA_fft, energyKr_fft

    def compute_energiesKA_fft(self):
        state = self.sim.state
        eta_fft = state('eta_fft')
        energyA_fft = self.sim.params.c2 * np.abs(eta_fft)**2/2
        Jx_fft = state('Jx_fft')
        Jy_fft = state('Jy_fft')
        ux_fft = state('ux_fft')
        uy_fft = state('uy_fft')
        energyK_fft = np.real(Jx_fft.conj()*ux_fft +
                              Jy_fft.conj()*uy_fft)/2

        return energyK_fft, energyA_fft
