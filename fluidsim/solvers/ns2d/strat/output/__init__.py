"""Output (:mod:`fluidsim.solvers.ns2d.strat.output`)
=====================================================

Provides the modules:

.. autosummary::
   :toctree:

   print_stdout
   spatial_means
   spectra
   spect_energy_budget

and the main output class for the ns2d.strat solver:

.. autoclass:: OutputStrat
   :members:
   :private-members:

"""

import numpy as np

from math import radians

from fluidsim.solvers.ns2d.output import Output


class OutputStrat(Output):
    """Output for ns2d.strat solver."""

    def __init__(self, sim):

        super(OutputStrat, self).__init__(sim)

        if self.sim.params.forcing.type.endswith('anisotropic'):
            self.froude_number = self._compute_froude_number()
            self.ratio_omegas = self._compute_ratio_omegas()

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""

        Output._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        base_name_mod = 'fluidsim.solvers.ns2d.strat.output'

        classes.PrintStdOut.module_name = base_name_mod + '.print_stdout'
        classes.PrintStdOut.class_name = 'PrintStdOutNS2DStrat'

        classes.PhysFields.module_name = base_name_mod + '.phys_fields'
        classes.PhysFields.class_name = 'PhysFields2DStrat'

        attribs = {
            'module_name': base_name_mod + '.spectra',
            'class_name': 'SpectraNS2DStrat'}
        classes.Spectra._set_attribs(attribs)

        attribs = {
            'module_name': base_name_mod + '.spatial_means',
            'class_name': 'SpatialMeansNS2DStrat'}
        classes.spatial_means._set_attribs(attribs)

        attribs = {
            'module_name': base_name_mod + '.spect_energy_budget',
            'class_name': 'SpectralEnergyBudgetNS2DStrat'}
        classes.spect_energy_budg._set_attribs(attribs)

        # classes._set_child(
        #     'spatio_temporal_spectra',
        #     attribs={'module_name': base_name_mod + '.spatio_temporal_spectra',
        #              'class_name': 'SpatioTempSpectra'})



    # @staticmethod
    # def _complete_params_with_default(params, info_solver):
    #     """Complete the `params` container (static method)."""
    #     OutputBasePseudoSpectral._complete_params_with_default(
    #         params, info_solver)

    #     params.output.phys_fields.field_to_plot = 'rot'

    def compute_energies_fft(self):
        """Compute the kinetic and potential energy (k)"""
        rot_fft = self.sim.state.state_spect.get_var('rot_fft')
        b_fft = self.sim.state.state_spect.get_var('b_fft')
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        energyK_fft = (np.abs(ux_fft)**2 + np.abs(uy_fft)**2)/2

        if self.sim.params.N == 0:
            energyA_fft = np.zeros_like(energyK_fft)
        else:
            energyA_fft = ((np.abs(b_fft)/self.sim.params.N)**2)/2
        return energyK_fft, energyA_fft

    def compute_energies2_fft(self):
        """Compute the two kinetic energies for u_x and u_y"""
        rot_fft = self.sim.state.state_spect.get_var('rot_fft')
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        energyK_ux_fft = (np.abs(ux_fft)**2)/2
        energyK_uy_fft = (np.abs(uy_fft)**2)/2
        return energyK_ux_fft, energyK_uy_fft

    def compute_energy_fft(self):
        """Compute energy(k)"""
        energyK_fft, energyA_fft = self.compute_energies_fft()
        return energyK_fft + energyA_fft

    def compute_enstrophy_fft(self):
        """Compute enstrophy(k)"""
        rot_fft = self.sim.state.state_spect.get_var('rot_fft')
        return np.abs(rot_fft)**2/2

    def compute_energy(self):
        """Compute the spatially averaged energy."""
        energy_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(energy_fft)

    def compute_energies(self):
        """Compute the kinetic and potential energy"""
        energyK_fft, energyA_fft = self.compute_energies_fft()
        energyK_ux_fft, energyK_uy_fft = self.compute_energies2_fft()
        energyK = self.sum_wavenumbers(energyK_fft)
        energyA = self.sum_wavenumbers(energyA_fft)

        # energyK_ux (##TODO: remove it. not sense)
        energyK_ux = self.sum_wavenumbers(energyK_ux_fft)
        return energyK, energyA, energyK_ux

    def compute_enstrophy(self):
        """Compute the spatially averaged enstrophy."""
        enstrophy_fft = self.compute_enstrophy_fft()
        return self.sum_wavenumbers(enstrophy_fft)

    def _compute_froude_number(self):
        """Compute froude number ONLY for anisotropic forcing."""
        return round(
            np.sin(radians(float(
                self.sim.params.forcing.tcrandom_anisotropic.angle))), 1)

    def _compute_ratio_omegas(self):
        """Compute ratio omegas; R = N * sin(angle)/P**(1/3.)"""
        P = self.sim.params.forcing.forcing_rate
        N = self.sim.params.N
        froude_number = self._compute_froude_number()
        return round(N * froude_number / P**(1./3), 1)

    def _produce_str_describing_attribs_strat(self):
        """
        Produce string describing the parameters froude_number and ratio_omegas.
        #TODO: not the best way to produce string. 
        """
        str_froude_number = str(self._compute_froude_number())
        str_ratio_omegas = str(self._compute_ratio_omegas())
        
        if '.' in str_froude_number:
            str_froude_number = str_froude_number.split('.')[0] + \
                                str_froude_number.split('.')[1]
        if str_froude_number.endswith('0'):
            str_froude_number = str_froude_number[:-1]
        if '.' in str_ratio_omegas:
            str_ratio_omegas = str_ratio_omegas.split('.')[0] + \
                                str_ratio_omegas.split('.')[1]
        if str_ratio_omegas.endswith('0'):
            str_ratio_omegas = str_ratio_omegas[:-1]
            
        return 'F' + str_froude_number + '_' + 'gamma' + str_ratio_omegas

    def _create_list_for_name_run(self):
        """Creates new name_run for the simulation."""
        list_for_name_run = super(OutputStrat, self)._create_list_for_name_run()
        if self.sim.params.forcing.type.endswith('anisotropic'):
            str_describing_attribs_strat = \
                        self._produce_str_describing_attribs_strat()
            if len(str_describing_attribs_strat) > 0:
                list_for_name_run.append(str_describing_attribs_strat)
        return list_for_name_run
