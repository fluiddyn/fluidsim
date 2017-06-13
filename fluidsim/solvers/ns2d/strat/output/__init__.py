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

from fluidsim.solvers.ns2d.output import Output


class OutputStrat(Output):
    """Output for ns2d.strat solver."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""

        Output._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        base_name_mod = 'fluidsim.solvers.ns2d.strat.output'

        classes.PrintStdOut.module_name = base_name_mod + '.print_stdout'
        classes.PrintStdOut.class_name = 'PrintStdOutNS2DStrat'

        classes.PhysFields.class_name = 'PhysFieldsBase2D'

        attribs = {
            'module_name': base_name_mod + '.spectra',
            'class_name': 'SpectraNS2DStrat'}
        classes.Spectra._set_attribs(attribs)

        # classes._set_child(
        #    'Spectra',
        #    attribs={'module_name': base_name_mod + '.spectra',
        #             'class_name': 'SpectraNS2DStrat'})

        attribs = {
            'module_name': base_name_mod + '.spatial_means',
            'class_name': 'SpatialMeansNS2DStrat'}
        classes.spatial_means._set_attribs(attribs)

        # classes._set_child(
        #     'spatial_means',
        #     attribs={'module_name': base_name_mod + '.spatial_means',
        #              'class_name': 'SpatialMeansNS2DStrat'})
        attribs = {
            'module_name': base_name_mod + '.spect_energy_budget',
            'class_name': 'SpectralEnergyBudgetNS2DStrat'}
        classes.spect_energy_budg._set_attribs(attribs)

        # attribs = {
        #     'module_name': base_name_mod + '.spect_energy_budget',
        #     'class_name': 'SpectralEnergyBudgetNS2DStrat'}
        # classes._set_child('spect_energy_budg', attribs=attribs)

        # attribs = {
        #     'module_name': 'fluidsim.base.output.increments',
        #     'class_name': 'Increments'}
        # classes._set_child('increments', attribs=attribs)

    # @staticmethod
    # def _complete_params_with_default(params, info_solver):
    #     """Complete the `params` container (static method)."""
    #     OutputBasePseudoSpectral._complete_params_with_default(
    #         params, info_solver)

    #     params.output.phys_fields.field_to_plot = 'rot'

    def compute_energies_fft(self):
        """Compute the kinetic and potential energy (k)"""
        rot_fft = self.sim.state.state_fft.get_var('rot_fft')
        b_fft = self.sim.state.state_fft.get_var('b_fft')
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        energyK_fft = (np.abs(ux_fft)**2 + np.abs(uy_fft)**2)/2
        energyA_fft = ((np.abs(b_fft)/self.sim.params.N)**2)/2
        return energyK_fft, energyA_fft

    def compute_energies2_fft(self):
        """Compute the two kinetic energies for u_x and u_y"""
        rot_fft = self.sim.state.state_fft.get_var('rot_fft')
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
        rot_fft = self.sim.state.state_fft.get_var('rot_fft')
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
        energyK_ux = self.sum_wavenumbers(energyK_ux_fft)
        return energyK, energyA, energyK_ux

    def compute_enstrophy(self):
        """Compute the spatially averaged enstrophy."""
        enstrophy_fft = self.compute_enstrophy_fft()
        return self.sum_wavenumbers(enstrophy_fft)
