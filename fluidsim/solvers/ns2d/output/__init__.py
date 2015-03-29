

import numpy as np

from fluidsim.base.output import OutputBasePseudoSpectral


class Output(OutputBasePseudoSpectral):

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ContainerXML info_solver.

        This is a static method!
        """

        OutputBasePseudoSpectral._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        base_name_mod = 'fluidsim.solvers.ns2d.output'

        classes.PrintStdOut.module_name = base_name_mod + '.print_stdout'
        classes.PrintStdOut.class_name = 'PrintStdOutNS2D'

        classes.set_child(
            'Spectra',
            attribs={'module_name': base_name_mod + '.spectra',
                     'class_name': 'SpectraNS2D'})

        classes.set_child(
            'spatial_means',
            attribs={'module_name': base_name_mod + '.spatial_means',
                     'class_name': 'SpatialMeansNS2D'})

        attribs = {
            'module_name': base_name_mod + '.spect_energy_budget',
            'class_name': 'SpectralEnergyBudgetNS2D'}
        classes.set_child('spect_energy_budg', attribs=attribs)

        attribs = {
            'module_name': 'fluidsim.base.output.increments',
            'class_name': 'Increments'}
        classes.set_child('increments', attribs=attribs)

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        OutputBasePseudoSpectral._complete_params_with_default(
            params, info_solver)

        params.output.phys_fields.field_to_plot = 'rot'

    def compute_energy_fft(self):
        rot_fft = self.sim.state.state_fft['rot_fft']
        ux_fft, uy_fft = self.vecfft_from_rotfft(rot_fft)
        return (np.abs(ux_fft)**2+np.abs(uy_fft)**2)/2

    def compute_enstrophy_fft(self):
        rot_fft = self.sim.state.state_fft['rot_fft']
        return np.abs(rot_fft)**2/2

    def compute_energy(self):
        energy_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(energy_fft)

    def compute_enstrophy(self):
        enstrophy_fft = self.compute_enstrophy_fft()
        return self.sum_wavenumbers(enstrophy_fft)
