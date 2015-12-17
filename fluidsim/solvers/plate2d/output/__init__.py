"""
Plate2d output (:mod:`fluidsim.solvers.plate2d.output`)
=============================================================

.. currentmodule:: fluidsim.solvers.plate2d.output

Provides:

.. autosummary::
   :toctree:

   print_stdout
   spatial_means
   spectra
   correlations_freq

"""

import numpy as np


from fluidsim.base.output import OutputBasePseudoSpectral


class Output(OutputBasePseudoSpectral):

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.Output._set_child('classes')
        classes = info_solver.classes.Output.classes

        package = 'fluidsim.solvers.plate2d.output'

        classes._set_child(
            'PrintStdOut',
            attribs={'module_name': package + '.print_stdout',
                     'class_name': 'PrintStdOutPlate2D'})

        classes._set_child(
            'PhysFields',
            attribs={'module_name': 'fluidsim.base.output.phys_fields',
                     'class_name': 'PhysFieldsBase2D'})

        classes._set_child(
            'Spectra',
            attribs={'module_name': package + '.spectra',
                     'class_name': 'SpectraPlate2D'})

        classes._set_child(
            'spatial_means',
            attribs={'module_name': package + '.spatial_means',
                     'class_name': 'SpatialMeansPlate2D'})

        classes._set_child(
            'correl_freq',
            attribs={'module_name': package + '.correlations_freq',
                     'class_name': 'CorrelationsFreq'})

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        OutputBasePseudoSpectral._complete_params_with_default(
            params, info_solver)

        params.output.phys_fields.field_to_plot = 'z'

    def create_list_for_name_run(self):
        list_for_name_run = super(Output, self).create_list_for_name_run()

        if self.sim.params.FORCING:
            str_P = ('P={:5.0e}'.format(self.sim.params.forcing.forcing_rate))
            str_P = str_P.replace('+', '')
            list_for_name_run.insert(2, str_P)

        return list_for_name_run


    def compute_energies_conversion_fft(self):
        w_fft = self.sim.state.compute('w_fft')
        z_fft = self.sim.state.compute('z_fft')
        chi_fft = self.sim.state.compute('chi_fft')
        K4 = self.sim.oper.K4

        Ek_fft = 0.5*np.abs(w_fft)**2
        El_fft = np.abs(0.5*K4*np.abs(z_fft)**2)
        Ee_fft = np.abs(
            0.25*self.sim.oper.laplacian2_fft(np.abs(chi_fft)**2+0j))

        conversion_k_to_l_fft = np.real(K4*w_fft*z_fft.conj())

        mamp_wz = self.sim.oper.monge_ampere_from_fft(w_fft, z_fft)
        conversion_l_to_e_fft = -np.real(
            self.sim.oper.fft2(mamp_wz)*chi_fft.conj())

        return (Ek_fft, El_fft, Ee_fft,
                conversion_k_to_l_fft, conversion_l_to_e_fft)

    def compute_energies_fft(self):
        w_fft = self.sim.state.state_fft.get_var('w_fft')
        z_fft = self.sim.state.state_fft.get_var('z_fft')
        chi_fft = self.sim.state.compute('chi_fft')
        Ek_fft = 0.5*np.abs(w_fft)**2
        El_fft = np.abs(0.5*self.sim.oper.laplacian2_fft(np.abs(z_fft)**2+0j))
        Ee_fft = np.abs(
            0.25*self.sim.oper.laplacian2_fft(np.abs(chi_fft)**2+0j))
        return Ek_fft, El_fft, Ee_fft

    def compute_energy_fft(self):
        Ek_fft, El_fft, Ee_fft = self.compute_energies_fft()
        return Ek_fft + El_fft + Ee_fft

    def compute_energy(self):
        E_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(E_fft)
