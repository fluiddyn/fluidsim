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
        """Complete the ContainerXML info_solver.

        This is a static method!
        """
        info_solver.classes.Output.set_child('classes')
        classes = info_solver.classes.Output.classes

        package = 'fluidsim.solvers.plate2d.output'

        classes.set_child(
            'PrintStdOut',
            attribs={'module_name': package + '.print_stdout',
                     'class_name': 'PrintStdOutPlate2D'})

        classes.set_child(
            'PhysFields',
            attribs={'module_name': 'fluidsim.base.output.phys_fields',
                     'class_name': 'PhysFieldsBase'})

        classes.set_child(
            'Spectra',
            attribs={'module_name': package + '.spectra',
                     'class_name': 'SpectraPlate2D'})

        classes.set_child(
            'spatial_means',
            attribs={'module_name': package + '.spatial_means',
                     'class_name': 'SpatialMeansPlate2D'})

        # classes.set_child(
        #     'spatial_means',
        #     attribs={'module_name': package + '.correlations_freq',
        #              'class_name': 'CorrelationsFreq'})

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        OutputBasePseudoSpectral._complete_params_with_default(
            params, info_solver)

        params.output.phys_fields.field_to_plot = 'z'

    def compute_energies_fft(self):
        w_fft = self.sim.state.state_fft['w_fft']
        z_fft = self.sim.state.state_fft['z_fft']
        chi_fft = self.sim.state.compute('chi_fft')
        Ee_fft = np.abs(
            0.25*self.sim.oper.laplacian2_fft(np.abs(chi_fft)**2+0j))
        El_fft = np.abs(0.5*self.sim.oper.laplacian2_fft(np.abs(z_fft)**2+0j))
        Ek_fft = 0.5*np.abs(w_fft)**2
        return Ek_fft, El_fft, Ee_fft

    def compute_energy_fft(self):
        Ek_fft, El_fft, Ee_fft = self.compute_energies_fft()
        return Ek_fft + El_fft + Ee_fft

    def compute_energy(self):
        E_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(E_fft)
