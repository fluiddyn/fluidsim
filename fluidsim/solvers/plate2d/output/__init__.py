"""
Plate2d output (:mod:`fluidsim.solvers.plate2d.output`)
=============================================================


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
        classes = info_solver.classes.Output._set_child("classes")

        package = "fluidsim.solvers.plate2d.output"

        classes._set_child(
            "PrintStdOut",
            attribs={
                "module_name": package + ".print_stdout",
                "class_name": "PrintStdOutPlate2D",
            },
        )

        classes._set_child(
            "PhysFields",
            attribs={
                "module_name": "fluidsim.base.output.phys_fields2d",
                "class_name": "PhysFieldsBase2D",
            },
        )

        classes._set_child(
            "Spectra",
            attribs={
                "module_name": package + ".spectra",
                "class_name": "SpectraPlate2D",
            },
        )

        classes._set_child(
            "spatial_means",
            attribs={
                "module_name": package + ".spatial_means",
                "class_name": "SpatialMeansPlate2D",
            },
        )

        classes._set_child(
            "correl_freq",
            attribs={
                "module_name": package + ".correlations_freq",
                "class_name": "CorrelationsFreq",
            },
        )

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container."""
        OutputBasePseudoSpectral._complete_params_with_default(
            params, info_solver
        )

        params.output.phys_fields.field_to_plot = "z"

    def compute_energies_conversion_fft(self):
        w_fft = self.sim.state.get_var("w_fft")
        z_fft = self.sim.state.get_var("z_fft")
        chi_fft = self.sim.state.get_var("chi_fft")
        K4 = self.sim.oper.K4

        Ek_fft = 0.5 * np.abs(w_fft) ** 2
        El_fft = np.abs(0.5 * K4 * np.abs(z_fft) ** 2)
        Ee_fft = np.abs(
            0.25 * self.sim.oper.laplacian_fft(np.abs(chi_fft) ** 2 + 0j, order=4)
        )

        conversion_k_to_l_fft = np.real(K4 * w_fft * z_fft.conj())

        mamp_wz = self.sim.oper.monge_ampere_from_fft(w_fft, z_fft)
        conversion_l_to_e_fft = -np.real(
            self.sim.oper.fft2(mamp_wz) * chi_fft.conj()
        )

        return (
            Ek_fft,
            El_fft,
            Ee_fft,
            conversion_k_to_l_fft,
            conversion_l_to_e_fft,
        )

    def compute_energies_fft(self):
        w_fft = self.sim.state.state_spect.get_var("w_fft")
        z_fft = self.sim.state.state_spect.get_var("z_fft")
        chi_fft = self.sim.state.get_var("chi_fft")
        Ek_fft = 0.5 * np.abs(w_fft) ** 2
        El_fft = np.abs(
            0.5 * self.sim.oper.laplacian_fft(np.abs(z_fft) ** 2 + 0j, order=4)
        )
        Ee_fft = np.abs(
            0.25 * self.sim.oper.laplacian_fft(np.abs(chi_fft) ** 2 + 0j, order=4)
        )
        return Ek_fft, El_fft, Ee_fft

    def compute_energy_fft(self):
        Ek_fft, El_fft, Ee_fft = self.compute_energies_fft()
        return Ek_fft + El_fft + Ee_fft

    def compute_energy(self):
        E_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(E_fft)
