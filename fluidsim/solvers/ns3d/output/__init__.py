import numpy as np

from fluidsim.base.output import OutputBasePseudoSpectral


class Output(OutputBasePseudoSpectral):
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""

        OutputBasePseudoSpectral._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        base_name_mod = "fluidsim.solvers.ns3d.output"

        classes.PrintStdOut.module_name = base_name_mod + ".print_stdout"
        classes.PrintStdOut.class_name = "PrintStdOutNS3D"

        classes.PhysFields.module_name = "fluidsim.base.output.phys_fields3d"
        classes.PhysFields.class_name = "PhysFieldsBase3D"

        classes._set_child(
            "Spectra",
            attribs={
                "module_name": base_name_mod + ".spectra",
                "class_name": "SpectraNS3D",
            },
        )

        classes._set_child(
            "Spatial_means",
            attribs={
                "module_name": base_name_mod + ".spatial_means",
                "class_name": "SpatialMeansNS3D",
            },
        )

    # attribs = {
    #     'module_name': base_name_mod + '.spect_energy_budget',
    #     'class_name': 'SpectralEnergyBudgetNS3D'}
    # classes._set_child('spect_energy_budg', attribs=attribs)

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        OutputBasePseudoSpectral._complete_params_with_default(
            params, info_solver
        )
        params.output.phys_fields.field_to_plot = "rotz"

    def compute_energies_fft(self):
        vx_fft = self.sim.state.state_spect.get_var("vx_fft")
        vy_fft = self.sim.state.state_spect.get_var("vy_fft")
        vz_fft = self.sim.state.state_spect.get_var("vz_fft")
        return (
            0.5 * np.abs(vx_fft) ** 2,
            0.5 * np.abs(vy_fft) ** 2,
            0.5 * np.abs(vz_fft) ** 2,
        )

    def compute_energy_fft(self):
        nrj_x_fft, nrj_y_fft, nrj_z_fft = self.compute_energies_fft()
        return nrj_x_fft + nrj_y_fft + nrj_z_fft

    def compute_energy(self):
        energy_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(energy_fft)
