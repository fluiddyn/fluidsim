"""Output for the ns3d solver
=============================

.. autoclass:: Output
   :members:
   :private-members:

.. autosummary::
   :toctree:

   spatial_means
   spectra
   spect_energy_budget

"""

from fluidsim.base.output import OutputBasePseudoSpectral

try:
    from fluidsim.operators.operators3d import compute_energy_from_1field
except ImportError:
    from fluidsim import _is_testing

    if not _is_testing:
        raise


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
            "SpatialMeans",
            attribs={
                "module_name": base_name_mod + ".spatial_means",
                "class_name": "SpatialMeansNS3D",
            },
        )

        classes._set_child(
            "SpectralEnergyBudget",
            attribs={
                "module_name": base_name_mod + ".spect_energy_budget",
                "class_name": "SpectralEnergyBudgetNS3D",
            },
        )

        classes._set_child(
            "TemporalSpectra",
            attribs={
                "module_name": "fluidsim.base.output.temporal_spectra",
                "class_name": "TemporalSpectra3D",
            },
        )

        classes._set_child(
            "SpatioTemporalSpectra",
            attribs={
                "module_name": base_name_mod + ".spatiotemporal_spectra",
                "class_name": "SpatioTemporalSpectraNS3D",
            },
        )

        classes._set_child(
            "CrossCorrelations",
            attribs={
                "module_name": "fluidsim.base.output.cross_corr3d",
                "class_name": "CrossCorrelations",
            },
        )

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container."""
        OutputBasePseudoSpectral._complete_params_with_default(
            params, info_solver
        )
        params.output.phys_fields.field_to_plot = "rotz"

    def compute_energies_fft(self):
        vx_fft = self.sim.state.state_spect.get_var("vx_fft")
        vy_fft = self.sim.state.state_spect.get_var("vy_fft")
        vz_fft = self.sim.state.state_spect.get_var("vz_fft")
        return (
            compute_energy_from_1field(vx_fft),
            compute_energy_from_1field(vy_fft),
            compute_energy_from_1field(vz_fft),
        )

    def compute_energy_fft(self):
        nrj_x_fft, nrj_y_fft, nrj_z_fft = self.compute_energies_fft()
        return nrj_x_fft + nrj_y_fft + nrj_z_fft

    def compute_energy(self):
        energy_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(energy_fft)

    def plot_summary(self, tmin=0, key_field=None):
        # pylint: disable=maybe-no-member
        self.spatial_means.plot()
        self.spectra.plot1d(tmin=tmin)
        self.spect_energy_budg.plot_fluxes(tmin=tmin)
        if key_field:
            self.phys_fields.plot(key_field)
