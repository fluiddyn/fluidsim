"""Output for the ns3d.strat solver
===================================

.. autoclass:: Output
   :members:
   :private-members:

.. autosummary::
   :toctree:

   spatial_means
   spectra
   spect_energy_budget

"""

try:
    from fluidsim.operators.operators3d import (
        compute_energy_from_1field,
        compute_energy_from_1field_with_coef,
        compute_energy_from_2fields,
        compute_energy_from_3fields,
    )
except ImportError:
    from fluidsim import _is_testing

    if not _is_testing:
        raise

from ...output import Output as OutputNS3D


class Output(OutputNS3D):
    """Main output class for ns3d.strat"""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        OutputNS3D._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes
        base_name_mod = "fluidsim.solvers.ns3d.strat.output"

        classes.SpatialMeans.module_name = base_name_mod + ".spatial_means"
        classes.SpatialMeans.class_name = "SpatialMeansNS3DStrat"

        classes.Spectra.module_name = base_name_mod + ".spectra"
        classes.Spectra.class_name = "SpectraNS3DStrat"

        classes.SpectralEnergyBudget.module_name = (
            base_name_mod + ".spect_energy_budget"
        )
        classes.SpectralEnergyBudget.class_name = "SpectralEnergyBudgetNS3DStrat"

    def compute_energies_fft(self):
        get_var = self.sim.state.state_spect.get_var
        b_fft = get_var("b_fft")
        vx_fft = get_var("vx_fft")
        vy_fft = get_var("vy_fft")
        vz_fft = get_var("vz_fft")

        urx_fft, ury_fft, udx_fft, udy_fft = self.oper.urudfft_from_vxvyfft(
            vx_fft, vy_fft
        )

        nrj_A = compute_energy_from_1field_with_coef(
            b_fft, 1.0 / self.sim.params.N**2
        )
        nrj_Kz = compute_energy_from_1field(vz_fft)
        nrj_Khr = compute_energy_from_2fields(urx_fft, ury_fft)
        nrj_Khd = compute_energy_from_2fields(udx_fft, udy_fft)
        return nrj_A, nrj_Kz, nrj_Khr, nrj_Khd

    def compute_energy_fft(self):
        get_var = self.sim.state.state_spect.get_var
        b_fft = get_var("b_fft")
        vx_fft = get_var("vx_fft")
        vy_fft = get_var("vy_fft")
        vz_fft = get_var("vz_fft")
        nrj_A = compute_energy_from_1field_with_coef(
            b_fft, 1.0 / self.sim.params.N**2
        )
        nrj_K = compute_energy_from_3fields(vx_fft, vy_fft, vz_fft)
        return nrj_A + nrj_K

    def compute_energies(self):
        energies_fft = self.compute_energies_fft()
        return tuple(self.sum_wavenumbers(e_fft) for e_fft in energies_fft)

    def compute_energy(self):
        energy_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(energy_fft)
