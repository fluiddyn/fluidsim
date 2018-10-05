import numpy as np

from ...output import Output as OutputNS3D


class Output(OutputNS3D):
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        OutputNS3D._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes
        base_name_mod = "fluidsim.solvers.ns3d.strat.output"

        classes.Spatial_means.module_name = base_name_mod + ".spatial_means"
        classes.Spatial_means.class_name = "SpatialMeansNS3DStrat"

        classes.Spectra.module_name = base_name_mod + ".spectra"
        classes.Spectra.class_name = "SpectraNS3DStrat"

    def compute_energies_fft(self):
        get_var = self.sim.state.state_spect.get_var
        b_fft = get_var("b_fft")
        vx_fft = get_var("vx_fft")
        vy_fft = get_var("vy_fft")
        vz_fft = get_var("vz_fft")

        urx_fft, ury_fft, udx_fft, udy_fft = self.oper.urudfft_from_vxvyfft(
            vx_fft, vy_fft
        )

        nrj_A = 0.5 / self.sim.params.N ** 2 * np.abs(b_fft) ** 2
        nrj_Kz = 0.5 * np.abs(vz_fft) ** 2
        nrj_Khr = 0.5 * (np.abs(urx_fft) ** 2 + np.abs(ury_fft) ** 2)
        nrj_Khd = 0.5 * (np.abs(udx_fft) ** 2 + np.abs(udy_fft) ** 2)
        return nrj_A, nrj_Kz, nrj_Khr, nrj_Khd

    def compute_energy_fft(self):
        get_var = self.sim.state.state_spect.get_var
        b_fft = get_var("b_fft")
        vx_fft = get_var("vx_fft")
        vy_fft = get_var("vy_fft")
        vz_fft = get_var("vz_fft")
        nrj_A = 0.5 / self.sim.params.N ** 2 * np.abs(b_fft) ** 2
        nrj_K = 0.5 * (
            np.abs(vx_fft) ** 2 + np.abs(vy_fft) ** 2 + np.abs(vz_fft) ** 2
        )
        return nrj_A + nrj_K

    def compute_energies(self):
        energies_fft = self.compute_energies_fft()
        return tuple(self.sum_wavenumbers(e_fft) for e_fft in energies_fft)

    def compute_energy(self):
        energy_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(energy_fft)
