import numpy as np

from fluidsim.solvers.ns3d.output.spectra import SpectraNS3D


class SpectraNS3DStrat(SpectraNS3D):
    """Save and plot spectra."""

    def compute(self):
        """compute the values at one time."""

        get_var = self.sim.state.state_spect.get_var
        b_fft = get_var("b_fft")
        vx_fft = get_var("vx_fft")
        vy_fft = get_var("vy_fft")
        vz_fft = get_var("vz_fft")

        urx_fft, ury_fft, udx_fft, udy_fft = self.sim.oper.urudfft_from_vxvyfft(
            vx_fft, vy_fft
        )

        nrj_vx_fft = 0.5 * np.abs(vx_fft) ** 2
        nrj_vy_fft = 0.5 * np.abs(vy_fft) ** 2
        nrj_vz_fft = 0.5 * np.abs(vz_fft) ** 2

        nrj_A_fft = 0.5 / self.sim.params.N ** 2 * np.abs(b_fft) ** 2
        nrj_Khr_fft = 0.5 * (np.abs(urx_fft) ** 2 + np.abs(ury_fft) ** 2)
        nrj_Khd_fft = 0.5 * (np.abs(udx_fft) ** 2 + np.abs(udy_fft) ** 2)

        s_vx_kx, s_vx_ky, s_vx_kz = self.oper.compute_1dspectra(nrj_vx_fft)
        s_vy_kx, s_vy_ky, s_vy_kz = self.oper.compute_1dspectra(nrj_vy_fft)
        s_vz_kx, s_vz_ky, s_vz_kz = self.oper.compute_1dspectra(nrj_vz_fft)

        s_A_kx, s_A_ky, s_A_kz = self.oper.compute_1dspectra(nrj_A_fft)
        s_Khr_kx, s_Khr_ky, s_Khr_kz = self.oper.compute_1dspectra(nrj_Khr_fft)
        s_Khd_kx, s_Khd_ky, s_Khd_kz = self.oper.compute_1dspectra(nrj_Khd_fft)

        s_kx = s_vx_kx + s_vy_kx + s_vz_kx
        s_ky = s_vx_ky + s_vy_ky + s_vz_ky
        s_kz = s_vx_kz + s_vy_kz + s_vz_kz

        dict_spectra1d = {
            "vx_kx": s_vx_kx,
            "vx_ky": s_vx_ky,
            "vx_kz": s_vx_kz,
            "vy_kx": s_vy_kx,
            "vy_ky": s_vy_ky,
            "vy_kz": s_vy_kz,
            "vz_kx": s_vz_kx,
            "vz_ky": s_vz_ky,
            "vz_kz": s_vz_kz,
            "E_kx": s_kx,
            "E_ky": s_ky,
            "E_kz": s_kz,
            "A_kx": s_A_kx,
            "A_ky": s_A_ky,
            "A_kz": s_A_kz,
            "Khr_kx": s_Khr_kx,
            "Khr_ky": s_Khr_ky,
            "Khr_kz": s_Khr_kz,
            "Khd_kx": s_Khd_kx,
            "Khd_ky": s_Khd_ky,
            "Khd_kz": s_Khd_kz,
        }
        dict_spectra1d = {"spectra_" + k: v for k, v in dict_spectra1d.items()}

        s_vx = self.oper.compute_3dspectrum(nrj_vx_fft)
        s_vy = self.oper.compute_3dspectrum(nrj_vy_fft)
        s_vz = self.oper.compute_3dspectrum(nrj_vz_fft)
        s_A = self.oper.compute_3dspectrum(nrj_A_fft)
        s_Khr = self.oper.compute_3dspectrum(nrj_Khr_fft)
        s_Khd = self.oper.compute_3dspectrum(nrj_Khd_fft)
        dict_spectra3d = {
            "vx": s_vx,
            "vy": s_vy,
            "vz": s_vz,
            "E": s_vx + s_vy + s_vy,
            "A": s_A,
            "Khr": s_Khr,
            "Khd": s_Khd,
        }
        dict_spectra3d = {"spectra_" + k: v for k, v in dict_spectra3d.items()}

        return dict_spectra1d, dict_spectra3d
