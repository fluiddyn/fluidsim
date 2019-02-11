from builtins import range
import h5py

import numpy as np

from fluidsim.base.output.spectra3d import Spectra


class SpectraNS3D(Spectra):
    """Save and plot spectra."""

    def compute(self):
        """compute the values at one time."""
        nrj_vx_fft, nrj_vy_fft, nrj_vz_fft = self.output.compute_energies_fft()

        s_vx_kx, s_vx_ky, s_vx_kz = self.oper.compute_1dspectra(nrj_vx_fft)
        s_vy_kx, s_vy_ky, s_vy_kz = self.oper.compute_1dspectra(nrj_vy_fft)
        s_vz_kx, s_vz_ky, s_vz_kz = self.oper.compute_1dspectra(nrj_vz_fft)

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
        }
        dict_spectra1d = {"spectra_" + k: v for k, v in dict_spectra1d.items()}

        s_vx = self.oper.compute_3dspectrum(nrj_vx_fft)
        s_vy = self.oper.compute_3dspectrum(nrj_vy_fft)
        s_vz = self.oper.compute_3dspectrum(nrj_vz_fft)

        dict_spectra3d = {
            "vx": s_vx,
            "vy": s_vy,
            "vz": s_vz,
            "E": s_vx + s_vy + s_vy,
        }
        dict_spectra3d = {"spectra_" + k: v for k, v in dict_spectra3d.items()}

        return dict_spectra1d, dict_spectra3d

    def plot1d(
        self,
        tmin=0,
        tmax=None,
        delta_t=None,
        coef_compensate=0,
        key="E",
        key_k="kx",
        coef_plot_k3=None,
        coef_plot_k53=None,
        xlim=None,
        ylim=None,
    ):

        with h5py.File(self.path_file1d, "r") as h5file:
            times = h5file["times"][...]

            if tmax is None:
                tmax = times.max()

            ks = h5file[key_k][...]
            ks_no0 = ks.copy()
            ks_no0[ks == 0] = np.nan

            dset_spectra_E = h5file["spectra_" + key + "_" + key_k]

            if delta_t is not None:
                delta_t_save = np.mean(times[1:] - times[0:-1])
                delta_i_plot = int(np.round(delta_t / delta_t_save))
                delta_t = delta_t_save * delta_i_plot
                if delta_i_plot == 0:
                    delta_i_plot = 1
            else:
                delta_i_plot = None

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            print(
                f"plot1d(tmin={tmin}, tmax={tmax}, delta_t={delta_t},"
                + f" coef_compensate={coef_compensate:.3f})"
            )

            print(
                """plot 1D spectra
    tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {}
    imin = {:8d} ; imax = {:8d} ; delta_i = {}""".format(
                    tmin_plot,
                    tmax_plot,
                    delta_t,
                    imin_plot,
                    imax_plot,
                    delta_i_plot,
                )
            )

            fig, ax = self.output.figure_axe()
            ax.set_xlabel(f"${key_k}$")
            ax.set_ylabel("spectra " + key)
            ax.set_title(
                "1D spectra, solver "
                + self.output.name_solver
                + f", nx = {self.nx:5d}"
            )
            ax.set_xscale("log")
            ax.set_yscale("log")

            coef_norm = ks_no0 ** (coef_compensate)
            if delta_t is not None:
                for it in range(imin_plot, imax_plot + 1, delta_i_plot):
                    spectrum = dset_spectra_E[it]
                    spectrum[spectrum < 10e-16] = 0.0
                    ax.plot(ks, spectrum * coef_norm)

            spectra = dset_spectra_E[imin_plot : imax_plot + 1]
        spectrum = spectra.mean(0)
        ax.plot(ks, spectrum * coef_norm, "k", linewidth=2)

        if coef_plot_k3 is not None:
            to_plot = coef_plot_k3 * ks_no0 ** (-3) * coef_norm
            ax.plot(ks[1:], to_plot[1:], "k--")

        if coef_plot_k53 is not None:
            to_plot = coef_plot_k53 * ks_no0 ** (-5.0 / 3) * coef_norm
            ax.plot(ks[1:], to_plot[1:], "k-.")

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

    def plot3d(
        self,
        tmin=0,
        tmax=1000,
        delta_t=2,
        coef_compensate=0,
        key="E",
        coef_plot_k3=None,
        coef_plot_k53=None,
    ):
        with h5py.File(self.path_file3d, "r") as h5file:
            times = h5file["times"][...]

            ks = h5file["k_spectra3d"][...]
            ks_no0 = ks.copy()
            ks_no0[ks == 0] = 1e-15

            dset_spectrum = h5file["spectra_" + key]

            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))
            if delta_i_plot == 0 and delta_t != 0.0:
                delta_i_plot = 1
            delta_t = delta_i_plot * delta_t_save

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            print(
                "plot3d(tmin={}, tmax={}, delta_t={:.2f},".format(
                    tmin, tmax, delta_t
                )
                + f" coef_compensate={coef_compensate:.3f})"
            )

            print(
                """plot 3d spectra
    tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
    imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
                    tmin_plot,
                    tmax_plot,
                    delta_t,
                    imin_plot,
                    imax_plot,
                    delta_i_plot,
                )
            )

            fig, ax1 = self.output.figure_axe()
            ax1.set_xlabel("$k$")
            ax1.set_ylabel("3D spectra")
            ax1.set_title(
                "3D spectra, solver "
                + self.output.name_solver
                + f", nx = {self.nx:5d}"
            )
            ax1.set_xscale("log")
            ax1.set_yscale("log")

            coef_norm = ks_no0 ** coef_compensate

            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot + 1, delta_i_plot):
                    EK = dset_spectrum[it]
                    EK[EK < 10e-16] = 0.0
                    ax1.plot(ks, EK * coef_norm, "k", linewidth=1)

            EK = dset_spectrum[imin_plot : imax_plot + 1].mean(0)
        EK[EK < 10e-16] = 0.0
        ax1.plot(ks, EK * coef_norm, "k", linewidth=2)

        if coef_plot_k3 is not None:
            ax1.plot(ks, coef_plot_k3 * ks_no0 ** (-3) * coef_norm, "k--")

        if coef_plot_k53 is not None:
            ax1.plot(ks, coef_plot_k53 * ks_no0 ** (-5.0 / 3) * coef_norm, "k-.")
