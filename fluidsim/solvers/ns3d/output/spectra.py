from functools import partial

import numpy as np
import matplotlib as mpl
import h5py

from fluidsim.util import ensure_radians

from fluidsim.base.output.spectra3d import Spectra


def _get_averaged_spectrum(key, h5file, imin_plot, imax_plot):
    dset_spectra = h5file[key]
    spectra = dset_spectra[imin_plot : imax_plot + 1]
    spectrum = spectra.mean(0)
    spectrum[spectrum < 10e-16] = 0.0
    return spectrum


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
            "E": s_vx + s_vy + s_vz,
        }
        dict_spectra3d = {"spectra_" + k: v for k, v in dict_spectra3d.items()}

        get_var = self.sim.state.state_spect.get_var
        vx_fft = get_var("vx_fft")
        vy_fft = get_var("vy_fft")

        urx_fft, ury_fft, udx_fft, udy_fft = self.sim.oper.urudfft_from_vxvyfft(
            vx_fft, vy_fft
        )
        nrj_Khr_fft = 0.5 * (np.abs(urx_fft) ** 2 + np.abs(ury_fft) ** 2)
        nrj_Khd_fft = 0.5 * (np.abs(udx_fft) ** 2 + np.abs(udy_fft) ** 2)

        s_Khr_kx, s_Khr_ky, s_Khr_kz = self.oper.compute_1dspectra(nrj_Khr_fft)
        s_Khd_kx, s_Khd_ky, s_Khd_kz = self.oper.compute_1dspectra(nrj_Khd_fft)

        dict_spectra1d.update(
            {
                "Khr_kx": s_Khr_kx,
                "Khr_ky": s_Khr_ky,
                "Khr_kz": s_Khr_kz,
                "Khd_kx": s_Khd_kx,
                "Khd_ky": s_Khd_ky,
                "Khd_kz": s_Khd_kz,
            }
        )

        s_Khr = self.oper.compute_3dspectrum(nrj_Khr_fft)
        s_Khd = self.oper.compute_3dspectrum(nrj_Khd_fft)

        dict_spectra3d.update(
            {
                "Khr": s_Khr,
                "Khd": s_Khd,
            }
        )
        if self.has_to_save_kzkh():
            dict_kzkh = {
                "K": self.oper.compute_spectrum_kzkh(
                    nrj_vx_fft + nrj_vy_fft + nrj_vz_fft
                ),
                "Khr": self.oper.compute_spectrum_kzkh(nrj_Khr_fft),
                "Khd": self.oper.compute_spectrum_kzkh(nrj_Khd_fft),
            }
        else:
            dict_kzkh = None

        return dict_spectra1d, dict_spectra3d, dict_kzkh

    def plot1d_times(
        self,
        tmin=0,
        tmax=None,
        delta_t=None,
        coef_compensate=0,
        key="E",
        key_k="kx",
        coef_plot_k3=None,
        coef_plot_k53=None,
        coef_plot_k2=None,
        xlim=None,
        ylim=None,
        only_time_average=False,
        cmap=None,
    ):

        self._plot_times(
            tmin=tmin,
            tmax=tmax,
            delta_t=delta_t,
            coef_compensate=coef_compensate,
            key=key,
            key_k=key_k,
            coef_plot_k3=coef_plot_k3,
            coef_plot_k53=coef_plot_k53,
            coef_plot_k2=coef_plot_k2,
            xlim=xlim,
            ylim=ylim,
            ndim=1,
            only_time_average=only_time_average,
            cmap=cmap,
        )

    def plot3d_times(
        self,
        tmin=0,
        tmax=None,
        delta_t=None,
        coef_compensate=0,
        key="E",
        coef_plot_k3=None,
        coef_plot_k53=None,
        coef_plot_k2=None,
        xlim=None,
        ylim=None,
        only_time_average=False,
        cmap=None,
    ):

        self._plot_times(
            tmin=tmin,
            tmax=tmax,
            delta_t=delta_t,
            coef_compensate=coef_compensate,
            key=key,
            coef_plot_k3=coef_plot_k3,
            coef_plot_k53=coef_plot_k53,
            coef_plot_k2=coef_plot_k2,
            xlim=xlim,
            ylim=ylim,
            ndim=3,
            only_time_average=only_time_average,
            cmap=cmap,
        )

    def _plot_times(
        self,
        tmin=0,
        tmax=None,
        delta_t=None,
        coef_compensate=0,
        key="E",
        key_k="kx",
        coef_plot_k3=None,
        coef_plot_k53=None,
        coef_plot_k2=None,
        xlim=None,
        ylim=None,
        only_time_average=False,
        ndim=1,
        cmap=None,
    ):

        if ndim not in [1, 3]:
            raise ValueError

        path_file = getattr(self, f"path_file{ndim}d")

        if ndim == 1:
            key_spectra = "spectra_" + key + "_" + key_k
            key_k_label = "k_" + key_k[-1]
        else:
            key_spectra = "spectra_" + key
            key_k = "k_spectra3d"
            key_k_label = "k"

        with h5py.File(path_file, "r") as h5file:
            times = h5file["times"][...]
            ks = h5file[key_k][...]

        ks_no0 = ks.copy()
        ks_no0[ks == 0] = np.nan

        if tmax is None:
            tmax = times.max()

        imin_plot = np.argmin(abs(times - tmin))
        imax_plot = np.argmin(abs(times - tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]

        delta_t_save = np.diff(times).mean()

        if delta_t is None:
            nb_curves = min(20, len(times))
            delta_t = (tmax_plot - tmin_plot) / nb_curves

        delta_i_plot = int(np.round(delta_t / delta_t_save))
        if delta_i_plot == 0:
            delta_i_plot = 1
        delta_t = delta_t_save * delta_i_plot

        print(
            f"plot{ndim}d_times(tmin={tmin:8.6g}, tmax={tmax:8.6g}, delta_t={delta_t:8.6g},"
            f" coef_compensate={coef_compensate:.3f})\n"
            f"""plot {ndim}D spectra
tmin = {tmin_plot:8.6g} ; tmax = {tmax_plot:8.6g} ; delta_t = {delta_t:8.6g}
imin = {imin_plot:8d} ; imax = {imax_plot:8d} ; delta_i = {delta_i_plot}"""
        )

        fig, ax = self.output.figure_axe()
        ax.set_xlabel(f"${key_k_label}$")
        ax.set_ylabel("spectra " + key)
        ax.set_title(
            f"{ndim}D spectra (tmin={tmin:.2g}, tmax={tmax:.2g})\n"
            + self.output.summary_simul
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        if cmap is None:
            cmap = mpl.rcParams["image.cmap"]
        cmapper = getattr(mpl.pyplot.cm, cmap)
        nb_plots = imax_plot - imin_plot + 1
        colors = cmapper(np.linspace(0, 1, nb_plots))

        with h5py.File(path_file, "r") as h5file:
            dset_spectra = h5file[key_spectra]
            coef_norm = ks_no0 ** (coef_compensate)
            if not only_time_average:
                for ic, it in enumerate(
                    range(imin_plot, imax_plot + 1, delta_i_plot)
                ):
                    spectrum = dset_spectra[it]
                    spectrum[spectrum < 10e-16] = 0.0
                    ax.plot(ks, spectrum * coef_norm, color=colors[ic])

            spectra = dset_spectra[imin_plot : imax_plot + 1]
        spectrum = spectra.mean(0)
        spectrum[spectrum < 10e-16] = 0.0
        ax.plot(ks, spectrum * coef_norm, "k", linewidth=2)

        if coef_plot_k3 is not None:
            to_plot = coef_plot_k3 * ks_no0 ** (-3) * coef_norm
            ax.plot(ks[1:], to_plot[1:], "k--")

        if coef_plot_k53 is not None:
            to_plot = coef_plot_k53 * ks_no0 ** (-5.0 / 3) * coef_norm
            ax.plot(ks[1:], to_plot[1:], "k-.")

        if coef_plot_k2 is not None:
            to_plot = coef_plot_k2 * ks_no0 ** (-2) * coef_norm
            ax.plot(ks[1:], to_plot[1:], "k--")

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

    def plot1d(
        self,
        tmin=0,
        tmax=None,
        coef_compensate=0,
        coef_plot_k3=None,
        coef_plot_k53=None,
        coef_plot_k2=None,
        xlim=None,
        ylim=None,
        directions=("x", "z"),
        plot_forcing_region=False,
        plot_dissipative_scales=False,
    ):

        ax = self._plot_ndim(
            tmin,
            tmax,
            coef_compensate=coef_compensate,
            coef_plot_k3=coef_plot_k3,
            coef_plot_k53=coef_plot_k53,
            coef_plot_k2=coef_plot_k2,
            xlim=xlim,
            ylim=ylim,
            ndim=1,
            directions=directions,
        )

        factor = 2
        ymin, ymax = ax.get_ybound()

        if plot_forcing_region:

            with h5py.File(self.path_file1d, "r") as h5file:
                kx = h5file["kx"][...]
                ky = h5file["ky"][...]
                kz = h5file["kz"][...]

            if self.params.forcing.type == "tcrandom_anisotropic":
                angle = ensure_radians(
                    self.params.forcing.tcrandom_anisotropic.angle
                )

                kf_min = self.params.forcing.nkmin_forcing * self.oper.deltak
                kf_max = self.params.forcing.nkmax_forcing * self.oper.deltak

                tmp = self.params.forcing.tcrandom_anisotropic
                try:
                    delta_angle = tmp.delta_angle
                except AttributeError:
                    # loading old simul with delta_angle
                    delta_angle = None
                else:
                    delta_angle = ensure_radians(delta_angle)

                if delta_angle is None:
                    khmin_forcing = np.sin(angle) * kf_min
                    kvmin_forcing = np.cos(angle) * kf_min
                    khmax_forcing = np.sin(angle) * kf_max
                    kvmax_forcing = np.cos(angle) * kf_max
                else:
                    khmin_forcing = np.sin(angle - 0.5 * delta_angle) * kf_min
                    kvmin_forcing = np.cos(angle + 0.5 * delta_angle) * kf_min
                    khmax_forcing = np.sin(angle + 0.5 * delta_angle) * kf_max
                    kvmax_forcing = np.cos(angle - 0.5 * delta_angle) * kf_max

                khmean_forcing = 0.5 * (khmin_forcing + khmax_forcing)
                kvmean_forcing = 0.5 * (kvmin_forcing + kvmax_forcing)

                fill_between = partial(
                    ax.fill_between, facecolor="gray", alpha=0.5
                )
                text = partial(ax.text, ha="center", va="center", size=10)

                if "x" in directions:
                    where = (kx > khmin_forcing) & (kx < khmax_forcing)
                    fill_between(kx, ymin, ymax, where=where)
                    text(khmean_forcing, factor * ymin, r"$k_{f,x}$")
                if "y" in directions:
                    where = (ky > khmin_forcing) & (ky < khmax_forcing)
                    fill_between(ky, ymin, ymax, where=where)
                    text(khmean_forcing, factor * ymin, r"$k_{f,y}$")
                if "z" in directions:
                    where = (kz > kvmin_forcing) & (kz < kvmax_forcing)
                    fill_between(kz, ymin, ymax, where=where)
                    text(kvmean_forcing, factor * ymin, r"$k_{f,z}$")
            else:
                raise NotImplementedError

        if plot_dissipative_scales:
            nu_2 = self.params.nu_2
            nu_4 = self.params.nu_4
            nu_8 = self.params.nu_8
            # warning: this wrongly assumes that it is an energy forcing rate
            Pf = self.params.forcing.forcing_rate

            text = partial(ax.text, ha="left", va="center", size=10)

            if nu_2 is not None and nu_2 != 0.0:
                eta_2 = (nu_2 / (Pf ** (1.0 / 3.0))) ** (3.0 / 4.0)
                kd_2 = 1.0 / eta_2
                ax.axvline(x=kd_2, color="k")
                text(1.1 * kd_2, factor * ymin, r"$k_{d2}$")
            if nu_4 is not None and nu_4 != 0.0:
                eta_4 = (nu_4 / (Pf ** (1.0 / 3.0))) ** (3.0 / 10.0)
                kd_4 = 1.0 / eta_4
                ax.axvline(x=kd_4, color="k")
                text(1.1 * kd_4, factor * ymin, r"$k_{d4}$")
            if nu_8 is not None and nu_8 != 0.0:
                eta_8 = (nu_8 / (Pf ** (1.0 / 3.0))) ** (3.0 / 22.0)
                kd_8 = 1.0 / eta_8
                ax.axvline(x=kd_8, color="k")
                text(1.1 * kd_8, factor * ymin, r"$k_{d8}$")

    def plot3d(
        self,
        tmin=0,
        tmax=None,
        coef_compensate=0,
        coef_plot_k3=None,
        coef_plot_k53=None,
        coef_plot_k2=None,
        xlim=None,
        ylim=None,
    ):

        ax = self._plot_ndim(
            tmin,
            tmax,
            coef_compensate=coef_compensate,
            coef_plot_k3=coef_plot_k3,
            coef_plot_k53=coef_plot_k53,
            coef_plot_k2=coef_plot_k2,
            xlim=xlim,
            ylim=ylim,
            ndim=3,
        )

        return ax

    def _plot_ndim(
        self,
        tmin=0,
        tmax=None,
        coef_compensate=0,
        coef_plot_k3=None,
        coef_plot_k53=None,
        coef_plot_k2=None,
        xlim=None,
        ylim=None,
        ndim=1,
        directions=("x", "z"),
    ):
        if ndim not in [1, 3]:
            raise ValueError

        path_file = getattr(self, f"path_file{ndim}d")

        if ndim == 1:
            key_k = "k" + directions[0]
            key_k_label = r",\ ".join(["k_" + letter for letter in directions])
        else:
            key_k = "k_spectra3d"
            key_k_label = "k"

        with h5py.File(path_file, "r") as h5file:
            times = h5file["times"][...]
            ks = h5file[key_k][...]

        if tmax is None:
            tmax = times.max()

        imin_plot = np.argmin(abs(times - tmin))
        imax_plot = np.argmin(abs(times - tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]

        print(
            f"plot{ndim}d(tmin={tmin:8.6g}, tmax={tmax:8.6g},"
            f" coef_compensate={coef_compensate:.3f})\n"
            f"""plot {ndim}D spectra
tmin = {tmin_plot:8.6g} ; tmax = {tmax_plot:8.6g}
imin = {imin_plot:8d} ; imax = {imax_plot:8d}"""
        )

        fig, ax = self.output.figure_axe()
        ax.set_xlabel(f"${key_k_label}$")
        ax.set_ylabel("spectra")
        ax.set_title(
            f"{ndim}D spectra (tmin={tmin:.2g}, tmax={tmax:.2g})\n"
            + self.output.summary_simul
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        if ndim == 1:
            for direction in directions:
                self._plot1d_direction(
                    direction, imin_plot, imax_plot, coef_compensate, ax
                )
        else:
            ks_no0 = ks.copy()
            ks_no0[ks == 0] = np.nan
            with h5py.File(path_file, "r") as h5file:
                spectrum = h5file["spectra_E"][imin_plot : imax_plot + 1].mean(0)
                coef_norm = ks_no0 ** (coef_compensate)

            spectrum[spectrum < 10e-16] = 0.0
            ax.plot(ks, spectrum * coef_norm, "k", linewidth=2)

        ks = np.linspace(ks[1], 0.8 * ks[-1], 20)

        ks_no0 = ks.copy()
        ks_no0[ks == 0] = np.nan
        coef_norm = ks_no0 ** (coef_compensate)

        if coef_plot_k3 is not None:
            to_plot = coef_plot_k3 * ks_no0 ** (-3) * coef_norm
            ax.plot(ks[1:], to_plot[1:], "k--", label=r"$\propto k^{-3}$")

        if coef_plot_k53 is not None:
            to_plot = coef_plot_k53 * ks_no0 ** (-5.0 / 3) * coef_norm
            ax.plot(ks[1:], to_plot[1:], "k-.", label=r"$\propto k^{-5/3}$")

        if coef_plot_k2 is not None:
            to_plot = coef_plot_k2 * ks_no0 ** (-2) * coef_norm
            ax.plot(ks[1:], to_plot[1:], "k:", label=r"$\propto k^{-2}$")

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if ndim == 1:
            ax.legend(loc="lower left")

        return ax

    def _plot1d_direction(
        self, direction, imin_plot, imax_plot, coef_compensate, ax
    ):

        with h5py.File(self.path_file1d, "r") as h5file:
            ks = h5file["k" + direction][...]
            spectrum = _get_averaged_spectrum(
                "spectra_E_k" + direction, h5file, imin_plot, imax_plot
            )

        ks_no0 = ks.copy()
        ks_no0[ks == 0] = np.nan
        coef_norm = ks_no0 ** (coef_compensate)

        style_line = "r"
        if direction == "z":
            style_line = "g"

        ax.plot(
            ks,
            spectrum * coef_norm,
            style_line,
            linewidth=2,
            label=f"$E(k_{direction})$",
        )

    def plot3d_cumul_diss(self, tmin=0, tmax=None):
        path_file = self.path_file3d
        with h5py.File(path_file, "r") as h5file:
            times = h5file["times"][...]
        if tmax is None:
            tmax = times.max()

        # load 3D spectra
        data = self.load3d_mean(tmin=tmin, tmax=tmax)
        spectra_K = (
            data["spectra_vx"] + data["spectra_vy"] + data["spectra_vz"]
        )  # 'spectra_E' key was wrong for fluidsim<=0.3.3: use components instead
        k = data["k"]
        deltak = k[1]

        # nu_2 dissipation flux
        fd = self.sim.params.nu_2 * k**2
        diss = fd * spectra_K
        flux_diss_nu2 = deltak * np.cumsum(diss)

        # nu_4 dissipation flux
        fd = self.sim.params.nu_4 * k**4
        diss = fd * spectra_K
        flux_diss_nu4 = deltak * np.cumsum(diss)

        # normalize by total dissipation
        eps_tot = flux_diss_nu2[-1] + flux_diss_nu4[-1]
        flux_diss_nu2 /= eps_tot
        flux_diss_nu4 /= eps_tot

        # plot
        fig, ax = self.output.figure_axe()
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$D(k)/\epsilon$")

        ax.plot(k, flux_diss_nu2, "r-", label=r"$D_2(k)$")
        ax.plot(k, flux_diss_nu4, "m-", label=r"$D_4(k)$")

        ax.set_title(
            f"3D cumulative dissipation spectra (tmin={tmin:.2g}, tmax={tmax:.2g})\n"
            + self.output.summary_simul
        )
        ax.set_xscale("log")
        fig.legend()
