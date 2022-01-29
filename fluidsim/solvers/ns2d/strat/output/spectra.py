"""Spectra output (:mod:`fluidsim.solvers.ns2d.strat.output.spectra`)
=====================================================================

.. autoclass:: SpectraNS2DStrat
   :members:
   :private-members:

"""

import h5py

import numpy as np

from fluidsim.base.output.spectra import Spectra


class SpectraNS2DStrat(Spectra):
    """Save and plot spectra."""

    def compute(self):
        """compute the values at one time."""
        # energy_fft = self.output.compute_energy_fft()
        energyK_fft, energyA_fft = self.output.compute_energies_fft()
        energy_fft = energyK_fft + energyA_fft
        energyK_ux_fft, energyK_uy_fft = self.output.compute_energies2_fft()
        energyK, energyA, energyK_ux = self.output.compute_energies()

        # Compute the kinetic energy spectra 1D for the two velocity components
        # and two directions
        spectrum1Dkx_EK_ux, spectrum1Dky_EK_ux = self.spectra1D_from_fft(
            energyK_ux_fft
        )
        spectrum1Dkx_EK_uy, spectrum1Dky_EK_uy = self.spectra1D_from_fft(
            energyK_uy_fft
        )
        spectrum1Dkx_EK, spectrum1Dky_EK = self.spectra1D_from_fft(energyK_fft)

        # Compute the potential energy spectra 1D two directions
        spectrum1Dkx_EA, spectrum1Dky_EA = self.spectra1D_from_fft(energyA_fft)

        # Compute the total energy spectra 1D
        spectrum1Dkx_E, spectrum1Dky_E = self.spectra1D_from_fft(energy_fft)
        # Dictionary with the 1D kinetic energy spectra
        dict_spectra1D = {
            "spectrum1Dkx_EK_ux": spectrum1Dkx_EK_ux,
            "spectrum1Dky_EK_ux": spectrum1Dky_EK_ux,
            "spectrum1Dkx_EK_uy": spectrum1Dkx_EK_uy,
            "spectrum1Dky_EK_uy": spectrum1Dky_EK_uy,
            "spectrum1Dkx_EK": spectrum1Dkx_EK,
            "spectrum1Dky_EK": spectrum1Dky_EK,
            "spectrum1Dkx_EA": spectrum1Dkx_EA,
            "spectrum1Dky_EA": spectrum1Dky_EA,
            "spectrum1Dkx_E": spectrum1Dkx_E,
            "spectrum1Dky_E": spectrum1Dky_E,
        }

        # compute the kinetic energy spectra 2D
        spectrum2D_E = self.spectrum2D_from_fft(energy_fft)
        spectrum2D_EK = self.spectrum2D_from_fft(energyK_fft)
        spectrum2D_EK_ux = self.spectrum2D_from_fft(energyK_ux_fft)
        spectrum2D_EK_uy = self.spectrum2D_from_fft(energyK_uy_fft)
        spectrum2D_EA = self.spectrum2D_from_fft(energyA_fft)
        dict_spectra2D = {
            "spectrum2D_EK_ux": spectrum2D_EK_ux,
            "spectrum2D_EK_uy": spectrum2D_EK_uy,
            "spectrum2D_EK": spectrum2D_EK,
            "spectrum2D_EA": spectrum2D_EA,
            "spectrum2D_E": spectrum2D_E,
        }

        return dict_spectra1D, dict_spectra2D

    def _online_plot_saving(self, dict_spectra1D, dict_spectra2D):
        if (
            self.nx == self.params.oper.ny
            and self.params.oper.Lx == self.params.oper.Ly
        ):
            spectrum2D_EK = dict_spectra2D["spectrum2D_EK"]
            spectrum2D_EA = dict_spectra2D["spectrum2D_EA"]
            spectrum2D = spectrum2D_EK + spectrum2D_EA
            khE = self.oper.khE
            coef_norm = khE ** (3.0)
            self.axe.loglog(khE, spectrum2D * coef_norm, "k")
            lin_inf, lin_sup = self.axe.get_ylim()
            if lin_inf < 10e-6:
                lin_inf = 10e-6
            self.axe.set_ylim([lin_inf, lin_sup])
        else:
            print(
                "you need to implement the ploting "
                "of the spectra for this case"
            )

    def load1d_means(
        self, tmin=0, tmax=None, delta_t=2, versus_kx=True, versus_ky=True
    ):
        means = {}
        with h5py.File(self.path_file1D, "r") as h5file:

            # Open data from file
            dset_times = h5file["times"]
            dset_kxE = h5file["kxE"]
            dset_kyE = h5file["kyE"]
            times = dset_times[...]
            means["times"] = times
            means["kx"] = dset_kxE[...]
            means["ky"] = dset_kyE[...]

            if tmin is None:
                tmin = 0
            if tmax is None:
                tmax = np.max(times)

            # Compute average from tmin and tmax for plot
            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))
            delta_t = delta_t_save * delta_i_plot
            if delta_i_plot == 0:
                delta_i_plot = 1

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            if versus_kx:
                # Open data set 1D spectra
                dset_spectrum1Dkx_EA = h5file["spectrum1Dkx_EA"]
                dset_spectrum1Dkx_EK = h5file["spectrum1Dkx_EK"]
                dset_spectrum1Dkx_E = h5file["spectrum1Dkx_E"]

                # Average in time between tmin and tmax
                means["E_kx"] = (
                    dset_spectrum1Dkx_E[imin_plot : imax_plot + 1]
                ).mean(0)
                means["EK_kx"] = (
                    dset_spectrum1Dkx_EK[imin_plot : imax_plot + 1]
                ).mean(0)
                means["EA_kx"] = (
                    dset_spectrum1Dkx_EA[imin_plot : imax_plot + 1]
                ).mean(0)

            if versus_ky:
                # Open data set 1D spectra
                dset_spectrum1Dky_EA = h5file["spectrum1Dky_EA"]
                dset_spectrum1Dky_EK = h5file["spectrum1Dky_EK"]
                dset_spectrum1Dky_E = h5file["spectrum1Dky_E"]

                # Average in time between tmin and tmax
                means["E_ky"] = (
                    dset_spectrum1Dky_E[imin_plot : imax_plot + 1]
                ).mean(0)
                means["EK_ky"] = (
                    dset_spectrum1Dky_EK[imin_plot : imax_plot + 1]
                ).mean(0)
                means["EA_ky"] = (
                    dset_spectrum1Dky_EA[imin_plot : imax_plot + 1]
                ).mean(0)

        print(
            """load spectra
        tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
        imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
                tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
            )
        )

        return means

    def plot1d(
        self,
        tmin=0,
        tmax=None,
        delta_t=2,
        coef_compensate=5 / 3,
        level2=1.0,
        level3=1.0,
        yrange=5,
    ):
        """Plot spectra one-dimensional."""
        print(
            "plot1d(tmin={}, tmax={}, delta_t={:.2f},".format(tmin, tmax, delta_t)
            + f" coef_compensate={coef_compensate:.3f})"
        )

        means = self.load1d_means(
            tmin, tmax, delta_t, versus_kx=True, versus_ky=True
        )

        kx = means["kx"]
        ky = means["ky"]

        # Parameters figure E(k_x)
        fig, ax = self.output.figure_axe()
        ax.set_xlabel("$k_x$, $k_z$")
        ax.set_ylabel(r"$E(k)k^{{{}}}$".format(round(coef_compensate, 2)))
        ax.set_title("1D spectra\n" + self.output.summary_simul)
        ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.set_ylim(ymin=1e-6, ymax=1e3)

        id_kx_dealiasing = (
            np.argmin(abs(kx - (kx.max() * self.sim.oper.coef_dealiasing))) - 1
        )
        id_ky_dealiasing = (
            np.argmin(abs(ky - (ky.max() * self.sim.oper.coef_dealiasing))) - 1
        )

        # Remove modes dealiased.
        E_kx_plot = means["E_kx"][:id_kx_dealiasing]
        EK_kx_plot = means["EK_kx"][:id_kx_dealiasing]
        EA_kx_plot = means["EA_kx"][:id_kx_dealiasing]
        kx_plot = kx[:id_kx_dealiasing]

        # Remove shear modes if there is no energy on them.
        if self.sim.params.oper.NO_SHEAR_MODES:
            E_kx_plot = E_kx_plot[1:]
            EK_kx_plot = EK_kx_plot[1:]
            EA_kx_plot = EA_kx_plot[1:]
            kx_plot = kx_plot[1:]

        # Compute k_b: L_b = U / N
        U = np.sqrt(np.mean(abs(self.sim.state.get_var("ux")) ** 2))
        k_b = self.sim.params.N / U
        ax.axvline(x=k_b, color="y", linestyle="--", label="$k_b$")

        # Plot ozmidov scale
        k_o = (self.sim.params.N**3 / self.sim.params.forcing.forcing_rate) ** (
            1 / 2
        )
        ax.axvline(x=k_o, color="y", linestyle=":", label="$k_o$")

        # ax.plot(
        # kx_plot, E_kx_plot * kx_plot ** coef_compensate, "k", label="$E(k_x)$"
        # )
        ax.plot(
            kx_plot,
            EK_kx_plot * kx_plot**coef_compensate,
            "r",
            label="$E_K(k_x)$",
        )
        ax.plot(
            kx_plot,
            EA_kx_plot * kx_plot**coef_compensate,
            "b",
            label="$E_A(k_x)$",
        )

        # Remove modes dealiased.
        E_ky_plot = means["E_ky"][:id_ky_dealiasing]
        EK_ky_plot = means["EK_ky"][:id_ky_dealiasing]
        EA_ky_plot = means["EA_ky"][:id_ky_dealiasing]
        ky_plot = ky[:id_ky_dealiasing]

        # Remove shear modes if there is no energy on them.
        if self.sim.params.oper.NO_SHEAR_MODES:
            E_ky_plot = E_ky_plot[1:]
            EK_ky_plot = EK_ky_plot[1:]
            EA_ky_plot = EA_ky_plot[1:]
            ky_plot = ky_plot[1:]

        # ax.plot(
        # ky_plot,
        # E_ky_plot * ky_plot ** coef_compensate,
        # "k--",
        # label="$E(k_z)$",
        # )
        ax.plot(
            ky_plot,
            EK_ky_plot * ky_plot**coef_compensate,
            "r--",
            label="$E_K(k_z)$",
        )
        ax.plot(
            ky_plot,
            EA_ky_plot * ky_plot**coef_compensate,
            "b--",
            label="$E_A(k_z)$",
        )

        # Plot scaling lines
        kx = kx_plot[40:id_kx_dealiasing]
        if level3:
            ax.plot(
                kx,
                level3 * kx ** (-3) * kx**coef_compensate,
                "k:",
                label=r"$k^{-3}$",
            )
        if level2:
            ax.plot(
                kx,
                level2 * kx ** (-2) * kx**coef_compensate,
                "k-.",
                label=r"$k^{-2}$",
            )

        # Plot forcing wave-number k_f
        nkmax = self.sim.params.forcing.nkmax_forcing
        nkmin = self.sim.params.forcing.nkmin_forcing
        pforcing = self.sim.params.forcing
        if pforcing.enable and pforcing.type == "tcrandom_anisotropic":
            angle = pforcing.tcrandom_anisotropic.angle
            try:
                if angle.endswith("°"):
                    angle = np.pi / 180 * float(angle[:-1])
            except AttributeError:
                pass

            # Band forcing region kx
            k_fxmin = nkmin * self.sim.oper.deltak * np.sin(angle)
            k_fxmin = max(k_fxmin, self.sim.oper.deltakx)
            k_fxmax = nkmax * self.sim.oper.deltak * np.sin(angle)

            # Band forcing region ky
            k_fymin = nkmin * self.sim.oper.deltak * np.cos(angle)
            k_fymin = max(k_fymin, self.sim.oper.deltaky)
            k_fymax = nkmax * self.sim.oper.deltak * np.cos(angle)

            # Plot forcing band
            ax.axvspan(k_fxmin, k_fxmax, alpha=0.15, color="black")
            ax.axvspan(k_fymin, k_fymax, alpha=0.15, color="black")

        # Set limits axis y
        if yrange is not None:
            ymax = 2 * max(
                (E_kx_plot * kx_plot**coef_compensate).max(),
                (E_ky_plot * ky_plot**coef_compensate).max(),
            )
            ax.set_ylim((10 ** (-yrange) * ymax, ymax))

        ax.legend()

        # from ipdb import set_trace
        # set_trace()

    def load2d_means(self, tmin=0, tmax=1000, delta_t=2):

        means = {}
        # Load data from file
        with h5py.File(self.path_file2D, "r") as h5file:
            dset_times = h5file["times"]
            means["kh"] = h5file["khE"][...]
            times = dset_times[...]
            means["times"] = times

            # Compute average from tmin and tmax for plot
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
                """plot 2D spectra
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

            dset_spectrum_E = h5file["spectrum2D_E"]
            dset_spectrum_EK = h5file["spectrum2D_EK"]
            dset_spectrum_EA = h5file["spectrum2D_EA"]

            means["E"] = dset_spectrum_E[imin_plot : imax_plot + 1].mean(0)
            means["EK"] = dset_spectrum_EK[imin_plot : imax_plot + 1].mean(0)
            means["EA"] = dset_spectrum_EA[imin_plot : imax_plot + 1].mean(0)

        return means

    def plot2d(self, tmin=0, tmax=1000, delta_t=2, coef_compensate=3):
        """Plot 2D spectra."""

        print(
            "plot2s(tmin={}, tmax={}, delta_t={:.2f},".format(tmin, tmax, delta_t)
            + f" coef_compensate={coef_compensate:.3f})"
        )

        means = self.load2d_means(tmin, tmax, delta_t)

        kh = means["kh"]
        E = means["E"]
        EK = means["EK"]
        EA = means["EA"]

        # Parameters figure
        fig, ax = self.output.figure_axe()
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$E(k)$")
        ax.set_title("2D spectra\n" + self.output.summary_simul)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.plot(kh, E, "k", label="E")
        ax.plot(kh, EK, "r", label="EK")
        ax.plot(kh, EA, "b", label="EA")

        ax.legend()
