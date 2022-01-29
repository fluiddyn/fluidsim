"""Energy budget (:mod:`fluidsim.solvers.ns2d.strat.output.spect_energy_budget`)
================================================================================

.. autoclass:: SpectralEnergyBudgetNS2DStrat
   :members:
   :private-members:

"""

import numpy as np
import h5py

from fluiddyn.util import mpi

from fluidsim.base.output.spect_energy_budget import (
    SpectralEnergyBudgetBase,
    cumsum_inv,
)


class SpectralEnergyBudgetNS2DStrat(SpectralEnergyBudgetBase):
    """Save and plot energy budget in spectral space."""

    def compute(self):
        """compute the spectral energy budget at one time."""
        oper = self.sim.oper

        ux = self.sim.state.state_phys.get_var("ux")
        uy = self.sim.state.state_phys.get_var("uy")

        rot_fft = self.sim.state.state_spect.get_var("rot_fft")
        b_fft = self.sim.state.state_spect.get_var("b_fft")
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)

        px_b_fft, py_b_fft = oper.gradfft_from_fft(b_fft)
        px_b = oper.ifft2(px_b_fft)
        py_b = oper.ifft2(py_b_fft)

        Fb = -ux * px_b - uy * py_b
        Fb_fft = oper.fft2(Fb)
        oper.dealiasing(Fb_fft)

        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot = oper.ifft2(px_rot_fft)
        py_rot = oper.ifft2(py_rot_fft)

        px_ux_fft, py_ux_fft = oper.gradfft_from_fft(ux_fft)
        px_ux = oper.ifft2(px_ux_fft)
        py_ux = oper.ifft2(py_ux_fft)

        px_uy_fft, py_uy_fft = oper.gradfft_from_fft(uy_fft)
        px_uy = oper.ifft2(px_uy_fft)
        py_uy = oper.ifft2(py_uy_fft)

        Frot = -ux * px_rot - uy * (py_rot + self.params.beta)
        Frot_fft = oper.fft2(Frot)
        oper.dealiasing(Frot_fft)

        Fx = -ux * px_ux - uy * (py_ux)
        Fx_fft = oper.fft2(Fx)
        oper.dealiasing(Fx_fft)

        Fy = -ux * px_uy - uy * (py_uy)
        Fy_fft = oper.fft2(Fy)
        oper.dealiasing(Fy_fft)

        # Frequency dissipation viscosity
        f_d, f_d_hypo = self.sim.compute_freq_diss()
        freq_diss_EK = f_d + f_d_hypo

        # Energy budget terms. Nonlinear transfer terms, exchange kinetic and
        # potential energy B, dissipation terms.
        transferZ_fft = (
            np.real(rot_fft.conj() * Frot_fft + rot_fft * Frot_fft.conj()) / 2.0
        )
        transferEKu_fft = np.real(ux_fft.conj() * Fx_fft)
        transferEKv_fft = np.real(uy_fft.conj() * Fy_fft)
        B_fft = np.real(uy_fft.conj() * b_fft)

        if self.params.N == 0:
            transferEA_fft = np.zeros_like(transferZ_fft)
        else:
            transferEA_fft = (1 / self.params.N**2) * np.real(
                b_fft.conj() * Fb_fft
            )

        dissEKu_fft = np.real(freq_diss_EK * (ux_fft.conj() * ux_fft))
        dissEKv_fft = np.real(freq_diss_EK * (uy_fft.conj() * uy_fft))

        dissEK_fft = np.real(
            freq_diss_EK
            * (
                ux_fft.conj() * ux_fft
                + ux_fft * ux_fft.conj()
                + uy_fft.conj() * uy_fft
                + uy_fft * uy_fft.conj()
            )
            / 2.0
        )
        if self.params.N == 0:
            dissEA_fft = np.zeros_like(dissEK_fft)
        else:
            dissEA_fft = (1 / self.params.N**2) * np.real(
                freq_diss_EK * (b_fft.conj() * b_fft)
            )

        transferEK_fft = (
            np.real(
                ux_fft.conj() * Fx_fft
                + ux_fft * Fx_fft.conj()
                + uy_fft.conj() * Fy_fft
                + uy_fft * Fy_fft.conj()
            )
            / 2.0
        )

        # Transfer spectrum 1D Kinetic energy, potential energy and exchange
        # energy
        transferEK_kx, transferEK_ky = self.spectra1D_from_fft(transferEK_fft)
        transferEKu_kx, transferEKu_ky = self.spectra1D_from_fft(transferEKu_fft)
        transferEKv_kx, transferEKv_ky = self.spectra1D_from_fft(transferEKv_fft)
        transferEA_kx, transferEA_ky = self.spectra1D_from_fft(transferEA_fft)
        B_kx, B_ky = self.spectra1D_from_fft(B_fft)

        dissEK_kx, dissEK_ky = self.spectra1D_from_fft(dissEK_fft)
        dissEKu_kx, dissEKu_ky = self.spectra1D_from_fft(dissEKu_fft)
        dissEKv_kx, dissEKv_ky = self.spectra1D_from_fft(dissEKv_fft)
        dissEA_kx, dissEA_ky = self.spectra1D_from_fft(dissEA_fft)

        # Transfer spectrum shell mean
        transferEK_2d = self.spectrum2D_from_fft(transferEK_fft)
        transferEKu_2d = self.spectrum2D_from_fft(transferEKu_fft)
        transferEKv_2d = self.spectrum2D_from_fft(transferEKv_fft)
        transferEA_2d = self.spectrum2D_from_fft(transferEA_fft)
        B_2d = self.spectrum2D_from_fft(B_fft)
        dissEKu_2d = self.spectrum2D_from_fft(dissEKu_fft)
        dissEKv_2d = self.spectrum2D_from_fft(dissEKv_fft)
        dissEA_2d = self.spectrum2D_from_fft(dissEA_fft)
        transferZ_2d = self.spectrum2D_from_fft(transferZ_fft)

        # Dissipation rate at one time
        epsilon_kx = dissEKu_kx.sum() + dissEKv_kx.sum() + dissEA_kx.sum()
        epsilon_ky = dissEKu_ky.sum() + dissEKv_ky.sum() + dissEA_ky.sum()

        # Variables saved in a dictionary
        dict_results = {
            "transferEK_kx": transferEK_kx,
            "transferEK_ky": transferEK_ky,
            "transferEKu_kx": transferEKu_kx,
            "transferEKu_ky": transferEKu_ky,
            "transferEKv_kx": transferEKv_kx,
            "transferEKv_ky": transferEKv_ky,
            "transferEKu_2d": transferEKu_2d,
            "transferEKv_2d": transferEKv_2d,
            "transferEK_2d": transferEK_2d,
            "transferEA_kx": transferEA_kx,
            "transferEA_ky": transferEA_ky,
            "transferEA_2d": transferEA_2d,
            "transferZ_2d": transferZ_2d,
            "B_kx": B_kx,
            "B_ky": B_ky,
            "B_2d": B_2d,
            "dissEK_kx": dissEK_kx,
            "dissEK_ky": dissEK_ky,
            "dissEKu_kx": dissEKu_kx,
            "dissEKu_ky": dissEKu_ky,
            "dissEKu_2d": dissEKu_2d,
            "dissEKv_kx": dissEKv_kx,
            "dissEKv_ky": dissEKv_ky,
            "dissEKv_2d": dissEKv_2d,
            "dissEA_kx": dissEA_kx,
            "dissEA_ky": dissEA_ky,
            "dissEA_2d": dissEA_2d,
            "epsilon_kx": epsilon_kx,
            "epsilon_ky": epsilon_ky,
        }

        if mpi.rank == 0:
            small_value = 3e-10
            for k, v in dict_results.items():
                if k.startswith("transfer"):
                    if abs(v.sum()) > small_value:
                        print("warning: (abs(v.sum()) > small_value) for " + k)
                        print("k = ", k)
                        print("abs(v.sum()) = ", abs(v.sum()))

        return dict_results

    def _online_plot_saving(self, dict_results):

        transfer2D_EA = dict_results["transferEA_2d"]
        transfer2D_EK = dict_results["transferEK_2d"]
        transfer2D_E = transfer2D_EA + transfer2D_EK
        transfer2D_Z = dict_results["transferZ_2d"]
        khE = self.oper.khE
        PiE = cumsum_inv(transfer2D_E) * self.oper.deltak
        PiZ = cumsum_inv(transfer2D_Z) * self.oper.deltak
        self.axe_a.plot(khE + khE[1], PiE, "k")
        self.axe_b.plot(khE + khE[1], PiZ, "g")

    def load_mean(self, tmin=None, tmax=None, keys_to_load=None):
        """Loads data spect_energy_budget.

        It computes the mean between tmin and tmax."""
        means = {}
        with h5py.File(self.path_file, "r") as file:
            times = file["times"][...]
            nt = len(times)
            if tmin is None:
                imin_plot = 0
            else:
                imin_plot = np.argmin(abs(times - tmin))
            if tmax is None:
                imax_plot = nt - 1
            else:
                imax_plot = np.argmin(abs(times - tmax))

            tmin = times[imin_plot]
            tmax = times[imax_plot]

            print(
                "compute mean spectral energy budget\n"
                f"tmin = {tmin:8.6g} ; tmax = {tmax:8.6g}\n"
                f"imin = {imin_plot:8d} ; imax = {imax_plot:8d}"
            )

            for key in list(file.keys()):
                if key.startswith("k"):
                    means[key] = file[key][...]

            if keys_to_load is not None:
                if isinstance(keys_to_load, str):
                    keys_to_load = [keys_to_load]
                for key in keys_to_load:
                    if key not in file.keys():
                        print(key, file.keys())
                        raise ValueError
            else:
                keys_to_load = [
                    key
                    for key in file.keys()
                    if key != "times" and not key.startswith(("k", "info"))
                ]

            for key in keys_to_load:
                spect = file[key][imin_plot : imax_plot + 1].mean(0)
                means[key] = spect
        return means

    def compute_fluxes_mean(self, tmin=None, tmax=None):
        """compute the fluxes mean."""
        # Load data from file
        with h5py.File(self.path_file, "r") as file:
            times = file["times"][...]
            nt = len(times)
            kxE = file["kxE"][...]
            kyE = file["kyE"][...]

            transferEK_kx = file["transferEK_kx"][...]
            transferEK_ky = file["transferEK_ky"][...]
            transferEA_kx = file["transferEA_kx"][...]
            transferEA_ky = file["transferEA_ky"][...]

            dissEK_kx = file["dissEK_kx"][...]
            dissEA_kx = file["dissEA_kx"][...]

            dissEK_ky = file["dissEK_ky"][...]
            dissEA_ky = file["dissEA_ky"][...]

            conv_kx = file["B_kx"][...]
            conv_ky = file["B_ky"][...]

        if tmin is None:
            imin_plot = 0
        else:
            imin_plot = np.argmin(abs(times - tmin))
        if tmax is None:
            imax_plot = nt - 1
        else:
            imax_plot = np.argmin(abs(times - tmax))

        tmin = times[imin_plot]
        tmax = times[imax_plot]

        # compute means kx
        transferEK_kx = transferEK_kx[imin_plot : imax_plot + 1].mean(0)
        transferEA_kx = transferEA_kx[imin_plot : imax_plot + 1].mean(0)

        dissEK_kx = dissEK_kx[imin_plot : imax_plot + 1].mean(0)
        dissEA_kx = dissEA_kx[imin_plot : imax_plot + 1].mean(0)

        id_kx_dealiasing = np.argmin(kxE - self.sim.oper.kxmax_dealiasing) - 1
        id_ky_dealiasing = np.argmin(kyE - self.sim.oper.kymax_dealiasing) - 1

        conv_kx = conv_kx[imin_plot : imax_plot + 1].mean(0)[:id_kx_dealiasing]
        conv_kx = cumsum_inv(conv_kx[1:]) * self.oper.deltakx

        transferEK_kx = transferEK_kx[:id_kx_dealiasing]
        transferEA_kx = transferEA_kx[:id_kx_dealiasing]
        dissEK_kx = dissEK_kx[:id_kx_dealiasing]
        dissEA_kx = dissEA_kx[:id_kx_dealiasing]
        kxE = kxE[:id_kx_dealiasing]

        # Transfer and dissipation kx
        PiEK_kx = cumsum_inv(transferEK_kx[1:]) * self.oper.deltakx
        PiEA_kx = cumsum_inv(transferEA_kx[1:]) * self.oper.deltakx

        DissEK_kx = dissEK_kx[1:].cumsum() * self.oper.deltakx
        DissEA_kx = dissEA_kx[1:].cumsum() * self.oper.deltakx

        # compute means ky
        transferEK_ky = transferEK_ky[imin_plot : imax_plot + 1].mean(0)
        transferEA_ky = transferEA_ky[imin_plot : imax_plot + 1].mean(0)

        dissEK_ky = dissEK_ky[imin_plot : imax_plot + 1].mean(0)
        dissEA_ky = dissEA_ky[imin_plot : imax_plot + 1].mean(0)

        conv_ky = conv_ky[imin_plot : imax_plot + 1].mean(0)[:id_ky_dealiasing]
        conv_ky = cumsum_inv(conv_ky[1:]) * self.oper.deltaky

        transferEK_ky = transferEK_ky[:id_ky_dealiasing]
        transferEA_ky = transferEA_ky[:id_ky_dealiasing]
        dissEK_ky = dissEK_ky[:id_ky_dealiasing]
        dissEA_ky = dissEA_ky[:id_ky_dealiasing]
        kyE = kyE[:id_ky_dealiasing]

        # Transfer and dissipation ky
        PiEK_ky = cumsum_inv(transferEK_ky[1:]) * self.oper.deltaky
        PiEA_ky = cumsum_inv(transferEA_ky[1:]) * self.oper.deltaky

        DissEK_ky = dissEK_ky[1:].cumsum() * self.oper.deltaky
        DissEA_ky = dissEA_ky[1:].cumsum() * self.oper.deltaky

        # save data in dictionary
        fluxes = {
            "kxE": kxE,
            "kyE": kyE,
            "PiEAmean_x": PiEA_kx,
            "PiEKmean_x": PiEK_kx,
            "DissEAmean_x": DissEA_kx,
            "DissEKmean_x": DissEK_kx,
            "PiEAmean_y": PiEA_ky,
            "PiEKmean_y": PiEK_ky,
            "DissEAmean_y": DissEA_ky,
            "DissEKmean_y": DissEK_ky,
            "Conv_x": conv_kx,
            "Conv_y": conv_ky,
        }
        return fluxes

    def plot(
        self, tmin=None, tmax=None, delta_t=2, plot_diss_EK=False, plot_conv=False
    ):
        """Plot the energy budget."""
        # load data from file
        with h5py.File(self.path_file, "r") as file:
            times = file["times"][...]
            nt = len(times)

        if tmin is None:
            imin_plot = 0
        else:
            imin_plot = np.argmin(abs(times - tmin))
        if tmax is None:
            imax_plot = nt - 1
        else:
            imax_plot = np.argmin(abs(times - tmax))

        # Average from tmin and tmax for plot
        delta_t_save = np.mean(times[1:] - times[0:-1])
        delta_i_plot = int(np.round(delta_t / delta_t_save))

        if delta_i_plot == 0 and delta_t != 0.0:
            delta_i_plot = 1
        delta_t = delta_i_plot * delta_t_save

        print(f"plot(tmin={tmin}, tmax={tmax}, delta_t={delta_t:.2f})")

        tmin = times[imin_plot]
        tmax = times[imax_plot]
        print(
            "plot spectral energy budget\n"
            f"tmin = {tmin:8.6g} ; tmax = {tmax:8.6g} ;\n"
            f"delta_t = {delta_t:8.6g} ; imin = {imin_plot:8d} ;\n"
            f"imax = {imax_plot:8d} ; delta_i = {delta_i_plot:8d}"
        )

        fluxes = self.compute_fluxes_mean(tmin, tmax)

        # Parameters of figure 1
        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel("$k_x/k_{f,x}$")
        ax1.set_ylabel(r"$\Pi(k_x)/\epsilon$")
        ax1.set_xscale("log")
        ax1.set_yscale("linear")
        ax1.set_title("Spectral energy budget\n" + self.output.summary_simul)

        # Parameters of figure 2
        fig, ax2 = self.output.figure_axe()
        ax2.set_xlabel("$k_z/k_{f,z}$")
        ax2.set_ylabel(r"$\Pi(k_z)/\epsilon$")
        ax2.set_xscale("log")
        ax2.set_yscale("linear")
        ax2.set_title("Spectral energy budget\n" + self.output.summary_simul)

        # Compute time average dissipation to normalize fluxes
        spatial_mean_results = self.output.spatial_means.load()

        t = spatial_mean_results["t"]
        epsK_tot = spatial_mean_results["epsK_tot"]
        epsA_tot = spatial_mean_results["epsA_tot"]
        eps_tot = epsK_tot + epsA_tot

        imin_plot_spatial_times = np.argmin(abs(t - tmin))
        imax_plot_spatial_times = np.argmin(abs(t - tmax))

        eps_tot_time_average = eps_tot[
            imin_plot_spatial_times : imax_plot_spatial_times + 1
        ].mean(0)

        # calculate forcing wave-number k_f to normalize k_x and k_z
        pforcing = self.sim.params.forcing
        if pforcing.enable and pforcing.type == "tcrandom_anisotropic":
            nkmax = pforcing.nkmax_forcing
            nkmin = pforcing.nkmin_forcing
            k_f = ((nkmax + nkmin) / 2) * self.sim.oper.deltak
            angle = pforcing.tcrandom_anisotropic.angle
            try:
                if angle.endswith("°"):
                    angle = np.pi / 180 * float(angle[:-1])
            except AttributeError:
                pass
            k_fx = np.sin(angle) * k_f
            k_fy = np.cos(angle) * k_f

            # Band forcing region kx
            k_fxmin = nkmin * self.sim.oper.deltak * np.sin(angle)
            k_fxmin = max(k_fxmin, self.sim.oper.deltakx)
            k_fxmax = nkmax * self.sim.oper.deltak * np.sin(angle)

            # Band forcing region ky
            k_fymin = nkmin * self.sim.oper.deltak * np.cos(angle)
            k_fymin = max(k_fymin, self.sim.oper.deltaky)
            k_fymax = nkmax * self.sim.oper.deltak * np.cos(angle)

            # plot forcing range
            ax1.axvspan(k_fxmin / k_fx, k_fxmax / k_fx, alpha=0.15, color="black")
            ax2.axvspan(k_fymin / k_fy, k_fymax / k_fy, alpha=0.15, color="black")

            # Plot ozmidov scale
            k_o = (self.params.N**3 / self.params.forcing.forcing_rate) ** (
                1 / 2
            )
            ax1.axvline(x=k_o / k_fx, color="black", linestyle="--")
            ax2.axvline(x=k_o / k_fy, color="black", linestyle="--")
            ax1.text((k_o / k_fx) + 2, 1, r"$k_o$")
            ax2.text((k_o / k_fy) + 2, 1.7, r"$k_o$")

        else:
            print("Forcing is not enabled k_x and k_z are not normalized")
            k_fx = 1
            k_fy = 1

        # extract variables from the dictionary to plot
        kxE = fluxes["kxE"]
        kyE = fluxes["kyE"]

        PiEK_kx = fluxes["PiEKmean_x"]
        PiEA_kx = fluxes["PiEAmean_x"]

        DissEK_kx = fluxes["DissEKmean_x"]
        DissEA_kx = fluxes["DissEAmean_x"]

        PiEK_ky = fluxes["PiEKmean_y"]
        PiEA_ky = fluxes["PiEAmean_y"]

        DissEK_ky = fluxes["DissEKmean_y"]
        DissEA_ky = fluxes["DissEAmean_y"]

        conv_kx = fluxes["Conv_x"]
        conv_ky = fluxes["Conv_y"]

        def plot_x(x, y, style, label):
            ax1.plot(x[1:] / k_fx, y / eps_tot_time_average, style, label=label)

        plot_x(kxE, (PiEK_kx + PiEA_kx), "k--", r"$\Pi/\epsilon$")
        plot_x(kxE, PiEK_kx, "r", r"$\Pi_K/\epsilon$")
        plot_x(kxE, PiEA_kx, "b", r"$\Pi_A/\epsilon$")
        plot_x(kxE, (DissEK_kx + DissEA_kx), "g", r"$D/\epsilon$")

        if plot_diss_EK:
            plot_x(kxE, DissEK_kx, "r--", r"$D_K$")

        if plot_conv:
            plot_x(kxE, conv_kx, "c", r"$C$")

        ax1.axhline(y=0, color="k", linestyle=":")

        def plot_y(x, y, style, label):
            ax2.plot(x[1:] / k_fy, y / eps_tot_time_average, style, label=label)

        plot_y(kyE, (PiEK_ky + PiEA_ky), "k--", r"$\Pi/\epsilon$")
        plot_y(kyE, PiEK_ky, "r", r"$\Pi_K/\epsilon$")
        plot_y(kyE, PiEA_ky, "b", r"$\Pi_A/\epsilon$")
        plot_y(kyE, (DissEK_ky + DissEA_ky), "g", r"$D/\epsilon$")

        if plot_diss_EK:
            plot_y(kyE, DissEK_ky, "r--", r"$D_K$")

        if plot_conv:
            plot_y(kyE, conv_ky, "c", r"$C$")

        ax2.axhline(y=0, color="k", linestyle=":")

        # Compute k_b: L_b = U / N
        U = np.sqrt(np.mean(abs(self.sim.state.get_var("ux")) ** 2))
        k_b = self.sim.params.N / U
        ax1.axvline(x=k_b / k_fx, color="k", linestyle="--")
        ax2.axvline(x=k_b / k_fy, color="k", linestyle="--")
        ax1.text((k_b / k_fx) + 0.5, 1, r"$k_b$")
        ax2.text((k_b / k_fy) + 0.5, 1.7, r"$k_b$")

        ax1.legend()
        ax2.legend()
