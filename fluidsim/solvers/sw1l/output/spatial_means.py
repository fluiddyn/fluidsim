import os

import numpy as np
import matplotlib.pyplot as plt

from fluiddyn.util import mpi
from fluidsim.base.output.spatial_means import SpatialMeansJSON, inner_prod
from ._old_spatial_means import load_txt as _old_load_txt


class SpatialMeansMSW1L(SpatialMeansJSON):
    """Handle the saving of spatial mean quantities.

    Viz. total energy, K.E., A.P.E. and Charney potential enstrophy. It also
    handles the computation of forcing and dissipation rates for
    sw1l.modified solver

    """

    def __init__(self, output):

        params = output.sim.params
        self.c2 = params.c2
        self.f = params.f
        if mpi.rank == 0:
            self._result = {}

        super().__init__(output)

    def _save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim

        if mpi.rank == 0:
            self._result["t"] = tsim

        (
            energyK_fft,
            energyA_fft,
            energyKr_fft,
        ) = self.output.compute_energies_fft()
        energyK = self.sum_wavenumbers(energyK_fft)
        energyA = self.sum_wavenumbers(energyA_fft)
        energyKr = self.sum_wavenumbers(energyKr_fft)
        energy = energyK + energyA

        CharneyPE_fft = self.output.compute_CharneyPE_fft()
        CharneyPE = self.sum_wavenumbers(CharneyPE_fft)

        if mpi.rank == 0:
            self._result.update(
                {
                    "E": energy,
                    "CPE": CharneyPE,
                    "EK": energyK,
                    "EA": energyA,
                    "EKr": energyKr,
                }
            )

        # Compute and save dissipation rates.
        self.treat_dissipation_rates(energyK_fft, energyA_fft, CharneyPE_fft)

        # Compute and save conversion rates.
        self.treat_conversion()

        # Compute and save skewness and kurtosis.
        eta = self.sim.state.state_phys.get_var("eta")
        meaneta2 = 2.0 / self.c2 * energyA
        if meaneta2 == 0:
            skew_eta = 0.0
            kurt_eta = 0.0
        else:
            skew_eta = np.mean(eta**3) / meaneta2 ** (3.0 / 2)
            kurt_eta = np.mean(eta**4) / meaneta2 ** (2)

        ux = self.sim.state.state_phys.get_var("ux")
        uy = self.sim.state.state_phys.get_var("uy")
        ux_fft = self.sim.oper.fft2(ux)
        uy_fft = self.sim.oper.fft2(uy)
        rot_fft = self.sim.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        rot = self.sim.oper.ifft2(rot_fft)
        div_fft = self.sim.oper.divfft_from_vecfft(ux_fft, uy_fft)
        div = self.sim.oper.ifft2(div_fft)
        meanrot2 = self.sum_wavenumbers(abs(rot_fft) ** 2)
        meandiv2 = self.sum_wavenumbers(abs(div_fft) ** 2)
        if meanrot2 == 0:
            skew_rot = 0.0
            kurt_rot = 0.0
        else:
            skew_rot = np.mean(rot**3) / meanrot2 ** (3.0 / 2)
            kurt_rot = np.mean(rot**4) / meanrot2 ** (2)

        if meandiv2 == 0:
            skew_div = 0.0
            kurt_div = 0.0
        else:
            skew_div = np.mean(div**3) / meandiv2 ** (3.0 / 2)
            kurt_div = np.mean(div**4) / meandiv2 ** (2)

        if mpi.rank == 0:
            self._result.update(
                {
                    "skew_eta": skew_eta,
                    "kurt_eta": kurt_eta,
                    "skew_rot": skew_rot,
                    "kurt_rot": kurt_rot,
                    "skew_div": skew_div,
                    "kurt_div": kurt_div,
                }
            )

        if self.sim.params.forcing.enable:
            self.treat_forcing()

        if mpi.rank == 0:
            super()._save_one_time(self._result)
            self.file.flush()
            os.fsync(self.file.fileno())

        if self.has_to_plot and mpi.rank == 0:
            self.axe_a.plot(tsim, energy, "k.")
            self.axe_a.plot(tsim, energyK, "r.")
            self.axe_a.plot(tsim, energyA, "b.")

            if tsim - self.t_last_show >= self.period_show:
                self.t_last_show = tsim
                fig = self.axe_a.get_figure()
                fig.canvas.draw()

    def treat_conversion(self):
        mean_global = self.sim.oper.mean_global

        c2 = self.sim.params.c2
        eta = self.sim.state.get_var("eta")
        div = self.sim.state.get_var("div")
        h = eta + 1

        Conv = c2 / 2 * mean_global(h**2 * div)
        c2eta1d = c2 * mean_global(eta * div)
        c2eta2d = c2 * mean_global(eta**2 * div)
        c2eta3d = c2 * mean_global(eta**3 * div)

        if mpi.rank == 0:
            self._result.update(
                {
                    "Conv": Conv,
                    "c2eta1d": c2eta1d,
                    "c2eta2d": c2eta2d,
                    "c2eta3d": c2eta3d,
                }
            )

    def treat_dissipation_rates(self, energyK_fft, energyA_fft, CharneyPE_fft):
        """Compute and save dissipation rates."""

        f_d, f_d_hypo = self.sim.compute_freq_diss()

        dict_eps = self.compute_dissipation_rates(
            f_d, f_d_hypo, energyK_fft, energyA_fft, CharneyPE_fft
        )

        if mpi.rank == 0:
            self._result.update(dict_eps)
            self._result["epsK_tot"] = epsK_tot = (
                dict_eps["epsK"] + dict_eps["epsK_hypo"]
            )
            self._result["epsA_tot"] = epsA_tot = (
                dict_eps["epsA"] + dict_eps["epsA_hypo"]
            )
            self._result["epsCPE_tot"] = (
                dict_eps["epsCPE"] + dict_eps["epsCPE_hypo"]
            )

            if self.has_to_plot:
                tsim = self.sim.time_stepping.t
                self.axe_b.plot(tsim, epsK_tot + epsA_tot, "k.")

    def compute_dissipation_rates(
        self, f_d, f_d_hypo, energyK_fft, energyA_fft, CharneyPE_fft
    ):

        epsK = self.sum_wavenumbers(f_d * 2 * energyK_fft)
        epsK_hypo = self.sum_wavenumbers(f_d_hypo * 2 * energyK_fft)
        epsA = self.sum_wavenumbers(f_d * 2 * energyA_fft)
        epsA_hypo = self.sum_wavenumbers(f_d_hypo * 2 * energyA_fft)
        epsCPE = self.sum_wavenumbers(f_d * 2 * CharneyPE_fft)
        epsCPE_hypo = self.sum_wavenumbers(f_d_hypo * 2 * CharneyPE_fft)

        dict_eps = {
            "epsK": epsK,
            "epsK_hypo": epsK_hypo,
            "epsA": epsA,
            "epsA_hypo": epsA_hypo,
            "epsCPE": epsCPE,
            "epsCPE_hypo": epsCPE_hypo,
        }
        return dict_eps

    def get_FxFyFetafft(self):
        forcing = self.sim.forcing
        set_keys = set(self.sim.state.keys_state_spect)

        if {"ux_fft", "uy_fft", "eta_fft"} == set_keys:
            Fx_fft = forcing("ux_fft")
            Fy_fft = forcing("uy_fft")
            Feta_fft = forcing("eta_fft")
        elif {"q_fft", "ap_fft", "am_fft"} == set_keys:
            Fx_fft, Fy_fft, Feta_fft = self.sim.oper.uxuyetafft_from_qapamfft(
                forcing("q_fft"), forcing("ap_fft"), forcing("am_fft")
            )
        elif {"ap_fft", "am_fft"} == set_keys:
            Fx_fft, Fy_fft, Feta_fft = self.sim.oper.uxuyetafft_from_afft(
                forcing("ap_fft") + forcing("am_fft")
            )
        elif {"div_fft", "eta_fft", "rot_fft"} == set_keys:
            Fx_fft, Fy_fft = self.sim.oper.vecfft_from_rotdivfft(
                forcing("rot_fft"), forcing("div_fft")
            )
            Feta_fft = forcing("eta_fft")
        else:
            raise NotImplementedError(
                "Not sure how to estimate forcing rate with "
                "keys_state_spect = {}".format(set_keys)
            )

        return Fx_fft, Fy_fft, Feta_fft

    def treat_forcing(self):
        """Save forcing injection rates."""
        get_var = self.sim.state.get_var
        ux_fft = get_var("ux_fft")
        uy_fft = get_var("uy_fft")
        eta_fft = get_var("eta_fft")
        Fx_fft, Fy_fft, Feta_fft = self.get_FxFyFetafft()
        deltat = self.sim.time_stepping.deltat

        PK1_fft = inner_prod(ux_fft, Fx_fft) + inner_prod(uy_fft, Fy_fft)
        PK2_fft = deltat / 2 * (abs(Fx_fft) ** 2 + abs(Fy_fft) ** 2)

        PK1 = self.sum_wavenumbers(PK1_fft)
        PK2 = self.sum_wavenumbers(PK2_fft)

        PA1_fft = self.c2 * inner_prod(eta_fft, Feta_fft)
        PA2_fft = deltat / 2 * self.c2 * (abs(Feta_fft) ** 2)

        PA1 = self.sum_wavenumbers(PA1_fft)
        PA2 = self.sum_wavenumbers(PA2_fft)

        if mpi.rank == 0:
            PK_tot = PK1 + PK2
            PA_tot = PA1 + PA2
            self._result.update(
                {
                    "PK1": PK1,
                    "PK2": PK2,
                    "PK_tot": PK_tot,
                    "PA1": PA1,
                    "PA2": PA2,
                    "PA_tot": PA_tot,
                }
            )

        if self.has_to_plot and mpi.rank == 0:
            tsim = self.sim.time_stepping.t
            self.axe_b.plot(tsim, PK_tot + PA_tot, "c.")

    def load(self):
        if self._file_exists():
            return super().load()
        else:
            dict_results = {"name_solver": self.output.name_solver}
            path_file = self.path_file.replace(".json", ".txt")
            if os.path.exists(path_file):
                return _old_load_txt(path_file, dict_results)
            else:
                raise FileNotFoundError(
                    f"Neither {self.path_file} nor {os.path.basename(path_file)}"
                    "exists"
                )

    def plot(self):
        dict_results = self.load()

        t = dict_results["t"]

        E = dict_results["E"]
        CPE = dict_results["CPE"]

        EK = dict_results["EK"]
        EA = dict_results["EA"]
        EKr = dict_results["EKr"]

        epsK = dict_results["epsK"]
        epsK_hypo = dict_results["epsK_hypo"]
        epsK_tot = dict_results["epsK_tot"]

        epsA = dict_results["epsA"]
        epsA_hypo = dict_results["epsA_hypo"]
        epsA_tot = dict_results["epsA_tot"]

        epsE = epsK + epsA
        epsE_hypo = epsK_hypo + epsA_hypo
        epsE_tot = epsK_tot + epsA_tot

        epsCPE = dict_results["epsCPE"]
        epsCPE_hypo = dict_results["epsCPE_hypo"]
        epsCPE_tot = dict_results["epsCPE_tot"]

        if "PK_tot" in dict_results:
            PK_tot = dict_results["PK_tot"]
            PA_tot = dict_results["PA_tot"]
            P_tot = PK_tot + PA_tot

        width_axe = 0.85
        height_axe = 0.37
        x_left_axe = 0.12
        z_bottom_axe = 0.56

        size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel("t")
        ax1.set_ylabel("$2E(t)/c^2$")
        ax1.set_title("mean energy\n" + self.output.summary_simul)
        norm = self.c2 / 2
        ax1.plot(t, E / norm, "k", linewidth=2, label="$E$")
        ax1.plot(t, EK / norm, "r", linewidth=1, label="$E_K$")
        ax1.plot(t, EA / norm, "b", linewidth=1, label="$E_A$")
        ax1.plot(t, EKr / norm, "r--", linewidth=1, label="$E_K^r$")
        ax1.plot(t, (EK - EKr) / norm, "r:", linewidth=1, label="$E_K^d$")
        ax1.legend()

        z_bottom_axe = 0.07
        size_axe[1] = z_bottom_axe
        ax2 = fig.add_axes(size_axe)
        ax2.set_xlabel("t")
        ax2.set_ylabel("Charney PE(t)")
        ax2.set_title("mean Charney PE(t)")
        ax2.plot(t, CPE, "k", linewidth=2)

        z_bottom_axe = 0.56
        size_axe[1] = z_bottom_axe
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel("t")
        ax1.set_ylabel(r"$P_E(t)$, $\epsilon(t)$")
        ax1.set_title("forcing and dissipation\n" + self.output.summary_simul)

        if "PK_tot" in dict_results:
            ax1.plot(t, P_tot, "c", linewidth=2, label="$P_{tot}$")

        ax1.plot(t, epsE, "k--", linewidth=2, label=r"$\epsilon$")
        ax1.plot(t, epsE_hypo, "g", linewidth=2, label=r"$\epsilon_{hypo}$")
        ax1.plot(t, epsE_tot, "k", linewidth=2, label=r"$\epsilon_{tot}$")

        ax1.legend(loc=2)

        z_bottom_axe = 0.07
        size_axe[1] = z_bottom_axe
        ax2 = fig.add_axes(size_axe)
        ax2.set_xlabel("t")
        ax2.set_ylabel(r"$\epsilon$ Charney PE(t)")
        ax2.set_title("dissipation Charney PE")
        ax2.plot(t, epsCPE, "k--", linewidth=2)
        ax2.plot(t, epsCPE_hypo, "g", linewidth=2)
        ax2.plot(t, epsCPE_tot, "r", linewidth=2)

    def plot_rates(self, keys="E"):
        """Plots the time history of the time derivative of a spatial mean,
        and also calculates the average of the same.

        Parameters
        ----------
        key : string or a list of strings

            Refers to the the spatial mean which you want to take time
            derivative of.  Legal value include:

            For ns2d ['E', 'Z']
            For sw1l ['E', 'EK', 'EA', 'EKr', 'CPE']

        Examples
        --------
        >>> plot_rates()
        >>> plot_rates('Z')
        >>> plot_rates(['E', 'Z'])
        >>> plot_rates(['E', 'EK', 'EA', 'EKr', 'CPE'])

        """

        dict_results = self.load()
        t = dict_results["t"]
        dt = np.gradient(t, 1.0)

        fig, axarr = plt.subplots(len(keys), sharex=True)
        i = 0
        for k in keys:
            E = dict_results[k]
            dE_dt = abs(np.gradient(E, 1.0) / dt)
            dE_dt_avg = "{:11.6e}".format(dE_dt.mean())
            try:
                axarr[i].semilogy(t, dE_dt, label=dE_dt_avg)
                axarr[i].set_ylabel(r"$\partial_t$" + keys[i])
                axarr[i].legend()
            # axarr[i].text(0.8, 0.9, 'mean = ' + dE_dt_avg, horizontalalignment='center', verticalalignment='center',)
            except TypeError:
                axarr.semilogy(t, dE_dt, label=dE_dt_avg)
                axarr.set_ylabel(keys)
                axarr.legend()
            i += 1

        try:
            axarr[i - 1].set_xlabel("t")
        except TypeError:
            axarr.set_xlabel("t")

        plt.draw()


class SpatialMeansSW1L(SpatialMeansMSW1L):
    """Handle the saving of spatial mean quantities.


    Viz. total energy, K.E., A.P.E. and Charney potential enstrophy. It also
    handles the computation of forcing and dissipation rates for sw1l solver.

    """

    def compute_dissipation_rates(
        self, f_d, f_d_hypo, energyK_fft, energyA_fft, CharneyPE_fft
    ):
        """Compute and save dissipation rates."""

        dict_eps = super().compute_dissipation_rates(
            f_d, f_d_hypo, energyK_fft, energyA_fft, CharneyPE_fft
        )

        (epsKsuppl, epsKsuppl_hypo) = self.compute_epsK(
            f_d, f_d_hypo, energyK_fft, dict_eps
        )

        dict_eps.update({"epsKsup": epsKsuppl, "epsKsup_hypo": epsKsuppl_hypo})
        return dict_eps

    def compute_epsK(self, f_d, f_d_hypo, energyK_fft, dict_eps):

        ux = self.sim.state.state_phys.get_var("ux")
        uy = self.sim.state.state_phys.get_var("uy")

        EKquad = 0.5 * (ux**2 + uy**2)
        EKquad_fft = self.sim.oper.fft2(EKquad)

        eta_fft = self.sim.state.get_var("eta_fft")

        epsKsuppl = self.sum_wavenumbers(f_d * inner_prod(EKquad_fft, eta_fft))

        epsKsuppl_hypo = self.sum_wavenumbers(
            f_d_hypo * inner_prod(EKquad_fft, eta_fft)
        )

        dict_eps["epsK"] += epsKsuppl
        dict_eps["epsK_hypo"] += epsKsuppl_hypo

        return epsKsuppl, epsKsuppl_hypo

    def treat_forcing(self):
        """
        Save forcing injection rates.
        """
        get_var = self.sim.state.get_var
        ux_fft = get_var("ux_fft")
        uy_fft = get_var("uy_fft")
        eta_fft = get_var("eta_fft")

        Fx_fft, Fy_fft, Feta_fft = self.get_FxFyFetafft()
        deltat = self.sim.time_stepping.deltat

        PA1_fft = self.c2 * inner_prod(eta_fft, Feta_fft)
        PA2_fft = deltat / 2 * self.c2 * (abs(Feta_fft) ** 2)

        PA1 = self.sum_wavenumbers(PA1_fft)
        PA2 = self.sum_wavenumbers(PA2_fft)

        Fx = self.sim.oper.ifft2(Fx_fft)
        Fy = self.sim.oper.ifft2(Fy_fft)
        Feta = self.sim.oper.ifft2(Feta_fft)

        eta = self.sim.state.state_phys.get_var("eta")
        h = eta + 1.0

        ux = self.sim.state.state_phys.get_var("ux")
        uy = self.sim.state.state_phys.get_var("uy")

        FetaFx_fft = self.sim.oper.fft2(Feta * Fx)
        FetaFy_fft = self.sim.oper.fft2(Feta * Fy)

        Jx_fft = self.sim.oper.fft2(h * ux)
        Jy_fft = self.sim.oper.fft2(h * uy)

        FJx_fft = self.sim.oper.fft2(h * Fx + Feta * ux)
        FJy_fft = self.sim.oper.fft2(h * Fy + Feta * uy)

        PK1_fft = 0.5 * (
            inner_prod(Jx_fft, Fx_fft)
            + inner_prod(Jy_fft, Fy_fft)
            + inner_prod(ux_fft, FJx_fft)
            + inner_prod(uy_fft, FJy_fft)
        )
        PK2_fft = (
            deltat
            / 2
            * (
                0.5 * (inner_prod(Fx_fft, FJx_fft) + inner_prod(Fy_fft, FJy_fft))
                + inner_prod(ux_fft, FetaFx_fft)
                + inner_prod(uy_fft, FetaFy_fft)
            )
        )

        PK1 = self.sum_wavenumbers(PK1_fft)
        PK2 = self.sum_wavenumbers(PK2_fft)

        if mpi.rank == 0:

            PK_tot = PK1 + PK2
            PA_tot = PA1 + PA2
            self._result.update(
                {
                    "PK1": PK1,
                    "PK2": PK2,
                    "PK_tot": PK_tot,
                    "PA1": PA1,
                    "PA2": PA2,
                    "PA_tot": PA_tot,
                }
            )

        if self.has_to_plot and mpi.rank == 0:
            tsim = self.sim.time_stepping.t
            self.axe_b.plot(tsim, PK_tot + PA_tot, "c.")
