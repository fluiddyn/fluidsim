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
            transferEA_fft = (1 / self.params.N ** 2) * np.real(
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
            dissEA_fft = (1 / self.params.N ** 2) * np.real(
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

    def plot(
        self, tmin=0, tmax=None, delta_t=2, plot_diss_EK=False, plot_conv=False
    ):
        """Plot the energy budget."""

        # Load data from file
        with h5py.File(self.path_file, "r") as f:
            times = f["times"].value
            kxE = f["kxE"].value
            kyE = f["kyE"].value

            dset_transferEK_kx = f["transferEK_kx"].value
            dset_transferEK_ky = f["transferEK_ky"].value
            dset_transferEA_kx = f["transferEA_kx"].value
            dset_transferEA_ky = f["transferEA_ky"].value

            dset_dissEK_kx = f["dissEK_kx"].value
            dset_dissEA_kx = f["dissEA_kx"].value

            dset_dissEK_ky = f["dissEK_ky"].value
            dset_dissEA_ky = f["dissEA_ky"].value

            if plot_conv:
                dset_conv_kx = f["B_kx"].value
                dset_conv_ky = f["B_ky"].value

        if tmin is None:
            tmin = 0

        if tmax is None:
            tmax = np.max(times)

        # Average from tmin and tmax for plot
        delta_t_save = np.mean(times[1:] - times[0:-1])
        delta_i_plot = int(np.round(delta_t / delta_t_save))

        if delta_i_plot == 0 and delta_t != 0.0:
            delta_i_plot = 1
        delta_t = delta_i_plot * delta_t_save

        imin_plot = np.argmin(abs(times - tmin))
        imax_plot = np.argmin(abs(times - tmax))

        to_print = "plot(tmin={}, tmax={}, delta_t={:.2f})".format(
            tmin, tmax, delta_t
        )
        print(to_print)

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]
        print(
            """plot spectral energy budget
            tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
            imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
                tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
            )
        )

        # Parameters of the figure
        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel("$k_x$")
        ax1.set_ylabel(r"$\Pi$")
        ax1.set_xscale("log")
        ax1.set_yscale("linear")
        ax1.set_title(
            "Spectral energy budget, solver "
            + self.output.name_solver
            + f", nh = {self.nx:5d}"
        )

        transferEK_kx = dset_transferEK_kx[imin_plot : imax_plot + 1].mean(0)
        transferEA_kx = dset_transferEA_kx[imin_plot : imax_plot + 1].mean(0)

        dissEK_kx = dset_dissEK_kx[imin_plot : imax_plot + 1].mean(0)
        dissEA_kx = dset_dissEA_kx[imin_plot : imax_plot + 1].mean(0)

        id_kx_dealiasing = np.argmin(kxE - self.sim.oper.kxmax_dealiasing) - 1
        id_ky_dealiasing = np.argmin(kyE - self.sim.oper.kymax_dealiasing) - 1

        transferEK_kx = transferEK_kx[:id_kx_dealiasing]
        transferEA_kx = transferEA_kx[:id_kx_dealiasing]
        dissEK_kx = dissEK_kx[:id_kx_dealiasing]
        dissEA_kx = dissEA_kx[:id_kx_dealiasing]
        kxE = kxE[:id_kx_dealiasing]

        PiEK_kx = cumsum_inv(transferEK_kx[1:]) * self.oper.deltakx
        PiEA_kx = cumsum_inv(transferEA_kx[1:]) * self.oper.deltakx

        DissEK_kx = dissEK_kx[1:].cumsum() * self.oper.deltakx
        DissEA_kx = dissEA_kx[1:].cumsum() * self.oper.deltakx

        ax1.plot(kxE[1:], PiEK_kx + PiEA_kx, "k", label=r"$\Pi$")
        ax1.plot(kxE[1:], PiEK_kx, "r", label=r"$\Pi_K$")
        ax1.plot(kxE[1:], PiEA_kx, "b", label=r"$\Pi_A$")
        ax1.plot(kxE[1:], DissEK_kx + DissEA_kx, "k--", label=r"$D$")

        if plot_diss_EK:
            ax1.plot(kxE[1:], DissEK_kx, "r--", label=r"$D_K$")

        if plot_conv:
            conv_kx = dset_conv_kx[imin_plot : imax_plot + 1].mean(0)[
                :id_kx_dealiasing
            ]
            conv_kx = cumsum_inv(conv_kx[1:]) * self.oper.deltakx
            ax1.plot(kxE[1:], conv_kx, "c", label=r"$C$")

        ax1.axhline(y=0, color="k", linestyle=":")

        # Parameters of the figure
        fig, ax2 = self.output.figure_axe()
        ax2.set_xlabel("$k_z$")
        ax2.set_ylabel(r"$\Pi$")
        ax2.set_xscale("log")
        ax2.set_yscale("linear")
        ax2.set_title(
            "Spectral energy budget, solver "
            + self.output.name_solver
            + f", nh = {self.nx:5d}"
        )

        transferEK_ky = dset_transferEK_ky[imin_plot : imax_plot + 1].mean(0)
        transferEA_ky = dset_transferEA_ky[imin_plot : imax_plot + 1].mean(0)

        dissEK_ky = dset_dissEK_ky[imin_plot : imax_plot + 1].mean(0)
        dissEA_ky = dset_dissEA_ky[imin_plot : imax_plot + 1].mean(0)

        transferEK_ky = transferEK_ky[:id_ky_dealiasing]
        transferEA_ky = transferEA_ky[:id_ky_dealiasing]
        dissEK_ky = dissEK_ky[:id_ky_dealiasing]
        dissEA_ky = dissEA_ky[:id_ky_dealiasing]
        kyE = kyE[:id_ky_dealiasing]

        PiEK_ky = cumsum_inv(transferEK_ky[1:]) * self.oper.deltaky
        PiEA_ky = cumsum_inv(transferEA_ky[1:]) * self.oper.deltaky

        DissEK_ky = dissEK_ky[1:].cumsum() * self.oper.deltaky
        DissEA_ky = dissEA_ky[1:].cumsum() * self.oper.deltaky

        ax2.plot(kyE[1:], PiEK_ky + PiEA_ky, "k", label=r"$\Pi$")
        ax2.plot(kyE[1:], PiEK_ky, "r", label=r"$\Pi_K$")
        ax2.plot(kyE[1:], PiEA_ky, "b", label=r"$\Pi_A$")
        ax2.plot(kyE[1:], DissEK_ky + DissEA_ky, "k--", label=r"$D$")

        if plot_diss_EK:
            ax2.plot(kyE[1:], DissEK_ky, "r--", label=r"$D_K$")

        if plot_conv:
            conv_ky = dset_conv_ky[imin_plot : imax_plot + 1].mean(0)[
                :id_ky_dealiasing
            ]
            conv_ky = cumsum_inv(conv_ky[1:]) * self.oper.deltaky
            ax2.plot(kyE[1:], conv_ky, "c", label=r"$C$")

        ax2.axhline(y=0, color="k", linestyle=":")

        # Plot forcing wave-number k_f
        nkmax = self.sim.params.forcing.nkmax_forcing
        nkmin = self.sim.params.forcing.nkmin_forcing
        k_f = ((nkmax + nkmin) / 2) * self.sim.oper.deltak

        pforcing = self.sim.params.forcing
        if pforcing.enable and pforcing.type == "tcrandom_anisotropic":
            angle = pforcing.tcrandom_anisotropic.angle
            try:
                if angle.endswith("°"):
                    angle = np.pi / 180 * float(angle[:-1])
            except AttributeError:
                pass
            k_fx = np.sin(angle) * k_f
            k_fy = np.cos(angle) * k_f
            # ax1.axvline(x=k_fx, color="y", linestyle="-.", label="$k_{f,x}$")
            # ax2.axvline(x=k_fy, color="y", linestyle="-.", label="$k_{f,z}$")

            # Band forcing region kx
            k_fxmin = nkmin * self.sim.oper.deltak * np.sin(angle)
            k_fxmax = nkmax * self.sim.oper.deltak * np.sin(angle)

            # Band forcing region ky
            k_fymin = nkmin * self.sim.oper.deltak * np.cos(angle)
            k_fymax = nkmax * self.sim.oper.deltak * np.cos(angle)

            ax1.axvspan(k_fxmin, k_fxmax, alpha=0.15, color="black")
            ax2.axvspan(k_fymin, k_fymax, alpha=0.15, color="black")

        # Compute k_b: L_b = U / N
        U = np.sqrt(np.mean(abs(self.sim.state.get_var("ux")) ** 2))
        k_b = self.sim.params.N / U
        ax1.axvline(x=k_b, color="y", linestyle="--", label="$k_b$")
        ax2.axvline(x=k_b, color="y", linestyle="--", label="$k_b$")

        ax1.legend()
        ax2.legend()

    def load_mean(self, tmin=None, tmax=None):
        """
        Loads data spect_energy_budget.
        
        It computes the mean between tmin and tmax.
        """

        with h5py.File(self.path_file, "r") as f:
            times = f["times"].value
            kxE = f["kxE"].value
            kyE = f["kyE"].value
            dset_dissEK_kx = f["dissEK_kx"].value
            dset_dissEA_kx = f["dissEA_kx"].value
            dset_dissEK_ky = f["dissEK_ky"].value
            dset_dissEA_ky = f["dissEA_ky"].value

            # If tmin and tmax is None
            if tmin is None:
                tmin = np.min(times)

            if tmax is None:
                tmax = np.max(times)

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            dset_dissE_kx = dset_dissEK_kx + dset_dissEA_kx
            dissE_kx = dset_dissE_kx[imin_plot : imax_plot + 1].mean(0)

            dset_dissE_ky = dset_dissEK_ky + dset_dissEA_ky
            dissE_ky = dset_dissE_ky[imin_plot : imax_plot + 1].mean(0)

        return kxE, kyE, dissE_kx, dissE_ky
