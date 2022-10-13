"""Spatial means output (:mod:`fluidsim.solvers.ns2d.output.spatial_means`)
===========================================================================

.. autoclass:: SpatialMeansNS2D
   :members:
   :private-members:

"""

import os
import numpy as np
import matplotlib.pyplot as plt


from fluiddyn.util import mpi

from fluidsim.base.output.spatial_means import SpatialMeansBase


class SpatialMeansNS2D(SpatialMeansBase):
    """Spatial means output."""

    def _save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim

        energy_fft = self.output.compute_energy_fft()
        enstrophy_fft = self.output.compute_enstrophy_fft()
        energy = self.sum_wavenumbers(energy_fft)
        enstrophy = self.sum_wavenumbers(enstrophy_fft)

        f_d, f_d_hypo = self.sim.compute_freq_diss()
        epsK = self.sum_wavenumbers(f_d * 2 * energy_fft)
        epsK_hypo = self.sum_wavenumbers(f_d_hypo * 2 * energy_fft)
        epsZ = self.sum_wavenumbers(f_d * 2 * enstrophy_fft)
        epsZ_hypo = self.sum_wavenumbers(f_d_hypo * 2 * enstrophy_fft)

        if self.sim.params.forcing.enable:
            deltat = self.sim.time_stepping.deltat
            Frot_fft = self.sim.forcing.get_forcing().get_var("rot_fft")
            Fx_fft, Fy_fft = self.vecfft_from_rotfft(Frot_fft)

            rot_fft = self.sim.state.state_spect.get_var("rot_fft")
            ux_fft, uy_fft = self.vecfft_from_rotfft(rot_fft)

            PZ1_fft = np.real(rot_fft.conj() * Frot_fft)
            PZ2_fft = abs(Frot_fft) ** 2

            PZ1 = self.sum_wavenumbers(PZ1_fft)
            PZ2 = deltat / 2 * self.sum_wavenumbers(PZ2_fft)

            PK1_fft = (
                np.real(
                    ux_fft.conj() * Fx_fft
                    + ux_fft * Fx_fft.conj()
                    + uy_fft.conj() * Fy_fft
                    + uy_fft * Fy_fft.conj()
                )
                / 2
            )
            PK2_fft = (abs(Fx_fft) ** 2 + abs(Fy_fft) ** 2) * deltat / 2

            PK1 = self.sum_wavenumbers(PK1_fft)
            PK2 = self.sum_wavenumbers(PK2_fft)

        if mpi.rank == 0:
            epsK_tot = epsK + epsK_hypo

            self.file.write(f"####\ntime = {tsim:11.5e}\n")
            to_print = (
                "E    = {:11.5e} ; Z         = {:11.5e} \n"
                "epsK = {:11.5e} ; epsK_hypo = {:11.5e} ; epsK_tot = {:11.5e} \n"
                "epsZ = {:11.5e} ; epsZ_hypo = {:11.5e} ; epsZ_tot = {:11.5e} \n"
            ).format(
                energy,
                enstrophy,
                epsK,
                epsK_hypo,
                epsK + epsK_hypo,
                epsZ,
                epsZ_hypo,
                epsZ + epsZ_hypo,
            )
            self.file.write(to_print)

            if self.sim.params.forcing.enable:
                PK_tot = PK1 + PK2
                to_print = (
                    "PK1  = {:11.5e} ; PK2       = {:11.5e} ; PK_tot   = {:11.5e} \n"
                    "PZ1  = {:11.5e} ; PZ2       = {:11.5e} ; PZ_tot   = {:11.5e} \n"
                ).format(PK1, PK2, PK1 + PK2, PZ1, PZ2, PZ1 + PZ2)
                self.file.write(to_print)

            self.file.flush()
            os.fsync(self.file.fileno())

        if self.has_to_plot and mpi.rank == 0:

            self.ax_a.plot(tsim, energy, "k.")

            self.axe_b.plot(tsim, epsK_tot, "k.")
            if self.sim.params.forcing.enable:
                self.axe_b.plot(tsim, PK_tot, "m.")

            if tsim - self.t_last_show >= self.period_show:
                self.t_last_show = tsim
                fig = self.ax_a.get_figure()
                fig.canvas.draw()

    def load(self):
        dict_results = {"name_solver": self.output.name_solver}

        with open(self.path_file) as file_means:
            lines = file_means.readlines()

        lines_t = []
        lines_E = []
        lines_PK = []
        lines_PZ = []
        lines_epsK = []
        lines_epsZ = []

        for il, line in enumerate(lines):
            if line.startswith("time ="):
                lines_t.append(line)
            if line.startswith("E    ="):
                lines_E.append(line)
            if line.startswith("PK1  ="):
                lines_PK.append(line)
            if line.startswith("PZ1  ="):
                lines_PZ.append(line)
            if line.startswith("epsK ="):
                lines_epsK.append(line)
            if line.startswith("epsZ ="):
                lines_epsZ.append(line)

        nt = len(lines_t)

        t = np.empty(nt)
        E = np.empty(nt)
        Z = np.empty(nt)
        if self.sim.params.forcing.enable:
            PK1 = np.empty(nt)
            PK2 = np.empty(nt)
            PK_tot = np.empty(nt)
            PZ1 = np.empty(nt)
            PZ2 = np.empty(nt)
            PZ_tot = np.empty(nt)
        epsK = np.empty(nt)
        epsK_hypo = np.empty(nt)
        epsK_tot = np.empty(nt)
        epsZ = np.empty(nt)
        epsZ_hypo = np.empty(nt)
        epsZ_tot = np.empty(nt)

        for il in range(nt):
            line = lines_t[il]
            words = line.split()
            t[il] = float(words[2])

            line = lines_E[il]
            words = line.split()
            E[il] = float(words[2])
            Z[il] = float(words[6])

            if self.sim.params.forcing.enable:
                line = lines_PK[il]
                words = line.split()
                PK1[il] = float(words[2])
                PK2[il] = float(words[6])
                PK_tot[il] = float(words[10])

                line = lines_PZ[il]
                words = line.split()
                PZ1[il] = float(words[2])
                PZ2[il] = float(words[6])
                PZ_tot[il] = float(words[10])

            line = lines_epsK[il]
            words = line.split()
            epsK[il] = float(words[2])
            epsK_hypo[il] = float(words[6])
            epsK_tot[il] = float(words[10])

            line = lines_epsZ[il]
            words = line.split()
            epsZ[il] = float(words[2])
            epsZ_hypo[il] = float(words[6])
            epsZ_tot[il] = float(words[10])

        dict_results["t"] = t
        dict_results["E"] = E
        dict_results["Z"] = Z

        if self.sim.params.forcing.enable:
            dict_results["PK1"] = PK1
            dict_results["PK2"] = PK2
            dict_results["PK_tot"] = PK_tot

            dict_results["PZ1"] = PZ1
            dict_results["PZ2"] = PZ2
            dict_results["PZ_tot"] = PZ_tot

        dict_results["epsK"] = epsK
        dict_results["epsK_hypo"] = epsK_hypo
        dict_results["epsK_tot"] = epsK_tot

        dict_results["epsZ"] = epsZ
        dict_results["epsZ_hypo"] = epsZ_hypo
        dict_results["epsZ_tot"] = epsZ_tot
        return dict_results

    def plot_dt_energy(self):
        """
        Checks if dE/dt = energy_injection - energy_dissipation.

        """
        dict_results = self.load()

        t = dict_results["t"]
        E = dict_results["E"]

        epsK_tot = dict_results["epsK_tot"]

        if self.sim.params.forcing.enable:
            PK_tot = dict_results["PK_tot"]
            model = PK_tot - epsK_tot
        else:
            model = -epsK_tot

        dtE = np.gradient(E, t)

        fig, ax = plt.subplots()
        ax.set_xlabel("t")
        ax.plot(t, dtE, label="dE/dt")

        ax.plot(t, model, label=r"$P_E - \epsilon$")

        ax.legend()

        fig.tight_layout()

    def plot_dt_enstrophy(self):
        """
        Checks dZ/dt = enstrophy_injection - enstrophy_dissipation.
        """
        dict_results = self.load()

        t = dict_results["t"]
        Z = dict_results["Z"]

        epsZ_tot = dict_results["epsZ_tot"]

        if self.sim.params.forcing.enable:
            PZ_tot = dict_results["PZ_tot"]
            PZ1 = dict_results["PZ1"]
            PZ2 = dict_results["PZ2"]
            model = PZ_tot - epsZ_tot
        else:
            model = -epsZ_tot

        dtZ = np.gradient(Z, t)

        fig, axes = plt.subplots(2)
        ax = axes[0]
        ax.plot(t, dtZ, label="dZ/dt")
        ax.plot(t, model, label=r"$P_Z-\epsilon_Z$")
        ax.legend()

        ax = axes[1]
        ax.plot(t, epsZ_tot, label=r"$\epsilon_{Z}$")
        ax.plot(t, PZ_tot, label=r"$P_{Z}$")
        ax.plot(t, PZ1, label=r"$P_{Z1}$")
        ax.plot(t, PZ2, label=r"$P_{Z2}$")
        ax.legend()

        fig.tight_layout()

    def plot(self):
        dict_results = self.load()

        t = dict_results["t"]
        E = dict_results["E"]
        Z = dict_results["Z"]

        epsK_hypo = dict_results["epsK_hypo"]
        epsK_tot = dict_results["epsK_tot"]

        epsZ_hypo = dict_results["epsZ_hypo"]
        epsZ_tot = dict_results["epsZ_tot"]

        fig, axes = plt.subplots(2)
        fig.suptitle("Energy and enstrophy")

        ax0 = axes[0]
        ax0.set_ylabel("$E(t)$")
        ax0.plot(t, E, "k.-", linewidth=2, label=r"$E$")
        ax0.legend()

        ax1 = axes[1]
        ax1.set_ylabel("$Z(t)$")
        ax1.set_xlabel("$t$")
        ax1.plot(t, Z, "k", linewidth=2, label=r"$Z$")
        ax1.legend()

        fig, axes = plt.subplots(2)
        fig.suptitle("Dissipation of energy and enstrophy")

        ax0 = axes[0]

        ax0.set_ylabel(r"$\epsilon_K(t)$")

        ax0.plot(t, epsK_tot, "k", linewidth=2, label=r"$\epsilon$")

        ax1 = axes[1]
        ax1.set_xlabel("$t$")
        ax1.set_ylabel(r"$\epsilon_Z(t)$")
        ax1.plot(t, epsZ_tot, "k", linewidth=2, label=r"$\epsilon_Z$")

        if self.sim.params.forcing.enable:
            PK1 = dict_results["PK1"]
            PK2 = dict_results["PK2"]
            PK_tot = dict_results["PK_tot"]
            PZ_tot = dict_results["PZ_tot"]
            ax0.plot(t, PK_tot, "c", linewidth=2, label="$P_K$")
            ax0.plot(t, PK1, "c--", linewidth=1, label="$P_{K1}$")
            ax0.plot(t, PK2, "c:", linewidth=1, label="$P_{K2}$")
            ax1.plot(t, PZ_tot, "c", linewidth=2, label="$P_Z$")
            ax0.set_ylabel("P_E(t), epsK(t)")
            ax1.set_ylabel("P_Z(t), epsZ(t)")

        if self.sim.params.nu_m4 != 0:
            ax0.plot(t, epsK_hypo, "g", linewidth=1, label=r"$\epsilon_{hypo}$")
            ax1.plot(t, epsZ_hypo, "g", linewidth=1, label=r"$\epsilon_{Zhypo}$")

        ax0.legend()
        ax1.legend()
