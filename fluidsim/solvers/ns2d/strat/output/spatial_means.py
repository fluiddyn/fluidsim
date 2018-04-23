"""Spatial means (:mod:`fluidsim.solvers.ns2d.strat.output.spatial_means`)
==========================================================================

.. autoclass:: SpatialMeansNS2DStrat
   :members:
   :private-members:

"""

from __future__ import division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt

from fluiddyn.util import mpi

from fluidsim.base.output.spatial_means import SpatialMeansBase


class SpatialMeansNS2DStrat(SpatialMeansBase):
    """Spatial means output stratified fluid"""

    def _save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim

        # Compute the kinetic, potential energy and enstrophy
        energyK_fft, energyA_fft = self.output.compute_energies_fft()
        energy_fft = self.output.compute_energy_fft()
        energy = self.sum_wavenumbers(energy_fft)
        energyK = self.sum_wavenumbers(energyK_fft)
        energyA = self.sum_wavenumbers(energyA_fft)

        enstrophy_fft = self.output.compute_enstrophy_fft()
        enstrophy = self.sum_wavenumbers(enstrophy_fft)

        # Dissipation rate kinetic and potential energy (kappa = viscosity)
        f_d, f_d_hypo = self.sim.compute_freq_diss()

        epsK = self.sum_wavenumbers(f_d * 2. * energyK_fft)
        epsK_hypo = self.sum_wavenumbers(f_d_hypo * 2. * energyK_fft)

        epsA = self.sum_wavenumbers(f_d * 2. * energyA_fft)
        epsA_hypo = self.sum_wavenumbers(f_d_hypo * 2. * energyA_fft)

        epsZ = self.sum_wavenumbers(f_d * 2. * enstrophy_fft)
        epsZ_hypo = self.sum_wavenumbers(f_d_hypo * 2. * enstrophy_fft)

        # Injection energy if forcing is True
        if self.sim.params.forcing.enable:
            deltat = self.sim.time_stepping.deltat
            Frot_fft = self.sim.forcing.get_forcing().get_var("rot_fft")
            Fb_fft = self.sim.forcing.get_forcing().get_var("b_fft")
            Fx_fft, Fy_fft = self.vecfft_from_rotfft(Frot_fft)

            rot_fft = self.sim.state.state_spect.get_var("rot_fft")
            b_fft = self.sim.state.state_spect.get_var("b_fft")

            ux_fft, uy_fft = self.vecfft_from_rotfft(rot_fft)

            PZ1_fft = np.real(rot_fft.conj() * Frot_fft)
            PZ2_fft = (abs(Frot_fft) ** 2)

            PZ1 = self.sum_wavenumbers(PZ1_fft)
            PZ2 = deltat / 2 * self.sum_wavenumbers(PZ2_fft)

            PK1_fft = np.real(ux_fft.conj() * Fx_fft + uy_fft.conj() * Fy_fft)
            PK2_fft = (abs(Fx_fft) ** 2 + abs(Fy_fft) ** 2)

            PK1 = self.sum_wavenumbers(PK1_fft)
            PK2 = deltat / 2 * self.sum_wavenumbers(PK2_fft)

            PA1_fft = np.real(b_fft.conj() * Fb_fft)
            PA2_fft = (abs(Fb_fft) ** 2)

            PA1 = self.sum_wavenumbers(PA1_fft) / self.params.N ** 2
            PA2 = deltat / 2 / self.params.N ** 2 * self.sum_wavenumbers(PA2_fft)

        if mpi.rank == 0:
            epsK_tot = epsK + epsK_hypo

            self.file.write("####\ntime = {:11.5e}\n".format(tsim))
            to_print = (
                "Z    = {:11.5e} \n"
                "E    = {:11.5e} ; EK         = {:11.5e} ; EA        = {:11.5e} \n"
                "epsA = {:11.5e} ; epsA_hypo  = {:11.5e} ; epsA_tot  = {:11.5e} \n"
                "epsK = {:11.5e} ; epsK_hypo  = {:11.5e} ; epsK_tot  = {:11.5e} \n"
                "epsZ = {:11.5e} ; epsZ_hypo = {:11.5e} ; epsZ_tot = {:11.5e} \n"
            ).format(
                enstrophy,
                energy,
                energyK,
                energyA,
                epsA,
                epsA_hypo,
                epsA + epsA_hypo,
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
                    "PA1  = {:11.5e} ; PA2       = {:11.5e} ; PA_tot   = {:11.5e} \n"
                    "PZ1  = {:11.5e} ; PZ2       = {:11.5e} ; PZ_tot   = {:11.5e} \n"
                ).format(
                    PK1, PK2, PK1 + PK2, PA1, PA2, PA1 + PA2, PZ1, PZ2, PZ1 + PZ2
                )
                self.file.write(to_print)

            self.file.flush()
            os.fsync(self.file.fileno())

        if self.has_to_plot and mpi.rank == 0:

            self.axe_a.plot(tsim, energy, "k.")

            self.axe_b.plot(tsim, epsK_tot, "k.")
            if self.sim.params.forcing.enable:
                self.axe_b.plot(tsim, PK_tot, "m.")

            if tsim - self.t_last_show >= self.period_show:
                self.t_last_show = tsim
                fig = self.axe_a.get_figure()
                fig.canvas.draw()

    def load(self):
        """Generates a dictionary with the output values"""
        dict_results = {"name_solver": self.output.name_solver}

        with open(self.path_file) as file_means:
            lines = file_means.readlines()

        lines_t = []
        lines_Z = []
        lines_E = []
        if self.sim.params.forcing.enable:
            lines_PK = []
            lines_PA = []
            lines_PZ = []
        lines_epsK = []
        lines_epsZ = []
        lines_epsA = []

        for il, line in enumerate(lines):
            if line.startswith("time ="):
                lines_t.append(line)
            if line.startswith("Z    ="):
                lines_Z.append(line)
            if line.startswith("E    ="):
                lines_E.append(line)

            if self.sim.params.forcing.enable:
                if line.startswith("PK1  ="):
                    lines_PK.append(line)
                if line.startswith("PA1  ="):
                    lines_PA.append(line)
                if line.startswith("PZ1  ="):
                    lines_PZ.append(line)
            if line.startswith("epsK ="):
                lines_epsK.append(line)
            if line.startswith("epsZ ="):
                lines_epsZ.append(line)
            if line.startswith("epsA ="):
                lines_epsA.append(line)

        nt = len(lines_t)
        if nt > 1:
            nt -= 1

        t = np.empty(nt)
        Z = np.empty(nt)
        E = np.empty(nt)
        EK = np.empty(nt)
        EA = np.empty(nt)

        if self.sim.params.forcing.enable:
            PK1 = np.empty(nt)
            PK2 = np.empty(nt)
            PK_tot = np.empty(nt)
            PA1 = np.empty(nt)
            PA2 = np.empty(nt)
            PA_tot = np.empty(nt)
            PZ1 = np.empty(nt)
            PZ2 = np.empty(nt)
            PZ_tot = np.empty(nt)

        epsK = np.empty(nt)
        epsK_hypo = np.empty(nt)
        epsK_tot = np.empty(nt)
        epsZ = np.empty(nt)
        epsZ_hypo = np.empty(nt)
        epsZ_tot = np.empty(nt)
        epsA = np.empty(nt)
        epsA_hypo = np.empty(nt)
        epsA_tot = np.empty(nt)

        for il in range(nt):
            line = lines_t[il]
            words = line.split()
            t[il] = float(words[2])

            line = lines_Z[il]
            words = line.split()
            Z[il] = float(words[2])

            line = lines_E[il]
            words = line.split()
            E[il] = float(words[2])
            EK[il] = float(words[6])
            EA[il] = float(words[10])

            if self.sim.params.forcing.enable:
                line = lines_PK[il]
                words = line.split()
                PK1[il] = float(words[2])
                PK2[il] = float(words[6])
                PK_tot[il] = float(words[10])

                line = lines_PA[il]
                words = line.split()
                PA1[il] = float(words[2])
                PA2[il] = float(words[6])
                PA_tot[il] = float(words[10])

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

            line = lines_epsA[il]
            words = line.split()
            epsA[il] = float(words[2])
            epsA_hypo[il] = float(words[6])
            epsA_tot[il] = float(words[10])

        dict_results["t"] = t
        dict_results["Z"] = Z

        dict_results["E"] = E
        dict_results["EK"] = EK
        dict_results["EA"] = EA

        if self.sim.params.forcing.enable:
            dict_results["PK1"] = PK1
            dict_results["PK2"] = PK2
            dict_results["PK_tot"] = PK_tot

            dict_results["PA1"] = PA1
            dict_results["PA2"] = PA2
            dict_results["PA_tot"] = PA_tot

            dict_results["PZ1"] = PZ1
            dict_results["PZ2"] = PZ2
            dict_results["PZ_tot"] = PZ_tot

        dict_results["epsK"] = epsK
        dict_results["epsK_hypo"] = epsK_hypo
        dict_results["epsK_tot"] = epsK_tot

        dict_results["epsZ"] = epsZ
        dict_results["epsZ_hypo"] = epsZ_hypo
        dict_results["epsZ_tot"] = epsZ_tot

        dict_results["epsA"] = epsA
        dict_results["epsA_hypo"] = epsA_hypo
        dict_results["epsA_tot"] = epsA_tot

        return dict_results

    def plot_dt_energy(self):
        """
        Checks if dE/dt = energy_injection - energy_dissipation.

        """
        if mpi.rank != 0:
            return

        dict_results = self.load()

        times = dict_results["t"]
        E = dict_results["E"]

        epsK_tot = dict_results["epsK_tot"]
        epsA_tot = dict_results["epsA_tot"]
        eps_tot = epsK_tot + epsA_tot

        dtE = np.gradient(E, times)
        dtE = np.diff(E) / np.diff(times)
        times_dtE = (times[1:] + times[:-1]) / 2

        if self.sim.params.forcing.enable:
            PK = dict_results["PK_tot"]
            PA = dict_results["PA_tot"]
            P = PK + PA
            model = P - eps_tot
        else:
            model = -eps_tot

        fig, axes = plt.subplots(2)
        ax = axes[0]
        ax.plot(times_dtE[2:], dtE[2:], label="$dE/dt$")

        times[:-1] += np.diff(times) / 2
        ax.plot(times[1:], model[1:], label=r"$P - \epsilon$")
        ax.legend()

        ax = axes[1]
        if self.sim.params.forcing.enable:
            ax.plot(times[1:], PA[1:], "b", label="PA")
            ax.plot(times[1:], PK[1:], "r", label="PK")
            ax.plot(times[1:], eps_tot[1:], "k:", label=r"$\epsilon$")
        ax.set_xlabel("t")
        ax.legend()

        fig.tight_layout()

    def plot_energy(self):
        """Plots the energy."""
        if mpi.rank != 0:
            return

        dict_results = self.load()

        t = dict_results["t"]
        E = dict_results["E"]
        EK = dict_results["EK"]
        EA = dict_results["EA"]

        fig, ax = plt.subplots()
        ax.plot(t, E, label="E")
        ax.plot(t, EK, label="EK")
        ax.plot(t, EA, label="EA")

        ax.legend()

        fig.tight_layout()

    def plot(self):
        if mpi.rank != 0:
            return

        dict_results = self.load()

        t = dict_results["t"]
        E = dict_results["E"]
        Z = dict_results["Z"]

        epsK = dict_results["epsK"]
        epsK_hypo = dict_results["epsK_hypo"]
        epsK_tot = dict_results["epsK_tot"]

        epsZ = dict_results["epsZ"]
        epsZ_hypo = dict_results["epsZ_hypo"]
        epsZ_tot = dict_results["epsZ_tot"]

        epsA = dict_results["epsA"]
        epsA_hypo = dict_results["epsA_hypo"]
        epsA_tot = dict_results["epsA_tot"]

        width_axe = 0.85
        height_axe = 0.39
        x_left_axe = 0.12
        z_bottom_axe = 0.55

        size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        fig.suptitle("Energy and enstrophy")
        ax1.set_ylabel("$E(t)$")
        ax1.plot(t, E, "k", linewidth=2)

        z_bottom_axe = 0.08
        size_axe[1] = z_bottom_axe
        ax2 = fig.add_axes(size_axe)
        ax2.set_ylabel("$Z(t)$")
        ax2.set_xlabel("$t$")
        ax2.plot(t, Z, "k", linewidth=2)

        z_bottom_axe = 0.54
        size_axe[1] = z_bottom_axe
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        fig.suptitle("Dissipation of energy and enstrophy")
        ax1.set_ylabel(r"$\epsilon (t) = \epsilon_K + \epsilon_A$")

        ax1.plot(t, epsK + epsA, "r", label=r"$\epsilon$", linewidth=2)
        ax1.plot(
            t, epsK_hypo + epsA_hypo, "g", label=r"$\epsilon_{hypo}$", linewidth=2
        )
        ax1.plot(
            t, epsK_tot + epsA_tot, "k", label=r"$\epsilon_{tot}$", linewidth=2
        )

        z_bottom_axe = 0.08
        size_axe[1] = z_bottom_axe
        ax2 = fig.add_axes(size_axe)
        ax2.set_xlabel("$t$")
        ax2.set_ylabel(r"$\epsilon_Z(t)$")

        ax2.plot(t, epsZ, "r", linewidth=2)
        ax2.plot(t, epsZ_hypo, "g", linewidth=2)
        ax2.plot(t, epsZ_tot, "k", linewidth=2)

        if self.sim.params.forcing.enable:
            PK = dict_results["PK_tot"]
            PA = dict_results["PA_tot"]
            P = PK + PA
            PZ_tot = dict_results["PZ_tot"]
            ax1.plot(t, P, "c", label="P", linewidth=2)
            ax2.plot(t, PZ_tot, "c", label="P", linewidth=2)
            ax1.set_ylabel("P_E(t), epsK(t)")
            ax2.set_ylabel("P_Z(t), epsZ(t)")

        ax1.legend()
