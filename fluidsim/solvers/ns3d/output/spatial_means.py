"""Spatial means output (:mod:`fluidsim.solvers.ns3d.output.spatial_means`)
===========================================================================

.. autoclass:: SpatialMeansNS3D
   :members:
   :private-members:

"""

import os

import numpy as np
import matplotlib.pyplot as plt

from fluiddyn.util import mpi

from fluidsim.base.output.spatial_means import SpatialMeansBase


class SpatialMeansNS3D(SpatialMeansBase):
    """Spatial means output."""

    def _save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim
        nrj_vx_fft, nrj_vy_fft, nrj_vz_fft = self.output.compute_energies_fft()
        energy_fft = nrj_vx_fft + nrj_vy_fft + nrj_vz_fft
        nrj_vx = self.sum_wavenumbers(nrj_vx_fft)
        nrj_vy = self.sum_wavenumbers(nrj_vy_fft)
        nrj_vz = self.sum_wavenumbers(nrj_vz_fft)
        energy = nrj_vx + nrj_vy + nrj_vz

        f_d, f_d_hypo = self.sim.compute_freq_diss()
        epsK = self.sum_wavenumbers(f_d * 2 * energy_fft)
        epsK_hypo = self.sum_wavenumbers(f_d_hypo * 2 * energy_fft)

        if self.sim.params.nu_4 > 0.0:
            f_d4 = self.params.nu_4 * self.oper.K4
            epsK4 = self.sum_wavenumbers(f_d4 * 2 * energy_fft)
            del f_d4

        if self.sim.params.nu_8 > 0.0:
            f_d8 = self.params.nu_8 * self.oper.K8
            epsK8 = self.sum_wavenumbers(f_d8 * 2 * energy_fft)
            del f_d8

        if self.sim.params.forcing.enable:
            deltat = self.sim.time_stepping.deltat
            forcing_fft = self.sim.forcing.get_forcing()

            fx_fft = forcing_fft.get_var("vx_fft")
            fy_fft = forcing_fft.get_var("vy_fft")
            fz_fft = forcing_fft.get_var("vz_fft")

            vx_fft = self.sim.state.state_spect.get_var("vx_fft")
            vy_fft = self.sim.state.state_spect.get_var("vy_fft")
            vz_fft = self.sim.state.state_spect.get_var("vz_fft")

            PK1_fft = np.ascontiguousarray(
                np.real(
                    vx_fft.conj() * fx_fft
                    + vy_fft.conj() * fy_fft
                    + vz_fft.conj() * fz_fft
                )
            )
            PK2_fft = (
                (abs(fx_fft) ** 2 + abs(fy_fft) ** 2 + abs(fz_fft) ** 2)
                * deltat
                / 2
            )

            PK1 = self.sum_wavenumbers(PK1_fft)
            PK2 = self.sum_wavenumbers(PK2_fft)

        if mpi.rank == 0:

            self.file.write(
                f"####\ntime = {tsim:11.5e}\n"
                f"E    = {energy:11.5e}\n"
                f"Ex   = {nrj_vx:11.5e} ; Ey   = {nrj_vy:11.5e} ; Ez   = {nrj_vz:11.5e}\n"
                f"epsK = {epsK:11.5e} ; epsK_hypo = {epsK_hypo:11.5e} ; "
                f"epsK_tot = {epsK + epsK_hypo:11.5e} \n"
            )

            if self.sim.params.nu_4 > 0.0:
                self.file.write(f"epsK4 = {epsK4:11.5e}\n")

            if self.sim.params.nu_8 > 0.0:
                self.file.write(f"epsK8 = {epsK8:11.5e}\n")

            if self.sim.params.forcing.enable:
                self.file.write(
                    f"PK1  = {PK1:11.5e} ; PK2       = {PK2:11.5e} ; "
                    f"PK_tot   = {PK1 + PK2:11.5e} \n"
                )

            self.file.flush()
            os.fsync(self.file.fileno())

        if self.has_to_plot and mpi.rank == 0:

            self.ax_a.plot(tsim, energy, "k.")

            # self.axe_b.plot(tsim, epsK_tot, 'k.')
            # if self.sim.params.forcing.enable:
            #     self.axe_b.plot(tsim, PK_tot, 'm.')

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
        lines_Ex = []
        lines_PK = []
        lines_epsK = []
        lines_epsK4 = []
        lines_epsK8 = []

        for il, line in enumerate(lines):
            if line.startswith("time ="):
                lines_t.append(line)
            if line.startswith("E    ="):
                lines_E.append(line)
            if line.startswith("Ex   ="):
                lines_Ex.append(line)
            if line.startswith("PK1  ="):
                lines_PK.append(line)
            if line.startswith("epsK ="):
                lines_epsK.append(line)
            elif line.startswith("epsK4 ="):
                lines_epsK4.append(line)
            elif line.startswith("epsK8 ="):
                lines_epsK8.append(line)

        # fmt: off
        nt = self._get_nb_points_from_lines(
            lines_t, lines_E, lines_Ex, lines_PK, lines_epsK,
            lines_epsK4, lines_epsK8
        )
        # fmt: on

        t = np.empty(nt)
        E = np.empty(nt)
        Ex = np.empty(nt)
        Ey = np.empty(nt)
        Ez = np.empty(nt)
        PK1 = np.zeros(nt)
        PK2 = np.zeros(nt)
        PK_tot = np.zeros(nt)
        epsK = np.empty(nt)
        epsK_hypo = np.zeros(nt)
        epsK_tot = np.zeros(nt)

        if lines_epsK4:
            epsK4 = np.empty(nt)

        if lines_epsK8:
            epsK8 = np.empty(nt)

        for il in range(nt):
            line = lines_t[il]
            words = line.split()
            t[il] = float(words[2])

            line = lines_E[il]
            words = line.split()
            E[il] = float(words[2])

            line = lines_Ex[il]
            words = line.split()
            Ex[il] = float(words[2])
            Ey[il] = float(words[6])
            Ez[il] = float(words[10])

            if self.sim.params.forcing.enable:
                line = lines_PK[il]
                words = line.split()
                PK1[il] = float(words[2])
                PK2[il] = float(words[6])
                PK_tot[il] = float(words[10])

            line = lines_epsK[il]
            words = line.split()
            epsK[il] = float(words[2])
            epsK_hypo[il] = float(words[6])
            epsK_tot[il] = float(words[10])

            if lines_epsK4:
                line = lines_epsK4[il]
                words = line.split()
                epsK4[il] = float(words[2])

            if lines_epsK8:
                line = lines_epsK8[il]
                words = line.split()
                epsK8[il] = float(words[2])

        dict_results["t"] = t
        dict_results["E"] = E
        dict_results["Ex"] = Ex
        dict_results["Ey"] = Ey
        dict_results["Ez"] = Ez

        dict_results["PK1"] = PK1
        dict_results["PK2"] = PK2
        dict_results["PK_tot"] = PK_tot

        dict_results["epsK"] = epsK
        dict_results["epsK_hypo"] = epsK_hypo
        dict_results["epsK_tot"] = epsK_tot

        if lines_epsK4:
            dict_results["epsK4"] = epsK4

        if lines_epsK8:
            dict_results["epsK8"] = epsK8

        return dict_results

    def plot(self, plot_injection=True, plot_hyper=False):
        dict_results = self.load()

        t = dict_results["t"]
        E = dict_results["E"]
        Ex = dict_results["Ex"]
        Ey = dict_results["Ey"]
        Ez = dict_results["Ez"]

        epsK = dict_results["epsK"]
        epsK_hypo = dict_results["epsK_hypo"]
        epsK_tot = dict_results["epsK_tot"]

        fig, ax = self.output.figure_axe()
        ax.set_title("Energy and enstrophy\n" + self.output.summary_simul)
        ax.set_ylabel("$E(t)$")
        ax.set_xlabel("$t$")
        ax.plot(t, E, "k", linewidth=2, label="Energy")
        ax.plot(t, Ex, "b", label="$E_x$")
        ax.plot(t, Ey, "r", label="$E_y$")
        ax.plot(t, Ez, "c", label="$E_z$")

        ax.legend()

        fig, ax = self.output.figure_axe()
        ax.set_title(
            "Dissipation of energy and enstrophy\n" + self.output.summary_simul
        )
        ax.set_ylabel(r"$\epsilon_K(t)$")
        ax.set_xlabel("$t$")

        def _plot(x, y, fmt, label=None, linewidth=1, zorder=10):
            ax.plot(
                x,
                y,
                fmt,
                label=label,
                linewidth=linewidth,
                zorder=zorder,
            )

        _plot(t, epsK, "r", r"$\epsilon$", linewidth=2)
        if self.sim.params.nu_m4 != 0:
            _plot(t, epsK_hypo, "g", r"$\epsilon_{-4}$", linewidth=2)
            _plot(t, epsK_tot, "k", r"$\epsilon_{tot}$", linewidth=2)

        if "epsK4" in dict_results and plot_hyper:
            epsK4 = dict_results["epsK4"]
            if not np.allclose(epsK, epsK4):
                _plot(
                    t,
                    epsK4,
                    "r:",
                    r"$\epsilon_4$",
                )

        if "epsK8" in dict_results and plot_hyper:
            epsK8 = dict_results["epsK8"]
            if not np.allclose(epsK, epsK8):
                _plot(
                    t,
                    epsK8,
                    "r:",
                    r"$\epsilon_8$",
                )

        if "PK_tot" in dict_results and plot_injection:
            PK_tot = dict_results["PK_tot"]
            ax.plot(t, PK_tot, "r--", label=r"$P$", zorder=0)

        ax.legend()

    def plot_dt_E(self):
        dict_results = self.load()

        times = dict_results["t"]
        E_tot = dict_results["E"]

        dt_E_tot = np.gradient(E_tot, times)
        try:
            eps_tot = dict_results["eps_tot"]
        except KeyError:
            eps_tot = dict_results["epsK_tot"]

        all_terms = -eps_tot.copy()

        if "PK_tot" in dict_results:
            P_tot = dict_results["PK_tot"]
            if "PA_tot" in dict_results:
                P_tot += dict_results["PA_tot"]
            all_terms += P_tot

        fig, ax = self.output.figure_axe()

        ax.plot(times, dt_E_tot, "--k", label="$d_t E$")
        if "PK_tot" in dict_results:
            ax.plot(times, P_tot, "b", label="forcing")
        ax.plot(times, -eps_tot, color="orange", label="viscosity")
        ax.plot(times, all_terms, "r", label="All terms")

        ax.set_title(self.output.summary_simul)
        ax.set_xlabel("time")

        fig.legend()
        fig.tight_layout()

    def get_dimless_numbers_versus_time(self, data=None):
        """Compute dimensionless numbers"""
        if data is None:
            data = self.load()
        results = {"t": data["t"]}

        try:
            EKh = Uh2 = data["Ex"] + data["Ey"]
            EKz = data["Ez"]
        except KeyError:
            EKhr = data["EKhr"]
            EKhd = data["EKhd"]
            EKhs = data["EKhs"]
            EKh = Uh2 = EKhr + EKhd + EKhs
            EKz = data["EKz"]

        epsK = data["epsK"]
        epsK_hyper = np.zeros_like(epsK)

        nu_2 = self.params.nu_2
        nu_4 = self.params.nu_4
        nu_8 = self.params.nu_8

        if nu_2:
            eta = (nu_2**3 / epsK) ** (1 / 4)
            delta_kz = 2 * np.pi / self.params.oper.Lz
            coef_dealiasing = self.params.oper.coef_dealiasing
            k_max = coef_dealiasing * delta_kz * self.params.oper.nz / 2
            results["k_max*eta"] = k_max * eta
        if nu_4:
            epsK_hyper += data["epsK4"]
        if nu_8:
            epsK_hyper += data["epsK8"]

        if nu_2:
            if nu_4 or nu_8:
                epsK2 = epsK - epsK_hyper
                results["epsK2/epsK"] = epsK2 / epsK
            else:
                results["epsK2/epsK"] = np.ones_like(epsK)

        results["dimensional"] = {
            "Uh2": Uh2,
            "epsK": epsK,
            "EKh": EKh,
            "EKz": EKz,
        }

        if nu_2:
            results["dimensional"]["eta"] = eta

        return results

    def get_dimless_numbers_averaged(self, tmin=0, tmax=None):
        """Compute averaged dimensionless numbers"""
        numbers_vs_time = self.get_dimless_numbers_versus_time()
        times = numbers_vs_time["t"]
        itmin, itmax = _compute_indices_tmin_tmax(times, tmin, tmax)
        stop = itmax + 1

        result = {
            k: q[itmin:stop].mean()
            for k, q in numbers_vs_time.items()
            if k not in ["t", "dimensional"]
        }
        result["dimensional"] = {
            k: q[itmin:stop].mean()
            for k, q in numbers_vs_time["dimensional"].items()
        }

        delta_kz = 2 * np.pi / self.params.oper.Lz
        coef_dealiasing = self.params.oper.coef_dealiasing
        result["dimensional"]["k_max"] = (
            coef_dealiasing * delta_kz * self.params.oper.nz / 2
        )

        return result

    def plot_dimless_numbers_versus_time(self, tmin=0, tmax=None):
        """Plot dimensionless numbers"""
        numbers_vs_time = self.get_dimless_numbers_versus_time()
        times = numbers_vs_time["t"]
        itmin, itmax = _compute_indices_tmin_tmax(times, tmin, tmax)
        stop = itmax + 1
        times = times[itmin:stop]

        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

        keys_ax1 = ["k_max*eta", "epsK2/epsK", "Gamma"]

        for key, quantity in numbers_vs_time.items():
            if key in ["t", "dimensional"]:
                continue
            quantity = quantity[itmin:stop]

            if key in keys_ax1:
                ax = ax1
            else:
                ax = ax0

            ax.plot(times, quantity, label=key)
            print(f"<{key}> = {np.mean(quantity):.3g}")

        for ax in (ax0, ax1):
            ax.set_yscale("log")
            ax.legend()

        ax0.set_xlabel("$t$")
        fig.suptitle(f"dimensionless numbers\n{self.output.summary_simul}")


def _compute_indices_tmin_tmax(times, tmin, tmax):
    if tmax is None:
        itmax = len(times) - 1
    else:
        itmax = abs(times - tmax).argmin()

    itmin = abs(times - tmin).argmin()
    return itmin, itmax
