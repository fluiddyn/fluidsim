"""Spatial means (:mod:`fluidsim.solvers.ns3d.strat.output.spatial_means`)
==========================================================================

.. autoclass:: SpatialMeansNS3DStrat
   :members:
   :private-members:

"""

import os
import numpy as np


from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.output.spatial_means import SpatialMeansNS3D


class SpatialMeansNS3DStrat(SpatialMeansNS3D):
    """Spatial means output."""

    def __init__(self, output):
        self.one_over_N2 = 1.0 / output.sim.params.N**2
        super().__init__(output)

    def _save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim
        nrj_A, nrj_Kz, nrj_Khr, nrj_Khd = self.output.compute_energies_fft()
        energyK_fft = nrj_Kz + nrj_Khr + nrj_Khd
        # shear modes
        COND_SHEAR = self.oper.Kx**2 + self.oper.Ky**2 == 0.0
        nrj_Khs = nrj_Khr * COND_SHEAR
        nrj_As = nrj_A * COND_SHEAR
        energyA_fft = nrj_A
        nrj_A = self.sum_wavenumbers(nrj_A)
        nrj_As = self.sum_wavenumbers(nrj_As)
        nrj_Kz = self.sum_wavenumbers(nrj_Kz)
        nrj_Khs = self.sum_wavenumbers(nrj_Khs)
        nrj_Khr = self.sum_wavenumbers(nrj_Khr)
        nrj_Khr = nrj_Khr - nrj_Khs
        nrj_Khd = self.sum_wavenumbers(nrj_Khd)
        energy = nrj_A + nrj_Kz + nrj_Khr + nrj_Khd + nrj_Khs

        f_d, f_d_hypo = self.sim.compute_freq_diss()
        epsK = self.sum_wavenumbers(f_d * 2 * energyK_fft)
        epsK_hypo = self.sum_wavenumbers(f_d_hypo * 2 * energyK_fft)
        epsA = self.sum_wavenumbers(f_d * 2 * energyA_fft)
        epsA_hypo = self.sum_wavenumbers(f_d_hypo * 2 * energyA_fft)

        if self.sim.params.nu_4 > 0.0:
            f_d4 = self.params.nu_4 * self.oper.K4
            epsK4 = self.sum_wavenumbers(f_d4 * 2 * energyK_fft)
            epsA4 = self.sum_wavenumbers(f_d4 * 2 * energyA_fft)
            del f_d4

        if self.sim.params.nu_8 > 0.0:
            f_d8 = self.params.nu_8 * self.oper.K8
            epsK8 = self.sum_wavenumbers(f_d8 * 2 * energyK_fft)
            epsA8 = self.sum_wavenumbers(f_d8 * 2 * energyA_fft)
            del f_d8

        if self.sim.params.forcing.enable:
            deltat = self.sim.time_stepping.deltat
            forcing_fft = self.sim.forcing.get_forcing()

            fx_fft = forcing_fft.get_var("vx_fft")
            fy_fft = forcing_fft.get_var("vy_fft")
            fz_fft = forcing_fft.get_var("vz_fft")
            fb_fft = forcing_fft.get_var("b_fft")

            get_var = self.sim.state.state_spect.get_var
            vx_fft = get_var("vx_fft")
            vy_fft = get_var("vy_fft")
            vz_fft = get_var("vz_fft")
            b_fft = get_var("b_fft")

            PK1_fft = np.real(
                vx_fft.conj() * fx_fft
                + vy_fft.conj() * fy_fft
                + vz_fft.conj() * fz_fft
            )
            PK2_fft = abs(fx_fft) ** 2 + abs(fy_fft) ** 2 + abs(fz_fft) ** 2

            PK1 = self.sum_wavenumbers(np.ascontiguousarray(PK1_fft))
            PK2 = self.sum_wavenumbers(PK2_fft) * deltat / 2

            PA1_fft = np.real(b_fft.conj() * fb_fft)
            PA2_fft = abs(fb_fft) ** 2

            PA1 = self.sum_wavenumbers(np.ascontiguousarray(PA1_fft))
            PA2 = self.sum_wavenumbers(PA2_fft) * deltat / 2

            PA1 *= self.one_over_N2
            PA2 *= self.one_over_N2

        if mpi.rank == 0:

            self.file.write(
                f"####\ntime = {tsim:11.5e}\n"
                f"E    = {energy:11.5e}\n"
                f"EA   = {nrj_A:11.5e} ; EKz   = {nrj_Kz:11.5e} ; "
                f"EKhr   = {nrj_Khr:11.5e} ; EKhd   = {nrj_Khd:11.5e} ; "
                f"EKhs   = {nrj_Khs:11.5e} ; EAs    = {nrj_As:11.5e}\n"
                f"epsK = {epsK:11.5e} ; epsK_hypo = {epsK_hypo:11.5e} ; "
                f"epsA = {epsA:11.5e} ; epsA_hypo = {epsA_hypo:11.5e} ; "
                f"eps_tot = {epsK + epsK_hypo + epsA + epsA_hypo:11.5e} \n"
            )

            if self.sim.params.nu_4 > 0.0:
                self.file.write(
                    f"epsK4 = {epsK4:11.5e} ; epsA4 = {epsA4:11.5e}\n"
                )

            if self.sim.params.nu_8 > 0.0:
                self.file.write(
                    f"epsK8 = {epsK8:11.5e} ; epsA8 = {epsA8:11.5e}\n"
                )

            if self.sim.params.forcing.enable:
                self.file.write(
                    f"PK1  = {PK1:11.5e} ; PK2       = {PK2:11.5e} ; "
                    f"PK_tot   = {PK1 + PK2:11.5e} \n"
                    f"PA1  = {PA1:11.5e} ; PA2       = {PA2:11.5e} ; "
                    f"PA_tot   = {PA1 + PA2:11.5e} \n"
                )

            self.file.flush()
            os.fsync(self.file.fileno())

        if self.has_to_plot and mpi.rank == 0:

            self.axe_a.plot(tsim, energy, "k.")

            # self.axe_b.plot(tsim, epsK_tot, 'k.')
            # if self.sim.params.forcing.enable:
            #     self.axe_b.plot(tsim, PK_tot, 'm.')

            if tsim - self.t_last_show >= self.period_show:
                self.t_last_show = tsim
                fig = self.axe_a.get_figure()
                fig.canvas.draw()

    def load(self):
        results = {"name_solver": self.output.name_solver}

        with open(self.path_file) as file_means:
            lines = file_means.readlines()

        lines_t = []
        lines_E = []
        lines_EA = []
        lines_PK = []
        lines_PA = []
        lines_epsK = []
        lines_epsK4 = []
        lines_epsK8 = []

        for il, line in enumerate(lines):
            if line.startswith("time ="):
                lines_t.append(line)
            elif line.startswith("E    ="):
                lines_E.append(line)
            elif line.startswith("EA   ="):
                lines_EA.append(line)
            elif line.startswith("PK1  ="):
                lines_PK.append(line)
            elif line.startswith("PA1  ="):
                lines_PA.append(line)
            elif line.startswith("epsK ="):
                lines_epsK.append(line)
            elif line.startswith("epsK4 ="):
                lines_epsK4.append(line)
            elif line.startswith("epsK8 ="):
                lines_epsK8.append(line)

        nt = len(lines_t)
        if nt > 1:
            nt -= 1

        # support files saved without EAs
        words = lines_EA[0].split()
        try:
            words[22]
        except IndexError:
            EAs_saved = False
        else:
            EAs_saved = True

        t = np.empty(nt)
        E = np.empty(nt)
        EA = np.empty(nt)
        EKz = np.empty(nt)
        EKhr = np.empty(nt)
        EKhd = np.empty(nt)
        EKhs = np.empty(nt)
        if EAs_saved:
            EAs = np.empty(nt)
        PK1 = np.zeros(nt)
        PK2 = np.zeros(nt)
        PK_tot = np.zeros(nt)
        PA1 = np.zeros(nt)
        PA2 = np.zeros(nt)
        PA_tot = np.zeros(nt)
        epsK = np.empty(nt)
        epsK_hypo = np.empty(nt)
        epsA = np.empty(nt)
        epsA_hypo = np.empty(nt)
        eps_tot = np.empty(nt)

        if lines_epsK4:
            epsK4 = np.empty(nt)
            epsA4 = np.empty(nt)

        if lines_epsK8:
            epsK8 = np.empty(nt)
            epsA8 = np.empty(nt)

        for il in range(nt):
            line = lines_t[il]
            words = line.split()
            t[il] = float(words[2])

            line = lines_E[il]
            words = line.split()
            E[il] = float(words[2])

            line = lines_EA[il]
            words = line.split()
            EA[il] = float(words[2])
            EKz[il] = float(words[6])
            EKhr[il] = float(words[10])
            EKhd[il] = float(words[14])
            EKhs[il] = float(words[18])
            if EAs_saved:
                EAs[il] = float(words[22])

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

            line = lines_epsK[il]
            words = line.split()
            epsK[il] = float(words[2])
            epsK_hypo[il] = float(words[6])
            epsA[il] = float(words[10])
            epsA_hypo[il] = float(words[14])
            eps_tot[il] = float(words[18])

            if lines_epsK4:
                line = lines_epsK4[il]
                words = line.split()
                epsK4[il] = float(words[2])
                epsA4[il] = float(words[6])

            if lines_epsK8:
                line = lines_epsK8[il]
                words = line.split()
                epsK8[il] = float(words[2])
                epsA8[il] = float(words[6])

        results["t"] = t
        results["E"] = E
        results["EA"] = EA
        results["EKz"] = EKz
        results["EKhr"] = EKhr
        results["EKhd"] = EKhd
        results["EKhs"] = EKhs
        if EAs_saved:
            results["EAs"] = EAs

        results["PK1"] = PK1
        results["PK2"] = PK2
        results["PK_tot"] = PK_tot

        results["PA1"] = PA1
        results["PA2"] = PA2
        results["PA_tot"] = PA_tot

        results["epsK"] = epsK
        results["epsK_hypo"] = epsK_hypo
        results["epsA"] = epsA
        results["epsA_hypo"] = epsA_hypo
        results["eps_tot"] = eps_tot

        if lines_epsK4:
            results["epsK4"] = epsK4
            results["epsA4"] = epsA4

        if lines_epsK8:
            results["epsK8"] = epsK8
            results["epsA8"] = epsA8

        return results

    def plot(self, plot_injection=True, plot_hyper=False):
        results = self.load()

        t = results["t"]
        E = results["E"]
        EA = results["EA"]
        EKz = results["EKz"]
        EKhr = results["EKhr"]
        EKhd = results["EKhd"]
        EKhs = results["EKhs"]
        EK = EKz + EKhr + EKhd + EKhs

        epsK = results["epsK"]
        epsK_hypo = results["epsK_hypo"]
        epsA = results["epsA"]
        epsA_hypo = results["epsA_hypo"]
        eps_tot = results["eps_tot"]

        # fig 1 : energies
        fig, ax = self.output.figure_axe()
        ax.set_title("Energy\n" + self.output.summary_simul)
        ax.set_ylabel("$E(t)$")
        ax.plot(t, E, "k", linewidth=2, label="$E$")
        ax.plot(t, EA, "b", label="$E_A$")
        ax.plot(t, EK, "r", label="$E_K$")
        ax.plot(t, EKhr, "r:", label="$E_{Khr}$")
        ax.plot(t, EKhs, "m:", label="$E_{Khs}$")

        try:
            EAs = results["EAs"]
        except KeyError:
            pass
        else:
            ax.plot(t, EAs, "g:", label="$E_{As}$")

        ax.legend()

        # figure 2 : dissipations
        fig, ax = self.output.figure_axe()
        ax.set_title("Dissipation of energy\n" + self.output.summary_simul)
        ax.set_ylabel(r"$\epsilon_K(t)$")

        def _plot(x, y, fmt, label=None, linewidth=1, zorder=10):
            ax.plot(x, y, fmt, label=label, linewidth=linewidth, zorder=zorder)

        _plot(t, epsK, "r", r"$\epsilon_K$")
        _plot(t, epsA, "b", r"$\epsilon_A$")
        _plot(t, eps_tot, "k", r"$\epsilon$", linewidth=2)

        eps_hypo = epsK_hypo + epsA_hypo
        if max(eps_hypo) > 0:
            _plot(t, eps_hypo, "g", r"$\epsilon_{hypo}$")

        if "epsK4" in results and plot_hyper:
            epsK4 = results["epsK4"]
            epsA4 = results["epsA4"]
            if not np.allclose(epsK, epsK4):
                _plot(t, epsK4, "r:", r"$\epsilon_{K4}$")
                _plot(t, epsA4, "b:", r"$\epsilon_{A4}$")

        if "epsK8" in results and plot_hyper:
            epsK8 = results["epsK8"]
            epsA8 = results["epsA8"]
            if not np.allclose(epsK, epsK8):
                _plot(t, epsK8, "r:", r"$\epsilon_{K8}$")
                _plot(t, epsA8, "b:", r"$\epsilon_{A8}$")

        if "PK_tot" in results and plot_injection:
            PK_tot = results["PK_tot"]
            PA_tot = results["PA_tot"]
            ax.plot(t, PK_tot, "r--", label=r"$P_K$", zorder=0)
            ax.plot(t, PA_tot, "b--", label=r"$P_A$", zorder=0)

        ax.legend()

    def get_dimless_numbers_versus_time(self):
        data = self.load()
        results = {"t": data["t"]}
        EKhr = data["EKhr"]
        EKhd = data["EKhd"]
        EKhs = data["EKhs"]
        epsK = data["epsK"]
        epsA = data["epsA"]

        epsK_hyper = np.zeros_like(epsK)

        Uh2 = EKhr + EKhd + EKhs

        N = self.params.N
        nu_2 = self.params.nu_2
        nu_4 = self.params.nu_4
        nu_8 = self.params.nu_8

        results["Fh"] = epsK / (Uh2 * N)

        if nu_2:
            results["R2"] = epsK / (nu_2 * N**2)
        if nu_4:
            results["R4"] = epsK * Uh2 / (nu_4 * N**4)
            epsK_hyper += data["epsK4"]
        if nu_8:
            results["R8"] = epsK * Uh2**3 / (nu_8 * N**8)
            epsK_hyper += data["epsK8"]

        if nu_2 and (nu_4 or nu_8):
            epsK2 = epsK - epsK_hyper
            results["epsK2/epsK"] = epsK2 / epsK

        results["Gamma"] = epsA / epsK
        results["dimensional"] = {"Uh2": Uh2, "epsK": epsK}
        return results

    def get_dimless_numbers_averaged(self, tmin=0, tmax=None):
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
        return result

    def plot_dimless_numbers_versus_time(self, tmin=0, tmax=None):
        numbers_vs_time = self.get_dimless_numbers_versus_time()
        times = numbers_vs_time["t"]
        itmin, itmax = _compute_indices_tmin_tmax(times, tmin, tmax)
        stop = itmax + 1
        times = times[itmin:stop]

        fig, ax = self.output.figure_axe()

        for key, quantity in numbers_vs_time.items():
            if key in ["t", "dimensional"]:
                continue
            ax.plot(times, quantity[itmin:stop], label=key)

        ax.set_yscale("log")
        ax.set_xlabel("$t$")
        ax.set_title(f"dimensionless numbers\n{self.output.summary_simul}")

        fig.legend()


def _compute_indices_tmin_tmax(times, tmin, tmax):
    if tmax is None:
        itmax = len(times) - 1
    itmin = abs(times - tmin).argmin()
    return itmin, itmax
