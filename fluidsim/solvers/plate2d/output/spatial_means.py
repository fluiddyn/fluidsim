"""
Spatial means (:mod:`fluidsim.solvers.plate2d.output.spatial_means`)
==========================================================================


Provides:

.. autoclass:: SpatialMeansPlate2D
   :members:
   :private-members:

"""

import os
import numpy as np


from fluiddyn.util import mpi

from fluidsim.base.output.spatial_means import SpatialMeansBase


class SpatialMeansPlate2D(SpatialMeansBase):
    r"""Compute, save, load and plot spatial means.

    .. |p| mathmacro:: \partial

    If only :math:`W` is forced and dissipated, the energy budget is
    quite simple and can be written as:

    .. math::

       \p_t E_w = - C_{w\rightarrow z} - C_{w\rightarrow \chi} + P_w - D_w,

       \p_t E_z = + C_{w\rightarrow z},

       \p_t E_\chi = + C_{w\rightarrow \chi},

    where

    .. math::

       C_{w\rightarrow z} = \sum_{\mathbf{k}} k^4\mathcal{R}(\hat w \hat z^*),

       C_{w\rightarrow \chi} = -\sum_{\mathbf{k}}
       \mathcal{R}( \widehat{\{ w, z\}} \hat \chi ^* ),

       P_w = \sum_{\mathbf{k}} \mathcal{R}( \hat f_w \hat w^* )

    and

    .. math::

       D_w = 2 \nu_\alpha \sum_{\mathbf{k}} k^{2\alpha} E_w(k).
    """

    def _save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim
        (
            Ek_fft,
            El_fft,
            Ee_fft,
            conversion_k_to_l_fft,
            conversion_l_to_e_fft,
        ) = self.output.compute_energies_conversion_fft()

        energy_k = self.sum_wavenumbers(Ek_fft)
        energy_l = self.sum_wavenumbers(El_fft)
        energy_e = self.sum_wavenumbers(Ee_fft)

        conversion_k_to_l = self.sum_wavenumbers(conversion_k_to_l_fft)
        conversion_l_to_e = self.sum_wavenumbers(conversion_l_to_e_fft)

        f_d, f_d_hypo = self.sim.compute_freq_diss()
        epsK = self.sum_wavenumbers(f_d[0] * 2 * Ek_fft)
        epsK_hypo = self.sum_wavenumbers(f_d_hypo[0] * 2 * Ek_fft)

        # assert that z is not dissipated
        assert not (f_d[1].any() and f_d_hypo[1].any())

        if self.sim.params.forcing.enable:
            deltat = self.sim.time_stepping.deltat
            state_spect = self.sim.state.state_spect
            w_fft = state_spect.get_var("w_fft")

            forcing_fft = self.sim.forcing.get_forcing()
            Fw_fft = forcing_fft.get_var("w_fft")

            # assert that z in not forced
            Fz_fft = forcing_fft.get_var("z_fft")
            assert np.allclose(
                abs(Fz_fft).max(), 0.0
            ), "abs(Fz_fft).max(): {}".format(abs(Fz_fft).max())

            P1_fft = np.real(w_fft.conj() * Fw_fft)
            P2_fft = (abs(Fw_fft) ** 2) * deltat / 2
            P1 = self.sum_wavenumbers(P1_fft)
            P2 = self.sum_wavenumbers(P2_fft)

        if mpi.rank == 0:
            energy = energy_k + energy_l + energy_e
            epsK_tot = epsK + epsK_hypo

            self.file.write(f"####\ntime = {tsim:17.13f}\n")
            to_print = (
                "E    = {:31.26e} ; E_k    = {:11.6e} ; "
                "E_l    = {:11.6e} ; E_e    = {:11.6e}\n"
                "epsK = {:11.6e} ; epsK_hypo = {:11.6e} ; "
                "epsK_tot = {:11.6e}\n"
                "conv_k_to_l = {:11.6e} : conv_l_to_e = {:11.6e}\n"
            ).format(
                energy,
                energy_k,
                energy_l,
                energy_e,
                epsK,
                epsK_hypo,
                epsK + epsK_hypo,
                conversion_k_to_l,
                conversion_l_to_e,
            )

            self.file.write(to_print)

            if self.sim.params.forcing.enable:
                to_print = (
                    "P1 = {:11.6e} ; P2 = {:11.6e} ; P_tot = {:11.6e} \n"
                ).format(P1, P2, P1 + P2)
                self.file.write(to_print)

            self.file.flush()
            os.fsync(self.file.fileno())

        if self.has_to_plot and mpi.rank == 0:

            self.ax_a.plot(tsim, energy, "k.")
            self.ax_a.plot(tsim, energy_k, "r.")
            self.ax_a.plot(tsim, energy_l, "b.")
            self.ax_a.plot(tsim, energy_e, "y.")

            self.axe_b.plot(tsim, epsK_tot, "k.")
            self.axe_b.plot(tsim, conversion_k_to_l, "c.")
            self.axe_b.plot(tsim, conversion_l_to_e, "g.")
            if self.sim.params.forcing.enable:
                self.axe_b.plot(tsim, P1 + P2, "m.")

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
        lines_P = []
        lines_epsK = []

        for il, line in enumerate(lines):
            if line.startswith("time ="):
                lines_t.append(line)
            if line.startswith("E    ="):
                lines_E.append(line)
            if line.startswith("P1 ="):
                lines_P.append(line)
            if line.startswith("epsK ="):
                lines_epsK.append(line)

        nt = len(lines_t)

        t = np.empty(nt)
        E = np.empty(nt)
        E_k = np.empty(nt)
        E_l = np.empty(nt)
        E_e = np.empty(nt)

        if self.sim.params.forcing.enable:
            P1 = np.empty(nt)
            P2 = np.empty(nt)
            P_tot = np.empty(nt)

        epsK = np.empty(nt)
        epsK_hypo = np.empty(nt)
        epsK_tot = np.empty(nt)

        for il in range(nt):
            line = lines_t[il]
            words = line.split()
            t[il] = float(words[2])

            line = lines_E[il]
            words = line.split()
            E[il] = float(words[2])
            E_k[il] = float(words[6])
            E_l[il] = float(words[10])
            E_e[il] = float(words[14])

            if self.sim.params.forcing.enable:

                line = lines_P[il]
                words = line.split()
                P1[il] = float(words[2])
                P2[il] = float(words[6])
                P_tot[il] = float(words[10])

            line = lines_epsK[il]
            words = line.split()
            epsK[il] = float(words[2])
            epsK_hypo[il] = float(words[6])
            epsK_tot[il] = float(words[10])

        dict_results["t"] = t
        dict_results["E"] = E
        dict_results["E_k"] = E_k
        dict_results["E_l"] = E_l
        dict_results["E_e"] = E_e

        if self.sim.params.forcing.enable:
            dict_results["P1"] = P1
            dict_results["P2"] = P2
            dict_results["P_tot"] = P_tot

        dict_results["epsK"] = epsK
        dict_results["epsK_hypo"] = epsK_hypo
        dict_results["epsK_tot"] = epsK_tot

        return dict_results

    def plot(self, with_dtE=False):
        dict_results = self.load()

        t = dict_results["t"]
        E = dict_results["E"]
        E_k = dict_results["E_k"]
        E_l = dict_results["E_l"]
        E_e = dict_results["E_e"]

        epsK = dict_results["epsK"]
        epsK_hypo = dict_results["epsK_hypo"]
        epsK_tot = dict_results["epsK_tot"]

        if with_dtE:
            nt = len(t)
            dtE = np.empty(nt)
            for il in range(nt - 2):
                dtE[il + 1] = (E[il + 2] - E[il]) / (t[il + 2] - t[il])
            dtE[0] = (E[1] - E[0]) / (t[1] - t[0])
            dtE[nt - 1] = (E[nt - 1] - E[nt - 2]) / (t[nt - 1] - t[nt - 2])

        width_axe = 0.85
        height_axe = 0.8
        x_left_axe = 0.12
        z_bottom_axe = 0.15

        size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel("t")
        ax1.set_ylabel("Energies")
        ax1.plot(t, E, "k", linewidth=2)
        ax1.plot(t, E_k, "r", linewidth=2)
        ax1.plot(t, E_l, "b", linewidth=2)
        ax1.plot(t, E_e, "y", linewidth=2)

        size_axe[1] = z_bottom_axe
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel("$t$")
        ax1.set_ylabel(r"$\varepsilon_{tot}$, $P_{tot}$, $\partial_t E$")
        ax1.plot(t, epsK, "k--", linewidth=2)
        ax1.plot(t, epsK_hypo, "k:", linewidth=2)
        ax1.plot(t, epsK_tot, "k", linewidth=2)

        if self.sim.params.forcing.enable:
            P_tot = dict_results["P_tot"]
            ax1.plot(t, P_tot, "m", linewidth=2)

        if with_dtE:
            ax1.plot(t, dtE, "b", linewidth=2)

            should_be_zeros = dtE + epsK_tot
            if self.sim.params.forcing.enable:
                should_be_zeros -= P_tot

            ax1.plot(t, should_be_zeros, "g", linewidth=2)
