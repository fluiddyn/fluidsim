"""Kolmogorov law 3d (:mod:`fluidsim.base.output.kolmo_law3d`)
==============================================================

Provides:

.. autoclass:: KolmoLaw
   :members:
   :private-members:

"""

import itertools

import numpy as np
import os
from fluiddyn.util import mpi

import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .base import SpecificOutput

from math import floor

""" Conversion from cartesian coordinates system to spherical coordinate system """


class OperKolmoLaw:
    def __init__(self, X, Y, Z, params):
        self.r = np.sqrt(X**2 + Y**2 + Z**2)

        self.rh = np.sqrt(X**2 + Y**2)

        self.rv = np.abs(Z)
        self.X = X
        self.Y = Y
        self.Z = Z


class KolmoLaw(SpecificOutput):
    r"""Kolmogorov law 3d.

    .. |J| mathmacro:: {\mathbf J}
    .. |v| mathmacro:: {\mathbf v}
    .. |x| mathmacro:: {\mathbf x}
    .. |r| mathmacro:: {\mathbf r}
    .. |Sum| mathmacro:: \sum_{\mathbf k}
    .. |bnabla| mathmacro:: \boldsymbol{\nabla}
    .. |Int| mathmacro:: \displaystyle\int
    .. |epsK| mathmacro:: \varepsilon_K
    .. |epsA| mathmacro:: \varepsilon_A
    .. |dd| mathmacro:: \mathrm{d}

    We want to test the prediction :

    .. math::

        \bnabla \cdot \left( \Jk + \Jp \right) = -4 \left( \epsK + \epsA \right),

    where

    .. math::

        \Jk(\r) \equiv
          \left\langle | \delta \v |^2 \delta \v \right\rangle_\x, \\
        \Jp(\r) \equiv
          \frac{1}{N^2} \left\langle | \delta b |^2 \delta \v \right\rangle_\x.

    This output saves the components in the spherical basis of the vectors
    :math:`\J_\alpha` averaged over the azimutal angle (i.e. as a function of
    :math:`r_h` and :math:`r_v`).

    We can take the example of the quantity :math:`\langle | \delta b |^2
    \delta \v \rangle_\x` to explain how these quantities are computed. Using
    the relation

    .. math::

        \left\langle a' b \right\rangle_\x(\r)
        =\left\langle a(\x+\r)b(\x) \right\rangle_x(\r)\\
        =\Int (a(\x-(-\r))b(x))\dd \x \\
        =a*b(-r) \\
        =TF^{-1} \left\{ \hat{a} \hat{b}^* \right\}(\r),

    it is easy to show that

    .. math::

        \left\langle |\delta b|^2 \delta \v \right\rangle_\x(\r)  = \left\langle (b'^2-bb'+b^2)\v' \right\rangle_\x(\r)-\left\langle (b'^2-bb'+b^2)\v \right\rangle_\x(\r)

        =TF^{-1} \left\{ \widehat{b^2} \hat{\v}^* \right\}(\r) - TF^{-1} \left\{ 2\hat{b} \widehat{b\v}^* \right\}(\r) + \left\langle b^2 \v\right\rangle_\x(\r)
        -\left\langle b'^2 \v \right\rangle_\x(\r) + TF^{-1} \left\{ 2\widehat{b^*} \widehat{b\v}\right\}(\r) - TF^{-1} \left\{ \widehat{b^2}^* \hat{\v} \right\}(\r)
        \\ \\
        \left\langle b^2 \v\right\rangle_\x(\r)= \left\langle b'^2 \v\right\rangle_\x(\r) \text{ with isotropy and } (ab^*)^*=a^*b
        \\ \\
        = TF^{-1} \left\{ \left(\widehat{b^2}^* \hat{\v}\right)^* \right\}(\r) - TF^{-1} \left\{ 2\hat{b} \widehat{b\v}^* \right\}(\r) + TF^{-1} \left\{ \left(2\hat{b} \widehat{b\v}^*\right)^* \right\}(\r)
        -  TF^{-1} \left\{ \widehat{b^2}^* \hat{\v}^* \right\}(\r)
        \\ \\
        ( z-z*=2i \Im(z) )
        \\ \\
        =TF^{-1} \left\{ i\Im \left[ 4 \widehat{(b \v)}^* \hat{b} + 2 \widehat{(b^2)}^* \hat{\v} \right] \right\}

    """

    _tag = "kolmo_law"
    _name_file = _tag + ".h5"

    @classmethod
    def _complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)

    def __init__(self, output):
        params = output.sim.params

        # dict containing rh and rv
        # TODO: complete arrays_1st_time
        try:
            period_save_kolmo_law = params.output.periods_save.kolmo_law
        except AttributeError:
            period_save_kolmo_law = 0.0
        period_save_kolmo_law = 0.1
        if period_save_kolmo_law != 0.0:
            X, Y, Z = output.sim.oper.get_XYZ_loc()
            self.oper_kolmo_law = OperKolmoLaw(X, Y, Z, params)

            self.rhrv = {
                "rh": self.oper_kolmo_law.rh,
                "rv": self.oper_kolmo_law.rv,
                "r": self.oper_kolmo_law.r,
            }
            self.xyz = {
                "X": self.oper_kolmo_law.X,
                "Y": self.oper_kolmo_law.Y,
                "Z": self.oper_kolmo_law.Z,
            }
            self.rh_max = np.sqrt(params.oper.Lx**2 + params.oper.Ly**2)
            self.rv_max = params.oper.Lz
            self.r_max = np.sqrt(
                params.oper.Lx**2 + params.oper.Ly**2 + params.oper.Lz**2
            )

            aspect_ratio = params.oper.nz / params.oper.nx
            self.r_store_max = self.r_max / np.sqrt(3 / 2)
            self.n_store = floor(
                np.sqrt(2 + aspect_ratio**2)
                * params.oper.nx
                * self.r_store_max
                / self.r_max
            )
            self.nv_store = params.oper.nz
            self.nh_store = floor(np.sqrt(2) * params.oper.nx)
            n_store = self.n_store
            rh_store = np.empty([self.nh_store])
            rv_store = np.empty([self.nv_store])
            r_store = np.empty([self.n_store])
            self.dict_proc = {}
            self.drhrv = {
                "drh": rh_store,
                "drv": rv_store,
                "dr": r_store,
            }

            for i in range(self.nh_store):
                index = (i + 1) / self.nh_store
                self.drhrv["drh"][i] = self.rh_max * index
            for i in range(self.nv_store):
                index = (i + 1) / self.nv_store
                self.drhrv["drv"][i] = self.rv_max * index
            for i in range(n_store):
                index = (i + 1) / n_store
                self.drhrv["dr"][i] = self.r_store_max * index

            arrays_1st_time = {
                "rh_store": self.drhrv["drh"],
                "rv_store": self.drhrv["drv"],
                "r_store": self.drhrv["dr"],
            }

        else:
            arrays_1st_time = None
        self.rhrv_store = arrays_1st_time

        super().__init__(
            output,
            # period_save=period_save_kolmo_law,
            period_save=params.output.periods_save.spectra,
            arrays_1st_time=arrays_1st_time,
        )

    def _init_path_files(self):
        path_run = self.output.path_run
        self.path_kolmo_law = path_run + "/kolmo_law.h5"
        self.path_file = self.path_kolmo_law

    def _init_files(self, arrays_1st_time=None):
        state = self.sim.state
        params = self.sim.params
        keys_state_phys = state.keys_state_phys

        result = self.compute()

        if mpi.rank == 0:
            if not os.path.exists(self.path_kolmo_law):
                self._create_file_from_dict_arrays(
                    self.path_kolmo_law, result, arrays_1st_time
                )
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_kolmo_law, "r") as file:
                    dset_times = file["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                    print(self.nb_saved_times)
                self._add_dict_arrays_to_file(self.path_kolmo_law, result)

        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Save the values at one time."""
        state = self.sim.state
        params = self.sim.params
        keys_state_phys = state.keys_state_phys
        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_save >= self.period_save:
            self.t_last_save = tsim
            result = self.compute()

            if mpi.rank == 0:
                self._add_dict_arrays_to_file(self.path_kolmo_law, result)
                self.nb_saved_times += 1

    def load_temp_average(  # load selected data and time average
        self,
        key_list=[],
        tmin=None,
        tmax=None,
    ):
        results = {}
        file = h5py.File(self.path_file, "r")
        times = file["times"][...]
        if tmax is None:
            tmax = times.max()
            imax_plot = np.argmax(times)
        else:
            imax_plot = np.argmin(abs(times - tmax))
            tmax = times[imax_plot]
        if tmin is None:
            tmin = times.min()
            imin_plot = np.argmin(times)
        else:
            imin_plot = np.argmin(abs(times - tmin))
            tmin = times[imin_plot]
        for key in key_list:
            results[key] = np.mean(file[key][imin_plot:imax_plot], axis=0)
        return results

    def plot_radial_dependencies(
        self,
        tmin=None,
        tmax=None,
        coef_comp3=1,
        coef_comp2=2 / 3,
        slope=None,
        scale=None,
        save="no",
    ):
        state = self.sim.state
        params = self.sim.params
        keys_state_phys = state.keys_state_phys
        key_list = ["S2_k_r", "divJ_k_r", "Jl_k_r"]
        if "b" in keys_state_phys:
            key_list.extend(["S2_p_r", "divJ_p_r", "Jl_p_r"])
        to_plot = self.load_temp_average(key_list, tmin, tmax)

        r_store = self.rhrv_store["r_store"][:]

        Jl_k_comp = -to_plot["Jl_k_r"][:] / (r_store**coef_comp3)
        divJ_k = to_plot["divJ_k_r"][:]
        S2_k_comp = to_plot["S2_k_r"][:] / (r_store**coef_comp2)
        Jl_k_th = 4 * (r_store / r_store) / 3
        S2_k_th = 22 * (r_store / r_store) / 3

        if "b" in keys_state_phys:
            Jl_p_comp = -to_plot["Jl_p_r"][:] / (r_store**coef_comp3)
            divJ_p = to_plot["divJ_p_r"][:]
            S2_p_comp = to_plot["S2_p_r"][:] / (r_store**coef_comp2)

        title = "$ R_\lambda=180, $" + f"$n_x={params.oper.nx}$"

        if mpi.rank == 0:
            fig1, ax1 = self.output.figure_axe()
            if scale == None or scale == "log":
                ax1.set_ylim([0.01, 2])

            ax1.set_ylabel("$-J_{KL}(r)/r\epsilon$", fontsize="x-large")
            ax1.plot(r_store, Jl_k_comp, "b", label="Numerical result")
            if "b" in keys_state_phys:
                ax1.plot(r_store, Jl_p_comp, "g", label="Numerical result")
            ax1.plot(r_store, Jl_k_th, "r", label="4/3 theoritical value")
            ax1.set_title("$-J_{KL}(r)/r\epsilon$, " + title, fontsize="x-large")
            ax1.legend()
            plt.show()

            if save == "yes":
                plt.savefig("Jk_r_compensate.pdf")
                plt.savefig("Jk_r_compensate.eps")

            fig2, ax2 = self.output.figure_axe()
            if scale == None or scale == "log":
                ax2.set_ylim([0.01, 2])

            ax2.set_ylabel("$-div(J_{KL}(r))/4\epsilon$", fontsize="x-large")
            ax2.plot(r_store, -divJ_k / 4, "b", label="Numerical result")
            if "b" in keys_state_phys:
                ax2.plot(r_store, -divJ_p, "g", label="Numerical result")
            ax2.plot(
                r_store, np.ones(len(r_store)), "r", label="1 theoritical value"
            )
            ax2.set_title(
                "$-div(J_{KL}(r))/4\epsilon$, " + title, fontsize="x-large"
            )
            ax2.legend()
            plt.show()

            if save == "yes":
                ax2.savefig("divJk_r_comp.pdf")
                ax2.savefig("divJk_r_comp.eps")

            fig3, ax3 = self.output.figure_axe()
            if scale == None or scale == "log":
                ax3.set_ylim([0.5, 10])

            ax3.set_ylabel(
                "$S2_K(r)/(r^{2/3}\epsilon^{2/3})$", fontsize="x-large"
            )
            ax3.plot(r_store, S2_k_comp, "b", label="Numerical result")
            if "b" in keys_state_phys:
                ax3.plot(r_store, S2_p_comp, "g", label="Numerical result")
            ax3.plot(r_store, S2_k_th, "r", label=" 22/3 theoritical value")
            ax3.set_title(
                "$S2_K(r)/(r^{2/3}\epsilon^{2/3}), $" + title, fontsize="x-large"
            )
            ax3.legend()
            plt.show()

            if save == "yes":
                ax3.savefig("S2k_r_comp.pdf")
                ax3.savefig("S2k_r_comp.eps")

            for axis in [ax1, ax2, ax3]:
                axis.set_xlabel("$r/\eta$", fontsize="x-large")
                axis.set_xscale("log")
                if scale == None or scale == "log":
                    axis.set_yscale("log")
                else:
                    axis.set_yscale(f"{scale}")

    def plot_hv_dependencies(
        self,
        tmin=None,
        tmax=None,
        save="no",
    ):
        state = self.sim.state
        keys_state_phys = state.keys_state_phys
        params = self.sim.params
        L = 3
        n = params.oper.nx
        dx = L / n
        eta = dx

        title = "$R_\lambda=180, $" + f"$n_x={params.oper.nx}$"

        key_list = ["Jl_k_hv", "divJ_k_hv", "divJ_k_hv"]

        if "b" in keys_state_phys:
            key_list.extend(["Jl_p_hv", "divJ_p_hv"])

        to_plot = self.load_temp_average(key_list, tmin, tmax)

        rh_store = self.rhrv_store["rh_store"][:]
        rv_store = self.rhrv_store["rv_store"][:]
        RH, RV = np.meshgrid(rv_store, rh_store)

        radius = np.zeros([self.nh_store, self.nv_store])
        for index_rh, value_rh in np.ndenumerate(rh_store):
            for index_rv, value_rv in np.ndenumerate(rv_store):
                radius[index_rh, index_rv] = np.sqrt(value_rv**2 + value_rh**2)

        Jk_l_comp = to_plot["Jl_k_hv"][:] / radius
        divJk_hv = to_plot["divJ_k_hv"][:]
        divJ_k_hv = to_plot["divJ_k_hv"][:]
        if "b" in keys_state_phys:
            Jp_l_comp = to_plot["Jl_p_hv"][:] / radius
            divJp_hv = to_plot["divJ_p_hv"][:]

        circle = patches.Circle(
            (0, 0), radius=10.0, facecolor="None", edgecolor="r", lw=5, zorder=10
        )
        circle2 = patches.Circle(
            (0, 0), radius=10.0, facecolor="None", edgecolor="r", lw=5, zorder=10
        )

        if mpi.rank == 0:
            fig1, ax1 = self.output.figure_axe()

            im = ax1.pcolormesh(
                RH,
                RV,
                -Jk_l_comp,
                cmap="Blues",
                vmin=0.0,
                vmax=1.33,
            )
            fig1.colorbar(im, ax=ax1)
            ax1.set_title(
                "$-J_{KL}(r_h,r_v)/r\epsilon$, " + title, fontsize="x-large"
            )
            ax1.legend()
            if save == "yes":
                plt.savefig("Jk_l.pdf")
                plt.savefig("Jk_l.eps")
            #           plt.show()

            fig2, ax2 = self.output.figure_axe()

            im = ax2.pcolormesh(
                RH,
                RV,
                -divJk_hv / 4,
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
            )

            fig2.colorbar(im, ax=ax2)
            ax2.set_title(
                "$-div(J_{KL}(r_h,r_v))/4\epsilon$, " + title, fontsize="x-large"
            )
            ax2.legend()
            if save == "yes":
                plt.savefig("divJk_hv.pdf")
                plt.savefig("divJk_hv.eps")
            #           plt.show()

            if "b" in keys_state_phys:
                fig3, ax3 = self.output.figure_axe()

                im = ax3.pcolormesh(
                    RH,
                    RV,
                    -divJp_hv / 4,
                    cmap="Greens",
                    vmin=0.0,
                    vmax=1.0,
                )

                fig3.colorbar(im, ax=ax3)
                ax3.set_title(
                    "$-div(J_{PL}(r_h,r_v))/4\epsilon$," + title,
                    fontsize="x-large",
                )
                ax3.legend()
                if save == "yes":
                    plt.savefig("Jp_l_comp.pdf")
                    plt.savefig("Jp_l_comp.eps")
                plt.show()

                fig4, ax4 = self.output.figure_axe()

                im = ax4.pcolormesh(
                    RH,
                    RV,
                    -Jp_l_comp,
                    cmap="Greens",
                    vmin=0.0,
                    vmax=1.33,
                )

                fig4.colorbar(im, ax=ax4)
                ax4.set_title(
                    "$-div(J_{PL}(r_h,r_v))/4\epsilon$," + title,
                    fontsize="x-large",
                )
                ax4.legend()
                if save == "yes":
                    plt.savefig("divJp_hv.pdf")
                    plt.savefig("divJp_hv.eps")
                plt.show()
                # ax8.add_patch(circle2)
                # ax8.add_patch(ellipse2)

            axes = [ax1, ax2]
            if "b" in keys_state_phys:
                key_list.extend([ax3, ax4])

            for axis in axes:
                axis.set_xlabel("$r_h/\eta$", fontsize="x-large")
                axis.set_ylabel("$r_v/\eta$", fontsize="x-large")
                axis.set_yscale("log")
                axis.set_xscale("log")

    #           axis.set_ylim([1,150])
    #          axis.set_xlim([1,150])

    def plot_Jhv_vector(
        self,
        tmin=None,
        tmax=None,
        save="no",
    ):
        state = self.sim.state
        keys_state_phys = state.keys_state_phys
        params = self.sim.params
        key_list = ["Jh_k_hv", "Jv_k_hv"]
        if "b" in keys_state_phys:
            key_list.extend(["Jh_p_hv", "Jv_p_hv"])
        to_plot = self.load_temp_average(key_list, tmin, tmax)

        Jk_v = to_plot["Jv_k_hv"][:]
        Jk_h = to_plot["Jh_k_hv"][:]
        rh_store = self.rhrv_store["rh_store"][:]
        rv_store = self.rhrv_store["rv_store"][:]
        RH, RV = np.meshgrid(rv_store, rh_store)

        title = "$R_\lambda=180, $" + f"$n_x={params.oper.nx}$"

        if mpi.rank == 0:
            fig1, ax1 = self.output.figure_axe()
            ax1.set_title("$-J_{KL}(r_h,r_v)$, " + title, fontsize="x-large")
            ax1.quiver(RH, RV, -Jk_v, -Jk_h, width=0.005)
            ax1.legend()
            #          plt.show()

            fig2, ax2 = self.output.figure_axe()
            ax2.set_title(
                "Normalised $-J_{KL}(r_h,r_v)$, " + title, fontsize="x-large"
            )
            ax2.quiver(RH, RV, -Jk_v / RV, -Jk_h / RH)
            ax2.legend()
            #           plt.show()

            axes = [ax1, ax2]
            for axis in axes:
                axis.set_xlabel("$rh$", fontsize="x-large")
                axis.set_ylabel("$rv$", fontsize="x-large")
                axis.set_xlim([0, 0.21])
                axis.set_ylim([0, 0.21])

    def plot_kolmo_law(
        self,
        tmin=None,
        tmax=None,
        delta_t=None,
        coef_comp3=1,
        coef_comp2=2 / 3,
        slope=None,
        scale=None,
    ):
        result = self.load()
        state = self.sim.state
        params = self.sim.params
        keys_state_phys = state.keys_state_phys

        times = result["times"][:]
        if tmax is None:
            tmax = times.max()
            imax_plot = np.argmax(times)
        else:
            imax_plot = np.argmin(abs(times - tmax))
            tmax = times[imax_plot]
        if tmin is None:
            tmin = times.min()
            imin_plot = np.argmin(times)
        else:
            imin_plot = np.argmin(abs(times - tmin))
            tmin = times[imin_plot]

        Jk_average = result["Jk_average"][imin_plot:imax_plot]
        divJk = result["divJk_average"][imin_plot:imax_plot]
        divJk_hv = result["divJk_hv"][imin_plot:imax_plot]
        Jk_h = result["Jk_h"][imin_plot:imax_plot]
        Jk_l = result["Jk_l"][imin_plot:imax_plot]
        Jk_v = result["Jk_v"][imin_plot:imax_plot]
        if "b" in keys_state_phys:
            Jp_l = result["Jp_l"][imin_plot:imax_plot]
            divJp_hv = result["divJp_hv"][imin_plot:imax_plot]
        S2_k_average = result["S2_k_average"][imin_plot:imax_plot]
        count = result["count"]
        count2 = result["count2"]

        Jk_average = np.mean(Jk_average, axis=0)
        Jk_l = np.mean(Jk_l, axis=0)
        divJk = np.mean(divJk, axis=0)
        divJk_hv = np.mean(divJk_hv, axis=0)
        S2_k_average = np.mean(S2_k_average, axis=0)
        Jk_h = np.mean(Jk_h, axis=0)
        Jk_v = np.mean(Jk_v, axis=0)
        Jk_hmean = np.mean(Jk_h, axis=1)
        Jk_vmean = np.mean(Jk_v, axis=0)

        L = 3
        n = params.oper.nx
        dx = L / n
        eta = dx

        rad = result["r_store"][:]

        posx = rad / eta
        unite = posx / posx
        posz = result["rv_store"][:]
        posh = result["rh_store"][:]
        RH, RV = np.meshgrid(posz, posh)

        nh_store = self.nh_store
        nv_store = self.nv_store
        if mpi.rank == 0:
            radius = np.zeros([nh_store, nv_store])
            for index_rh, value_rh in np.ndenumerate(posh):
                for index_rv, value_rv in np.ndenumerate(posz):
                    radius[index_rh, index_rv] = np.sqrt(
                        value_rv**2 + value_rh**2
                    )

            divJk_r_from_hv = np.zeros([self.n_store])
            Jk_r_from_hv = np.zeros([self.n_store])
            count_r_from_hv = np.zeros([self.n_store])

            for index, value in np.ndenumerate(divJk_hv):
                if (
                    floor(self.n_store * (radius[index] / self.r_store_max))
                    >= self.n_store
                ):
                    pass
                else:
                    pondr = 0
                    ind_r = floor(
                        self.n_store * (radius[index] / self.r_store_max)
                    )
                    ind_r1 = ind_r + 1
                    if ind_r1 < self.n_store:
                        pondr = (radius[index] - self.drhrv["dr"][ind_r]) / (
                            self.drhrv["dr"][ind_r1] - self.drhrv["dr"][ind_r]
                        )

                    divJk_r_from_hv[ind_r] += (1 - pondr) * divJk_hv[index]
                    Jk_r_from_hv[ind_r] += (1 - pondr) * Jk_l[index]
                    count_r_from_hv[ind_r] += 1 - pondr
                    if ind_r1 < self.n_store:
                        divJk_r_from_hv[ind_r] += pondr * divJk_hv[index]
                        Jk_r_from_hv[ind_r] += pondr * Jk_l[index]
                        count_r_from_hv[ind_r1] += pondr

            for index, value in np.ndenumerate(count_r_from_hv):
                if count_r_from_hv[index] == 0:
                    divJk_r_from_hv[index] = 0
                    Jk_r_from_hv[index] = 0
                else:
                    divJk_r_from_hv[index] = divJk_r_from_hv[index] / value
                    Jk_r_from_hv[index] = Jk_r_from_hv[index] / value

    def counter_proc(self):
        n_store = self.n_store
        nh_store = self.nh_store
        nv_store = self.nv_store
        arr_ind_rh = np.zeros_like(self.rhrv["r"][:], dtype=int)
        arr_ind_rv = np.zeros_like(self.rhrv["r"][:], dtype=int)
        arr_ind_r = np.zeros_like(self.rhrv["r"][:], dtype=int)
        proc_count_hv = np.zeros([nh_store, nv_store], dtype=int)
        proc_count_r = np.zeros([n_store], dtype=int)

        for index, value in np.ndenumerate(
            self.rhrv["r"][:]
        ):  # average on each process
            if floor((value / self.r_store_max) * n_store) >= n_store:
                arr_ind_rh[index] = n_store
                arr_ind_rv[index] = n_store
                arr_ind_r[index] = n_store
            else:
                arr_ind_rh[index] = floor(
                    (self.rhrv["rh"][index] / self.rh_max) * nh_store
                )
                arr_ind_rv[index] = floor(
                    (self.rhrv["rv"][index] / self.rv_max) * nv_store
                )
                arr_ind_r[index] = floor(
                    (self.rhrv["r"][index] / self.r_store_max) * n_store
                )

                proc_count_hv[arr_ind_rh[index], arr_ind_rv[index]] += 1
                proc_count_r[arr_ind_r[index]] += 1

        dict_useful = {
            "ind_rh": arr_ind_rh,
            "ind_rv": arr_ind_rv,
            "ind_r": arr_ind_r,
            "proc_count_r": proc_count_r,
            "proc_count_hv": proc_count_hv,
        }

        return dict_useful

    def counter_tot(self):
        count_ind = self.count_ind_rhv
        proc_count_r = count_ind["proc_count_r"]
        proc_count_hv = count_ind["proc_count_hv"]
        tot_count_r = np.zeros_like(proc_count_r, dtype=int)
        tot_count_hv = np.zeros_like(proc_count_r, dtype=int)
        if mpi.nb_proc > 1:
            collect_count_r = mpi.comm.gather(proc_count_r, root=0)
            collect_count_hv = mpi.comm.gather(proc_count_hv, root=0)

            if mpi.rank == 0:
                tot_count_r = np.sum(collect_count_r, axis=0)
                tot_count_hv = np.sum(collect_count_hv, axis=0)
        else:
            tot_count_r = proc_count_r
            tot_count_hv = proc_count_hv

        tot_count = {
            "tot_count_r": tot_count_r,
            "tot_count_hv": tot_count_hv,
        }
        return tot_count

    def azimut_average(self, name):
        dict_proc = self.dict_proc
        count_ind = self.count_ind_rhv
        tot_count_r = self.counter["tot_count_r"][:]
        tot_count_hv = self.counter["tot_count_hv"][:]
        mean_proc_name_hv = np.zeros([self.nh_store, self.nv_store])
        mean_proc_name_r = np.zeros([self.n_store])

        mean_tot_name_r = np.zeros([self.n_store])
        mean_tot_name_hv = np.zeros([self.nh_store, self.nv_store])

        name_r = ""
        name_hv = ""

        for index, value in np.ndenumerate(count_ind["ind_r"][:]):
            if count_ind["ind_r"][index] == self.n_store:
                pass
            else:
                mean_proc_name_r[count_ind["ind_r"][index]] += dict_proc[name][
                    index
                ]
                mean_proc_name_hv[
                    count_ind["ind_rh"][index], count_ind["ind_rv"][index]
                ] += dict_proc[name][index]

        if mpi.nb_proc > 1:
            collect_name_r = mpi.comm.gather(mean_proc_name_r, root=0)
            collect_name_hv = mpi.comm.gather(mean_proc_name_hv, root=0)

            if mpi.rank == 0:
                mean_tot_name_r = np.sum(collect_name_r, axis=0)
                mean_tot_name_hv = np.sum(collect_name_hv, axis=0)

                for index, value in np.ndenumerate(tot_count_r):
                    if tot_count_r[index] == 0:
                        mean_tot_name_r[index] = 0.0
                    else:
                        mean_tot_name_r[index] = (
                            mean_tot_name_r[index] / tot_count_r[index]
                        )
                for index, value in np.ndenumerate(tot_count_hv):
                    if tot_count_hv[index] == 0:
                        mean_tot_name_hv[index] = 0.0
                    else:
                        mean_tot_name_hv[index] = (
                            mean_tot_name_hv[index] / tot_count_hv[index]
                        )
        else:
            mean_tot_name_r = mean_proc_name_r
            mean_tot_name_hv = mean_proc_name_hv

        name_r = str(name) + "_r"
        name_hv = str(name) + "_hv"
        mean_values = {
            name_r: mean_tot_name_r,
            name_hv: mean_tot_name_hv,
        }
        return mean_values

    def compute(self):
        """compute the values at one time."""
        state = self.sim.state
        params = self.sim.params
        state_phys = state.state_phys
        state_spect = state.state_spect
        keys_state_phys = state.keys_state_phys
        X = self.xyz["X"][:]
        Y = self.xyz["Y"][:]
        Z = self.xyz["Z"][:]
        fft = self.sim.oper.fft
        letters = "xyz"
        n_store = self.n_store
        nh_store = self.nh_store
        nv_store = self.nv_store
        kx = self.sim.oper.Kx
        ky = self.sim.oper.Ky
        kz = self.sim.oper.Kz
        tf_vi = [state_spect.get_var(f"v{letter}_fft") for letter in letters]
        vel = [state_phys.get_var(f"v{letter}") for letter in letters]

        tf_vjvi = np.empty((3, 3), dtype=object)
        tf_K = None
        K = None

        if "b" in keys_state_phys:
            b = state_phys.get_var("b")
            tf_b = state_spect.get_var("b_fft")
            b2 = b * b
            tf_b2 = fft(b2)
            tf_bv = [None] * 3
            bv = [item * b for item in vel]
            for index in range(len(bv)):
                tf_bv[index] = fft(bv[index])
        for index, letter in enumerate(letters):
            vi = state_phys.get_var("v" + letter)
            vi2 = vi * vi
            tf_vjvi[index, index] = tmp = fft(vi2)
            if tf_K is None:
                tf_K = tmp
                K = vi2
            else:
                tf_K += tmp
                K += vi2

        for ind_i, ind_j in itertools.combinations(range(3), 2):
            letter_i = letters[ind_i]
            letter_j = letters[ind_j]
            vi = state_phys.get_var("v" + letter_i)
            vj = state_phys.get_var("v" + letter_j)
            tf_vjvi[ind_i, ind_j] = tf_vjvi[ind_j, ind_i] = fft(vi * vj)

        Jk_r = [None] * 3
        Jk_r_fft = [None] * 3
        if "b" in keys_state_phys:
            Jp_r = [None] * 3
            Jp_r_fft = [None] * 3

        E_k_mean = 0.0
        K_k = np.ones_like(K)
        E_k_proc = np.mean(K)

        if mpi.nb_proc > 1:
            collect_E_k = mpi.comm.gather(E_k_proc, root=0)
            if mpi.rank == 0:
                E_k_mean = np.mean(collect_E_k)
            else:
                E_k_mean = None
            E_k_mean = mpi.comm.bcast(E_k_mean, root=0)
        else:
            E_k_mean = E_k_proc
        E_k = E_k_mean * K_k

        if "b" in keys_state_phys:
            E_b_mean = 0.0
            K_b = np.ones_like(b2)
            E_b_proc = np.mean(b2)
            den_flux = np.ones_like(bv[2])
            bv_proc = np.mean(bv[2])

            if mpi.nb_proc > 1:
                collect_bv = mpi.comm.gather(bv_proc, root=0)
                collect_E_b = mpi.comm.gather(E_b_proc, root=0)
                if mpi.rank == 0:
                    bv_mean = np.mean(collect_bv)
                    E_b_mean = np.mean(collect_E_b)
                else:
                    bv_mean = None
                    E_b_mean = None
                bv_mean = mpi.comm.bcast(bv_mean, root=0)
                E_b_mean = mpi.comm.bcast(E_b_mean, root=0)
            else:
                bv_mean = bv_proc
                E_b_mean = E_b_proc

            bvz = bv_mean * den_flux

            E_b = E_b_mean * K_b

        val = None
        for ind_i in range(3):
            if val is None:
                val = tf_vi[ind_i] * tf_vi[ind_i].conj()
            else:
                val += tf_vi[ind_i] * tf_vi[ind_i].conj()

            tmp = 2 * tf_vi[ind_i] * tf_K.conj()

            if "b" in keys_state_phys:
                mom = (
                    4 * tf_bv[ind_i].conj() * tf_b
                    + 2 * tf_b2.conj() * tf_vi[ind_i]
                )
                mom.real = 0.0
            for ind_j in range(3):
                tmp += 4 * tf_vi[ind_j] * tf_vjvi[ind_i, ind_j].conj()

            tmp = 1j * tmp.imag
            mom = 1j * tmp.imag
            Jk_r_fft[ind_i] = tmp
            Jk_r[ind_i] = self.sim.oper.ifft(tmp)

            if "b" in keys_state_phys:
                Jp_r[ind_i] = self.sim.oper.ifft(mom) / (self.sim.params.N**2)
                Jp_r_fft[ind_i] = mom / (self.sim.params.N**2)

        S2_k_r = 2 * E_k - 2 * self.sim.oper.ifft(val)

        Jk_r_fft = np.array(Jk_r_fft)
        divJk_fft = 1j * (kx * Jk_r_fft[0] + ky * Jk_r_fft[1] + kz * Jk_r_fft[2])
        divJk = self.sim.oper.ifft(divJk_fft)

        if "b" in keys_state_phys:
            Jp_r_fft = np.array(Jp_r_fft)
            divJp_fft = 1j * (
                kx * Jp_r_fft[0] + ky * Jp_r_fft[1] + kz * Jp_r_fft[2]
            )
            divJp = self.sim.oper.ifft(divJp_fft)

        if "b" in keys_state_phys:
            pop = tf_vi[2] * tf_b.conj()

            KP_ex = -self.sim.oper.ifft(pop + pop.conj())
            cross_incr = 2 * bvz + 2 * KP_ex
            KP_exN = KP_ex / (self.sim.params.N**2)
            src = tf_b * tf_b.conj()
            S2_p_r = (2 * E_b - 2 * self.sim.oper.ifft(src)) / (
                self.sim.params.N**2
            )
            KP_exN = np.array(KP_exN)
            cross_incr = np.array(cross_incr)
            S2_p_r = np.array(S2_p_r)

        rhrv = self.rhrv_store
        Jk_r = np.array(Jk_r)

        S2_k_r = np.array(S2_k_r)

        Jk_r_pro = np.empty_like(X)
        Jk_h_pro = np.empty_like(X)

        for index, value in np.ndenumerate(self.rhrv["r"][:]):
            if self.rhrv["rh"][index] == 0.0:
                Jk_h_pro[index] = 0.0
            else:
                Jk_h_pro[index] = (
                    Jk_r[0][index] * X[index] + Jk_r[1][index] * Y[index]
                ) / self.rhrv["rh"][index]
            # Longitudinal projection
            if value == 0.0:
                Jk_r_pro[index] = 0.0
            else:
                Jk_r_pro[index] = (
                    Jk_r[0][index] * X[index]
                    + Jk_r[1][index] * Y[index]
                    + Jk_r[2][index] * Z[index]
                ) / value

        if "b" in keys_state_phys:
            Jp_r = np.array(Jp_r)
            Jp_r_pro = np.empty_like(X)
            Jp_h_pro = np.empty_like(X)
            for index, value in np.ndenumerate(self.rhrv["r"][:]):
                if self.rhrv["rh"][index] == 0.0:
                    Jp_h_pro[index] = 0.0
                else:
                    Jp_h_pro[index] = (
                        Jp_r[0][index] * X[index] + Jp_r[1][index] * Y[index]
                    ) / self.rhrv["rh"][index]
                # Longitudinal projection
                if value == 0.0:
                    Jp_r_pro[index] = 0.0
                else:
                    Jp_r_pro[index] = (
                        Jp_r[0][index] * X[index]
                        + Jp_r[1][index] * Y[index]
                        + Jp_r[2][index] * Z[index]
                    ) / value

        results = {}
        self.dict_proc.update(
            {
                "Jl_k": Jk_r_pro,
                "Jh_k": Jk_h_pro,
                "Jv_k": Jk_r[2],
                "S2_k": S2_k_r,
                "divJ_k": divJk,
            }
        )

        if "b" in keys_state_phys:
            self.dict_proc.update(
                {
                    "Jl_p": Jp_r_pro,
                    "Jh_p": Jp_h_pro,
                    "Jv_p": Jp_r[2],
                    "S2_p": S2_p_r,
                    "divJ_p": divJp,
                    "cross_incr": cross_incr,
                    "KP_exN": KP_exN,
                }
            )

        self.count_ind_rhv = self.counter_proc()
        self.counter = self.counter_tot()

        for key in self.dict_proc.keys():
            results.update(self.azimut_average(key))

        return results
