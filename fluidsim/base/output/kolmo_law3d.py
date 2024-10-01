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
import matplotlib.pyplot as plt

from .base import SpecificOutput

from math import floor

""" Conversion from cartesian coordinates system to spherical coordinate system """


class OperKolmoLaw:
    def __init__(self, X, Y, Z, params):
        self.r = np.sqrt(X**2 + Y**2 + Z**2)
        #        self.r = X
        self.rh = np.sqrt(X**2 + Y**2)
        #        self.rh = Y
        self.rz = np.abs(Z)
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

        \bnabla \cdot \left( \J_K + \J_p \right) = -4 \left( \epsK + \epsA \right),

    where

    .. math::

        \J_K(\r) \equiv
          \left\langle | \delta \v |^2 \delta \v \right\rangle_\x, \\
        \J_p(\r) \equiv
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

        # dict containing rh and rz
        # TODO: complete arrays_1st_time
        try:
            period_save_kolmo_law = params.output.periods_save.kolmo_law
        except AttributeError:
            period_save_kolmo_law = 0.0
        period_save_kolmo_law = 0.1
        if period_save_kolmo_law != 0.0:
            X, Y, Z = output.sim.oper.get_XYZ_loc()
            self.oper_kolmo_law = OperKolmoLaw(X, Y, Z, params)

            self.rhrz = {
                "rh": self.oper_kolmo_law.rh,
                "rv": self.oper_kolmo_law.rz,
                "r": self.oper_kolmo_law.r,
            }
            self.xyz = {
                "X": self.oper_kolmo_law.X,
                "Y": self.oper_kolmo_law.Y,
                "Z": self.oper_kolmo_law.Z,
            }
            self.rh_max = np.sqrt(params.oper.Lx**2 + params.oper.Ly**2)
            self.rz_max = params.oper.Lz
            self.r_max = np.sqrt(
                params.oper.Lx**2 + params.oper.Ly**2 + params.oper.Lz**2
            )

            self.n_store = 120
            n_store = self.n_store
            rh_store = np.empty([n_store])
            rz_store = np.empty([n_store])
            r_store = np.empty([n_store])
            self.drhrz = {
                "drh": rh_store,
                "drz": rz_store,
                "dr": r_store,
            }
            self.pow_store = 5 / 4
            pow_store = self.pow_store
            for i in range(n_store):
                index = ((i + 1) / n_store) ** (pow_store)
                self.drhrz["drh"][i] = self.rh_max * index
                self.drhrz["drz"][i] = self.rz_max * index
                self.drhrz["dr"][i] = self.r_max * index
            arrays_1st_time = {
                "rh_store": self.drhrz["drh"],
                "rz_store": self.drhrz["drz"],
                "r_store": self.drhrz["dr"],
            }

        else:
            arrays_1st_time = None
        self.rhrz_store = arrays_1st_time

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

        dict_J = {}
        result = self.compute()
        for name in result:
            dict_J.update(name)

        if mpi.rank == 0:

            if not os.path.exists(self.path_kolmo_law):

                self._create_file_from_dict_arrays(
                    self.path_kolmo_law, dict_J, arrays_1st_time
                )
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_kolmo_law, "r") as file:
                    dset_times = file["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                    print(self.nb_saved_times)
                self._add_dict_arrays_to_file(self.path_kolmo_law, dict_J)

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
            dict_J = {}
            for name in result:
                dict_J.update(name)

            if mpi.rank == 0:
                self._add_dict_arrays_to_file(self.path_kolmo_law, dict_J)
                self.nb_saved_times += 1

    def load(self):
        path_file = self.path_kolmo_law
        state = self.sim.state
        params = self.sim.params
        keys_state_phys = state.keys_state_phys
        if "b" in keys_state_phys:
            J_hv = {
                "J_k_average": None,
                "J_k_h": None,
                "J_k_v": None,
                "S2_k_average": None,
                "J_p_h": None,
                "J_p_v": None,
                "times": None,
                "count": None,
            }
        else:
            J_hv = {
                "J_k_average": None,
                "J_k_h": None,
                "J_k_v": None,
                "S2_k_average": None,
                "times": None,
                "count": None,
                "count2": None,
            }
        file = h5py.File(path_file, "r")
        for key in file.keys():
            J_hv[key] = file[key]
        return J_hv

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

        if "b" in keys_state_phys:

            J_k_h = result["J_k_h"][1]
            J_k_v = result["J_k_v"][2]
            J_p_h = result["J_p_h"][4]
            J_p_v = result["J_p_v"][5]
            J_tot_h = J_p_h + J_k_h
            J_tot_v = J_p_v + J_k_v
            count = result["count"]

            print("count = " + str(count[:]))
            print(
                "count_tot = "
                + str(320 * 320 * 80)
                + " "
                + "count_sum= "
                + str(sum(sum(count[0])))
            )

            posy = result["rz_store"][:]
            posx = result["rh_store"][:]
            U, V = np.meshgrid(posx, posy)
            toty = J_tot_v
            totx = J_tot_h

            bx = J_p_h
            by = J_p_v

            kx = J_k_h
            ky = J_k_v

            if mpi.rank == 0:
                fig, ax = self.output.figure_axe()
                self.ax = ax
                ax.set_xlabel("$rh$")
                ax.set_ylabel("$rz$")
                ax.set_title("J_tot")
                ax.quiver(posx, posy, totx, toty)

                fig2, ax2 = self.output.figure_axe()
                self.ax2 = ax2
                ax2.set_xlabel("$rh$")
                ax2.set_ylabel("$rz$")
                ax2.set_title("J_p")
                ax2.quiver(posx, posy, bx, by)

                fig3, ax3 = self.output.figure_axe()
                self.ax3 = ax3
                ax3.set_xlabel("$rh$")
                ax3.set_ylabel("$rz$")
                ax3.set_title("J_k_average")
                ax3.quiver(posx, posy, kx, ky)

        else:

            J_k_average = result["J_k_average"][imin_plot:imax_plot]
            J_k_average = -J_k_average
            S2_k_average = result["S2_k_average"][imin_plot:imax_plot]
            count = result["count"]
            count2 = result["count2"]
            J_k_average = np.mean(J_k_average, axis=0)
            S2_k_average = np.mean(S2_k_average, axis=0)
            L = 3
            n = params.oper.nx
            dx = L / n
            eta = dx
            print(
                "count_tot = "
                + str(n * n * n)
                + " "
                + "count_sum= "
                + str((sum(count[0])))
                + " "
                + str((count[0]))
            )

            rad = result["r_store"][:]
            print(str(rad))
            posx = rad / eta
            unite = posx / posx

            posxprime = rad[0:5] / eta
            compa = 4 / 3 * unite
            pos3y = J_k_average
            pos2y = S2_k_average

            pos3ycomp = pos3y / (rad ** (coef_comp3))
            pos2ycomp = pos2y / (rad ** (coef_comp2))

            if slope is not None:
                check_slope = 0.01 * (posxprime**slope)
            E_k = 4 * 1.27 * np.ones([len(rad)])
            if mpi.rank == 0:
                fig1, ax1 = self.output.figure_axe()
                self.ax1 = ax1
                ax1.set_xlabel("$r/eta$")
                ax1.set_ylabel("$J_L(r)$")
                # /(r\epsilon)
                if scale is None:
                    ax1.set_yscale("log")
                    ax1.set_xscale("log")
                    ax1.set_ylim([0.001, 2.0])
                else:
                    ax1.set_yscale(f"{scale}")
                    ax1.set_xscale("log")
                if slope is not None:
                    ax1.plot(posxprime, check_slope, label=f"$r^{2}$")
                ax1.plot(posx, pos3ycomp, label="$J_{LC}(r)$")
                ax1.set_title(f"tmin={tmin:.2g}, tmax={tmax:.2g}")
                #               ax1.plot(posx,compa, label = "4/3")
                ax1.set_xscale("log")
                #             ax1.set_yscale("log")
                ax1.legend()

                fig2, ax2 = self.output.figure_axe()
                self.ax2 = ax2
                ax2.set_xlabel("$r/eta$")
                ax2.set_ylabel("$S2(r)/(r^{2/3}\epsilon^{2/3})$")
                if scale is None:
                    ax2.set_yscale("log")
                    ax2.set_xscale("log")
                    ax2.set_ylim([0.9, 10.0])
                else:
                    ax2.set_yscale(f"{scale}")
                    ax2.set_xscale("log")
                if slope is not None:
                    ax2.plot(posxprime, check_slope, label=f"$r^{2}$")

                ax2.plot(posx, pos2ycomp)
                ax2.set_title(f"tmin={tmin:.2g}, tmax={tmax:.2g}")

                ax2.legend()

                fig3, ax3 = self.output.figure_axe()
                self.ax3 = ax3
                ax3.set_xlabel("$r/eta$")
                ax3.set_ylabel("$S2(r)$")
                if scale is None:
                    ax3.set_yscale("log")
                    ax3.set_xscale("log")
                    ax3.set_ylim([0.9, 10.0])
                else:
                    ax3.set_yscale(f"{scale}")
                    ax3.set_xscale("log")
                if slope is not None:
                    ax3.plot(posxprime, check_slope, label=f"$r^{2}$")
                ax3.plot(posx, E_k, label="4*E")
                ax3.plot(posx, pos2y, label=f"S2(r)")
                ax3.set_title(f"tmin={tmin:.2g}, tmax={tmax:.2g}")

                ax3.legend()

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
        vol = params.oper.nx * params.oper.ny * params.oper.nz

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

        J_k_r = [None] * 3
        if "b" in keys_state_phys:
            J_p_r = [None] * 3

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

            J_k_r[ind_i] = self.sim.oper.ifft(tmp)

            if "b" in keys_state_phys:
                J_p_r[ind_i] = self.sim.oper.ifft(mom)

        S2_k_r = 2 * E_k - 2 * self.sim.oper.ifft(val)

        nh_store = n_store
        nv_store = n_store

        J_k_v = np.zeros([nh_store, nv_store])
        J_k_h = np.zeros([nh_store, nv_store])
        J_k_average = np.zeros([n_store])
        S2_k_average = np.zeros([n_store])

        if "b" in keys_state_phys:
            J_p_v = np.zeros([nh_store, nv_store])
            J_p_h = np.zeros([nh_store, nv_store])

        count_final = np.empty([nh_store, nv_store])
        count_final_iso = np.empty([n_store])

        J_k_hv_average = {
            "J_k_h": J_k_h,
            "J_k_v": J_k_v,
            "count": count_final,
        }

        J_k_r_average = {
            "J_k_average": J_k_average,
            "count": count_final_iso,
        }

        S2_k_r_average = {
            "S2_k_average": S2_k_average,
            "count2": count_final_iso,
        }
        if "b" in keys_state_phys:
            J_p_hv_average = {
                "J_p_h": J_p_h,
                "J_p_v": J_p_v,
                "count": count_final,
            }

        count = np.zeros([nh_store, nv_store], dtype=int)
        count_iso = np.zeros([n_store], dtype=int)

        rhrz = self.rhrz_store
        J_k_r = np.array(J_k_r)
        S2_k_r = np.array(S2_k_r)
        J_k_r_pro = np.empty_like(X)

        for index, value in np.ndenumerate(
            self.rhrz["r"][:]
        ):  # Longitudinal projection
            if value == 0.0:
                J_k_r_pro[index] = 0.0
            else:
                J_k_r_pro[index] = (
                    J_k_r[0][index] * X[index]
                    + J_k_r[1][index] * Y[index]
                    + J_k_r[2][index] * Z[index]
                ) / value

        if "b" in keys_state_phys:
            J_p_r = np.array(J_p_r)

        pow_store = self.pow_store

        for index, value in np.ndenumerate(J_k_r[2]):  # average on each process

            rh_i = floor(
                ((self.rhrz["rh"][index] / self.rh_max) ** (1 / pow_store))
                * n_store
            )
            rv_i = floor(
                ((self.rhrz["rv"][index] / self.rh_max) ** (1 / pow_store))
                * n_store
            )
            r_i = floor(
                ((self.rhrz["r"][index] / self.r_max) ** (1 / pow_store))
                * n_store
            )

            count[rh_i, rv_i] += 1
            count_iso[r_i] += 1

            J_k_hv_average["J_k_v"][rh_i, rv_i] += value
            J_k_hv_average["J_k_h"][rh_i, rv_i] += np.sqrt(
                J_k_r[0][index] ** 2 + J_k_r[1][index] ** 2
            )
            J_k_r_average["J_k_average"][r_i] += J_k_r_pro[index]
            S2_k_r_average["S2_k_average"][r_i] += S2_k_r[index]

            if "b" in keys_state_phys:
                J_p_hv_average["J_p_v"][rh_i, rv_i] += J_p_r[2][index]
                J_p_hv_average["J_p_h"][rh_i, rv_i] += np.sqrt(
                    J_p_r[0][index] ** 2 + J_p_r[1][index] ** 2
                )

        if mpi.nb_proc > 1:  # average on one process

            collect_J_k_average = mpi.comm.gather(
                J_k_r_average["J_k_average"], root=0
            )  # gather all results on one process
            collect_S2_k_average = mpi.comm.gather(
                S2_k_r_average["S2_k_average"], root=0
            )
            collect_count_iso = mpi.comm.gather(count_iso, root=0)

            collect_J_k_v = mpi.comm.gather(J_k_hv_average["J_k_v"], root=0)
            collect_J_k_h = mpi.comm.gather(J_k_hv_average["J_k_h"], root=0)
            collect_count = mpi.comm.gather(count, root=0)

            if "b" in keys_state_phys:
                collect_J_p_v = mpi.comm.gather(J_p_hv_average["J_p_v"], root=0)
                collect_J_p_h = mpi.comm.gather(J_p_hv_average["J_p_h"], root=0)

            if mpi.rank == 0:

                J_k_r_average["J_k_average"] = np.sum(collect_J_k_average, axis=0)
                S2_k_r_average["S2_k_average"] = np.sum(
                    collect_S2_k_average, axis=0
                )
                J_k_hv_average["J_k_v"] = np.sum(collect_J_k_v, axis=0)
                J_k_hv_average["J_k_h"] = np.sum(collect_J_k_h, axis=0)

                if "b" in keys_state_phys:

                    J_p_hv_average["J_p_v"] = np.sum(collect_J_p_v, axis=0)
                    J_p_hv_average["J_p_h"] = np.sum(collect_J_p_h, axis=0)

                count_final = np.sum(collect_count, axis=0)
                J_k_hv_average["count"] = count_final

                count_final_iso = np.sum(collect_count_iso, axis=0)
                J_k_r_average["count"] = count_final_iso
                S2_k_r_average["count2"] = count_final_iso

                for index, value in np.ndenumerate(J_k_hv_average["J_k_v"]):
                    if count_final[index] == 0:
                        J_k_hv_average["J_k_v"][index] = 0.0
                        J_k_hv_average["J_k_h"][index] = 0.0
                        if "b" in keys_state_phys:
                            J_p_hv_average["J_p_v"][index] = 0.0
                            J_p_hv_average["J_p_h"][index] = 0.0
                    else:
                        J_k_hv_average["J_k_v"][index] = (
                            value / count_final[index]
                        )
                        J_k_hv_average["J_k_h"][index] = (
                            J_k_hv_average["J_k_h"][index] / count_final[index]
                        )
                        if "b" in keys_state_phys:
                            J_p_hv_average["J_p_v"][index] = (
                                J_p_hv_average["J_p_v"][index]
                                / count_final[index]
                            )
                            J_p_hv_average["J_p_h"][index] = (
                                J_p_hv_average["J_p_h"][index]
                                / count_final[index]
                            )

                for index, value in np.ndenumerate(J_k_r_average["J_k_average"]):
                    if count_final_iso[index] == 0:
                        J_k_r_average["J_k_average"][index] = 0.0
                        S2_k_r_average["S2_k_average"][index] = 0.0
                    else:
                        J_k_r_average["J_k_average"][index] = (
                            value / count_final_iso[index]
                        )
                        S2_k_r_average["S2_k_average"][index] = (
                            S2_k_r_average["S2_k_average"][index]
                            / count_final_iso[index]
                        )

        if "b" in keys_state_phys:
            result = J_k_r_average, J_k_hv_average, J_p_hv_average, S2_k_r_average

        else:
            result = J_k_r_average, J_k_hv_average, S2_k_r_average
        return result
