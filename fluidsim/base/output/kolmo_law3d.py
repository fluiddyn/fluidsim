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
        self.rh = np.sqrt(X**2 + Y**2)
        self.rz = Z




    def average_azimutal(self, arr):

        avg_arr = None
        if mpi.nb_proc == 1:
            avg_arr = np.mean(arr, axis=(0, 1))
        return avg_arr

        local_sum = np.sum(arr, axis=(0, 1))
        if mpi.rank == 0:
            global_arr = np.zeros(self.nz)  # define array to sum on all proc

        for rank in range(mpi.nb_proc):
            if mpi.rank == 0:
                nz_loc = self.nzs_local[rank]  # define size of array on each proc
            print("nz_loc " + str(nz_loc))
            if rank == 0 and mpi.rank == 0:
                sum = local_sum  # start the sum on rank 0
            else:
                # sum made on rank 0: receive local_array of rank
                if (
                    mpi.rank == 0
                ):
                    sum = np.empty(nz_loc)
                    mpi.comm.Recv(sum, source=rank, tag=42 * rank)
                elif mpi.rank == rank:  # send the local array to 0
                    mpi.comm.Send(sum_local, dest=0, tag=42 * rank)
            if mpi.rank == 0:  # construct global sum on 0
                iz_start = self.izs_start[rank]
                print("iz_start " + iz_start)
                global_array[iz_start : iz_start + nz_loc] += sum
        if mpi.rank == 0:
            avg_arr = global_array / (params.oper.nx * params.oper.ny)
            return avg_arr


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

        \bnabla \cdot \left( \J_K + \J_A \right) = -4 \left( \epsK + \epsA \right),

    where

    .. math::

        \J_K(\r) \equiv
          \left\langle | \delta \v |^2 \delta \v \right\rangle_\x, \\
        \J_A(\r) \equiv
          \frac{1}{N^2} \left\langle | \delta b |^2 \delta \v \right\rangle_\x.

    This output saves the components in the spherical basis of the vectors
    :math:`\J_\alpha` averaged over the azimutal angle (i.e. as a function of
    :math:`r_h` and :math:`r_z`).

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
                "rz": self.oper_kolmo_law.rz
            }
            self.rh_max = np.sqrt(params.oper.Lx**2 + params.oper.Ly**2)
            self.rz_max = params.oper.Lz
 #          n_sort = params.n_sort
            self.n_sort = 50
            n_sort = self.n_sort
            rh_sort = np.empty([n_sort])
            rz_sort = np.empty([n_sort])
            self.drhrz = {
            "drh": rh_sort,
            "drz": rz_sort,
            }


            for i in range(n_sort):
                self.drhrz["drh"][i] = self.rh_max * (i + 1) / n_sort
                self.drhrz["drz"][i] = self.rz_max * (i + 1) / n_sort
            arrays_1st_time = {
                "rh_sort": self.drhrz["drh"],
                "rz_sort": self.drhrz["drz"],
            }

        else:
            arrays_1st_time = None
        self.rhrz_sort = arrays_1st_time

        super().__init__(
            output,
            #period_save=period_save_kolmo_law,
            period_save = params.output.periods_save.spectra,
            arrays_1st_time = arrays_1st_time,
        )

    def _init_path_files(self):

        path_run = self.output.path_run
        self.path_kolmo_law = path_run + "/kolmo_law.h5"
        self.path_file = self.path_kolmo_law



    def _init_files(self, arrays_1st_time=None):
        dict_J_k_hv = self.compute()[0]
        dict_J_a_hv = self.compute()[1]
        dict_J = {}
        dict_J.update(dict_J_k_hv)
        dict_J.update(dict_J_a_hv)
        if mpi.rank == 0:
#            print("dict_J_k_hv= " + str(dict_J_k_hv))
            if not os.path.exists(self.path_kolmo_law):

                self._create_file_from_dict_arrays(
                    self.path_kolmo_law, dict_J, arrays_1st_time
                )
                self.nb_saved_times = 1
            else:
                print("Fichier modif")
                with h5py.File(self.path_kolmo_law, "r") as file:
                    dset_times = file["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                    print(self.nb_saved_times)
                self._add_dict_arrays_to_file(self.path_kolmo_law, dict_J)

        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Save the values at one time."""
        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_save >= self.period_save:
            self.t_last_save = tsim
            dict_J_k_hv = self.compute()[0]
            dict_J_a_hv = self.compute()[1]
            dict_J = {}
            dict_J.update(dict_J_k_hv)
            dict_J.update(dict_J_a_hv)
            if mpi.rank == 0:
                self._add_dict_arrays_to_file(self.path_kolmo_law, dict_J)
                self.nb_saved_times += 1

    def load(self):
        path_file = self.path_kolmo_law
        J_hv = {
            "J_k_h" : None,
            "J_k_z" : None,         
            "J_a_h" : None,
            "J_a_z" : None,
            "times" : None, 
            "count" : None                  
        }        
        file = h5py.File(path_file, "r")
        for key in file.keys():
            J_hv[key] = file[key]
        return J_hv

    def plot_kolmo_law(self):
        result = self.load()
        times=result["times"]
        J_k_h = result["J_k_h"][0]
        J_k_z = result["J_k_z"][0]
        J_a_h = result["J_a_h"][0]
        J_a_z = result["J_a_z"][0]
        J_tot_h = J_a_h + J_k_h
        J_tot_z = J_a_z + J_k_z
        count = result["count"]  

        print(str(result.items()))
        print("J_k_z = " + str(J_k_z[:]))
        print("J_k_h = " + str(J_k_h[:]))
        print("J_a_z = " + str(J_a_z[:]))
        print("J_a_h = " + str(J_a_h[:]))
        print("count = " + str(count[:]))
        print("count_tot = " + str(320*320*80) + " " + "count_sum= " + 
        str(sum(sum(count[0]))))


        posy = result["rz_sort"][:]
        posx= result["rh_sort"][:]
        U,V= np.meshgrid(posx,posy)
        toty = J_tot_z
        totx = J_tot_h

        bx = J_a_h
        by = J_a_z

        kx = J_k_h
        ky = J_k_z

        if mpi.rank == 0:
            fig, ax = self.output.figure_axe()
            self.ax = ax
            ax.set_xlabel("$rh$")
            ax.set_ylabel("$rz$")
            ax.set_title("J_tot")
            ax.quiver(posx,posy,totx,toty)

            fig2, ax2 = self.output.figure_axe()
            self.ax2 = ax2
            ax2.set_xlabel("$rh$")
            ax2.set_ylabel("$rz$")
            ax2.set_title("J_A")
            ax2.quiver(posx,posy,bx,by)

            fig3, ax3 = self.output.figure_axe()
            self.ax3 = ax3
            ax3.set_xlabel("$rh$")
            ax3.set_ylabel("$rz$")
            ax3.set_title("J_K")
            ax3.quiver(posx,posy,kx,ky)


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
        fft = self.sim.oper.fft
        letters = "xyz"

        tf_vi = [state_spect.get_var(f"v{letter}_fft") for letter in letters]
        vel = [state_phys.get_var(f"v{letter}") for letter in letters]
        tf_vjvi = np.empty((3, 3), dtype=object)
        tf_K = None

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
            else:
                tf_K += tmp

        for ind_i, ind_j in itertools.combinations(range(3), 2):
            letter_i = letters[ind_i]
            letter_j = letters[ind_j]
            vi = state_phys.get_var("v" + letter_i)
            vj = state_phys.get_var("v" + letter_j)
            tf_vjvi[ind_i, ind_j] = tf_vjvi[ind_j, ind_i] = fft(vi * vj)

        J_K_xyz = [None] * 3
        J_A_xyz = [None] * 3

        for ind_i in range(3):
            tmp = tf_vi[ind_i] * tf_K.conj()
            if "b" in keys_state_phys:
                mom = (
                    4 * tf_bv[ind_i].conj() * tf_b
                    + 2 * tf_b2.conj() * tf_vi[ind_i]
                )
                mom.real = 0.0
            for ind_j in range(3):
                tmp += tf_vi[ind_j] * tf_vjvi[ind_i, ind_j].conj()

            tmp.real = 0.0
 
            J_K_xyz[ind_i] = 4 * self.sim.oper.ifft(tmp)
            if "b" in keys_state_phys:
                J_A_xyz[ind_i] = self.sim.oper.ifft(mom)


        n_sort = self.n_sort
        J_k_z = np.zeros([n_sort, n_sort])
        J_k_h = np.zeros([n_sort, n_sort])
 

        J_a_z = np.zeros([n_sort, n_sort])
        J_a_h = np.zeros([n_sort, n_sort])
 
        count_final = np.empty([n_sort, n_sort])

        J_k_hv_average = {
            "J_k_h": J_k_h,
            "J_k_v": J_k_v,
            "count" : count_final,
        }
        J_a_hv_prov = {
            "J_a_h": J_a_h,
            "J_a_z": J_a_z,
            "count" : count_final,
        }
        count = np.zeros([n_sort, n_sort], dtype = int)
        rhrz_sort = self.rhrz_sort
        J_k_xyz = np.array(J_K_xyz)
        J_a_xyz = np.array(J_A_xyz)




        for index, value in np.ndenumerate(J_k_xyz[2]):  # average on each process

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
                J_k_xyz[0] ** 2 + J_k_xyz[1] ** 2
            )
            J_k_xyz_average["J_k_average"][r_i] += J_k_xyz_pro[index]
            S2_k_xyz_average["S2_k_average"][r_i] += S2_k_xyz[index]
            i = floor(self.rhrz["rh"][index] * n_sort / self.rh_max)
            j = floor(self.rhrz["rz"][index] * n_sort / self.rz_max)
            count[i, j] += 1
            J_k_hv_prov["J_k_z"][i, j] += np.abs(value)       
            J_k_hv_prov["J_k_h"][i, j] += np.sqrt(
                J_k_xyz[0][index] ** 2 + J_k_xyz[1][index] ** 2
            )
            J_a_hv_prov["J_a_z"][i, j] += np.abs(J_a_xyz[2][index]) 
             
            J_a_hv_prov["J_a_h"][i, j] += np.sqrt(
                J_a_xyz[0][index] ** 2 + J_a_xyz[1][index] ** 2
            )


        if mpi.nb_proc == 1:
            J_k_hv_prov["count"] = J_a_hv_prov["count"] =count
            

        if mpi.nb_proc > 0:
            collect_J_k_z = mpi.comm.gather(J_k_hv_prov["J_k_z"], root=0)
            collect_J_k_h = mpi.comm.gather(J_k_hv_prov["J_k_h"], root=0)
            collect_J_a_z = mpi.comm.gather(J_a_hv_prov["J_a_z"], root=0)
            collect_J_a_h = mpi.comm.gather(J_a_hv_prov["J_a_h"], root=0)           
            collect_count = mpi.comm.gather(count, root=0)


            if mpi.rank == 0:
                J_k_hv_prov["J_k_z"] = np.sum(collect_J_k_z, axis=0)
                J_k_hv_prov["J_k_h"] = np.sum(collect_J_k_h, axis=0)

                J_a_hv_prov["J_a_z"] = np.sum(collect_J_a_z, axis=0)
                J_a_hv_prov["J_a_h"] = np.sum(collect_J_a_h, axis=0) 

                count_final = np.sum(collect_count, axis = 0)
                J_k_hv_prov["count"] = J_a_hv_prov["count"] = count_final

                for index, value in np.ndenumerate(J_k_hv_prov["J_k_z"]):
                    if count_final[index] == 0:
                        J_k_hv_prov["J_k_z"][index] = 0.0
                        J_k_hv_prov["J_k_h"][index] = 0.0
                        J_a_hv_prov["J_a_z"][index] = 0.0
                        J_a_hv_prov["J_a_h"][index] = 0.0
                    else:
                        J_k_hv_prov["J_k_z"][index] = value / count_final[index]
                        J_k_hv_prov["J_k_h"][index] = (
                            J_k_hv_prov["J_k_h"][index] / count_final[index]
                        )
                        J_a_hv_prov["J_a_z"][index] = (
                            J_a_hv_prov["J_a_z"][index] / count_final[index]
                        )
                        J_a_hv_prov["J_a_h"][index] = (
                            J_a_hv_prov["J_a_h"][index] / count_final[index]
                        )                               
                
        result = J_k_hv_prov, J_a_hv_prov

        return result


    def check_diff_methods(self):
        first_method = compute(self)
        second_method = compute_alt(self)
        if not np.allclose(first_method, second_method):
            raise RuntimeError(
                "Both methods are inconsistent: "
                " ({self.sim.time_stepping.it = })"
            )
