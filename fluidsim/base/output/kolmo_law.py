"""Kolmogorov law 3d (:mod:`fluidsim.base.output.kolmo_law`)
==============================================================

Provides:

.. autoclass:: KolmoLaw
   :members:
   :private-members:

"""

import os
import itertools

import numpy as np
import h5py
import matplotlib.pyplot as plt

from fluiddyn.util import mpi

from fluidsim.base.output.base import SpecificOutput
from fluidsim.operators.coord_system3d import CoordSystem3DConverter
from fluidsim.operators.spatial_average3d import SpatialAverage


class KolmoLaw(SpecificOutput):
    r"""Kolmogorov law 3d.

    .. |J| mathmacro:: {\mathbf J}
    .. |Jk| mathmacro:: {\mathbf J_K}
    .. |Jp| mathmacro:: {\mathbf J_P}
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

    This output computes and saves the components of the vectors :math:`\J_\alpha`
    averaged over the azimuthal angle (i.e. as a function of :math:`r_h` and :math:`r_v`),
    as well as radially averaged (as a function of :math:`r`).

    """

    _tag = "kolmo_law"
    _name_file = _tag + ".h5"

    @classmethod
    def _complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)

    def __init__(self, output):
        params = output.sim.params

        try:
            period_save = params.output.periods_save.kolmo_law
        except AttributeError:
            period_save = 0.0

        # For testing, use a non-zero period
        if period_save == 0.0:
            period_save = params.output.periods_save.spectra
            if period_save == 0.0:
                arrays_1st_time = None
                super().__init__(output, period_save=0, arrays_1st_time=None)
                return

        # Get local coordinates
        X, Y, Z = output.sim.oper.get_XYZ_loc()
        Lx, Ly, Lz = params.oper.Lx, params.oper.Ly, params.oper.Lz

        # Initialize coordinate converter and spatial average operators
        self.coord_conv = CoordSystem3DConverter(
            X, Y, Z, Lx, Ly, Lz, shift_origin=True
        )

        ratio_dr_to_dx = 1.0
        ratio_drh_to_dx = 1.0
        ratio_drv_to_dx = 1.0

        self.spatial_avg = SpatialAverage(
            output.sim.oper,
            dr=ratio_dr_to_dx,
            drh=ratio_drh_to_dx,
            dz=ratio_drv_to_dx,
            shift_origin=True,
        )

        # Store bin centers for saving
        arrays_1st_time = {
            "r_store": self.spatial_avg.r_centers,
            "rh_store": self.spatial_avg.rho_centers,
            "rv_store": self.spatial_avg.z_centers,
        }

        super().__init__(
            output,
            period_save=period_save,
            arrays_1st_time=arrays_1st_time,
        )

    def _init_path_files(self):
        path_run = self.output.path_run
        self.path_file = os.path.join(path_run, "kolmo_law.h5")

    def _init_files(self, arrays_1st_time=None):
        if not hasattr(self, "spatial_avg"):
            return

        result = self.compute()

        if mpi.rank == 0:
            if not os.path.exists(self.path_file):
                self._create_file_from_dict_arrays(
                    self.path_file, result, arrays_1st_time
                )
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file, "r") as file:
                    dset_times = file["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                self._add_dict_arrays_to_file(self.path_file, result)

        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Save the values at one time."""
        if not hasattr(self, "spatial_avg"):
            return

        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_save >= self.period_save:
            self.t_last_save = tsim
            result = self.compute()

            if mpi.rank == 0:
                self._add_dict_arrays_to_file(self.path_file, result)
                self.nb_saved_times += 1

    def compute(self):
        """Compute the Kolmogorov law quantities at one time."""
        state = self.sim.state
        params = self.sim.params
        state_phys = state.state_phys
        state_spect = state.state_spect
        keys_state_phys = state.keys_state_phys

        fft = self.sim.oper.fft
        kx = self.sim.oper.Kx
        ky = self.sim.oper.Ky
        kz = self.sim.oper.Kz

        # Get velocity fields
        letters = "xyz"
        fft_vi = [state_spect.get_var(f"v{letter}_fft") for letter in letters]
        vel = [state_phys.get_var(f"v{letter}") for letter in letters]

        # Compute kinetic energy
        K = sum(v**2 for v in vel)
        fft_K = fft(K)

        # Compute cross products v_i * v_j
        fft_vjvi = np.empty((3, 3), dtype=object)
        for ind_i, ind_j in itertools.product(range(3), repeat=2):
            vi = vel[ind_i]
            vj = vel[ind_j]
            fft_vjvi[ind_i, ind_j] = fft(vi * vj)

        # Compute mean kinetic energy (global)
        E_k_mean = self._compute_global_mean(K)

        # Compute J_k in Fourier space
        Jk_r_fft = [None] * 3
        for ind_i in range(3):
            tmp = 2 * fft_vi[ind_i] * fft_K.conj()
            for ind_j in range(3):
                tmp += 4 * fft_vi[ind_j] * fft_vjvi[ind_i, ind_j].conj()
            tmp = 1j * tmp.imag
            Jk_r_fft[ind_i] = tmp

        # Compute divergence of J_k
        Jk_r_fft_array = np.array(Jk_r_fft)
        divJk_fft = 1j * (
            kx * Jk_r_fft_array[0]
            + ky * Jk_r_fft_array[1]
            + kz * Jk_r_fft_array[2]
        )
        divJk = self.sim.oper.ifft(divJk_fft)

        # Convert to real space
        Jk_r = [self.sim.oper.ifft(Jk_r_fft[i]) for i in range(3)]

        # Compute second-order structure function
        val = sum(fft_vi[i] * fft_vi[i].conj() for i in range(3))
        S2_k_r = 2 * E_k_mean - 2 * self.sim.oper.ifft(val)

        # If buoyancy field exists, compute J_p
        if "b" in keys_state_phys:
            b = state_phys.get_var("b")
            fft_b = state_spect.get_var("b_fft")
            b2 = b * b
            fft_b2 = fft(b2)

            # Compute mean buoyancy variance
            E_b_mean = self._compute_global_mean(b2)

            # Compute J_p
            Jp_r_fft = [None] * 3
            fft_bv = [fft(b * vel[i]) for i in range(3)]

            for ind_i in range(3):
                mom = (
                    4 * fft_bv[ind_i].conj() * fft_b
                    + 2 * fft_b2.conj() * fft_vi[ind_i]
                )
                mom = 1j * mom.imag
                Jp_r_fft[ind_i] = mom / (params.N**2)

            # Divergence of J_p
            Jp_r_fft_array = np.array(Jp_r_fft)
            divJp_fft = 1j * (
                kx * Jp_r_fft_array[0]
                + ky * Jp_r_fft_array[1]
                + kz * Jp_r_fft_array[2]
            )
            divJp = self.sim.oper.ifft(divJp_fft)

            # Convert to real space
            Jp_r = [self.sim.oper.ifft(Jp_r_fft[i]) for i in range(3)]

            # S2_p
            src = fft_b * fft_b.conj()
            S2_p_r = (2 * E_b_mean - 2 * self.sim.oper.ifft(src)) / (params.N**2)

        # Project onto coordinate system bases using CoordSystem3DConverter
        Jk_r_array = np.array(Jk_r)

        # Longitudinal (radial) component
        Jl_k = self.coord_conv.compute_radial_component(
            Jk_r_array[0], Jk_r_array[1], Jk_r_array[2]
        )

        # Cylindrical components
        Jh_k, Jt_k, Jv_k = self.coord_conv.compute_cylindrical_components(
            Jk_r_array[0], Jk_r_array[1], Jk_r_array[2]
        )

        # Azimuthal and radial averages
        results = {
            "Jl_k": Jl_k,
            "Jh_k": Jh_k,
            "Jv_k": Jv_k,
            "S2_k": S2_k_r,
            "divJ_k": divJk,
        }

        if "b" in keys_state_phys:
            Jp_r_array = np.array(Jp_r)

            Jl_p = self.coord_conv.compute_radial_component(
                Jp_r_array[0], Jp_r_array[1], Jp_r_array[2]
            )

            Jh_p, Jt_p, Jv_p = self.coord_conv.compute_cylindrical_components(
                Jp_r_array[0], Jp_r_array[1], Jp_r_array[2]
            )

            results.update(
                {
                    "Jl_p": Jl_p,
                    "Jh_p": Jh_p,
                    "Jv_p": Jv_p,
                    "S2_p": S2_p_r,
                    "divJ_p": divJp,
                }
            )

        # Compute radial and azimuthal averages using SpatialAverage
        averaged_results = {}

        for key, field in results.items():
            # Radial average
            _, avg_r = self.spatial_avg.compute_radial_average(field)
            averaged_results[f"{key}_r"] = avg_r

            # Azimuthal average
            _, _, avg_hv = self.spatial_avg.compute_azimuthal_average(field)
            averaged_results[f"{key}_hv"] = avg_hv

        return averaged_results

    def _compute_global_mean(self, field):
        """Compute global mean of a field across all MPI processes."""
        local_mean = np.mean(field)

        if mpi.nb_proc > 1:
            all_means = mpi.comm.gather(local_mean, root=0)
            if mpi.rank == 0:
                global_mean = np.mean(all_means)
            else:
                global_mean = None
            global_mean = mpi.comm.bcast(global_mean, root=0)
        else:
            global_mean = local_mean

        return global_mean

    def load_temp_average(self, key_list=[], tmin=None, tmax=None):
        """Load selected data and time average."""
        results = {}

        with h5py.File(self.path_file, "r") as file:
            times = file["times"][...]

            # Determine time range
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

            # Load and average data
            for key in key_list:
                results[key] = np.mean(file[key][imin_plot:imax_plot], axis=0)

        return results

    def plot_radial_dependencies(
        self,
        tmin=None,
        tmax=None,
        coef_comp3=1,
        coef_comp2=2 / 3,
        save=False,
    ):
        """Plot radial dependencies of Kolmogorov law quantities."""
        state = self.sim.state
        params = self.sim.params
        keys_state_phys = state.keys_state_phys

        key_list = ["S2_k_r", "divJ_k_r", "Jl_k_r"]
        if "b" in keys_state_phys:
            key_list.extend(["S2_p_r", "divJ_p_r", "Jl_p_r"])

        to_plot = self.load_temp_average(key_list, tmin, tmax)

        r_store = self.spatial_avg.r_centers

        # Compensated plots
        Jl_k_comp = -to_plot["Jl_k_r"] / (r_store**coef_comp3)
        divJ_k = to_plot["divJ_k_r"]
        S2_k_comp = to_plot["S2_k_r"] / (r_store**coef_comp2)

        # Theoretical values
        Jl_k_th = 4 / 3 * np.ones_like(r_store)
        S2_k_th = 22 / 3 * np.ones_like(r_store)

        if "b" in keys_state_phys:
            Jl_p_comp = -to_plot["Jl_p_r"] / (r_store**coef_comp3)
            divJ_p = to_plot["divJ_p_r"]
            S2_p_comp = to_plot["S2_p_r"] / (r_store**coef_comp2)

        title = f"$n_x={params.oper.nx}$"

        if mpi.rank == 0:
            # Plot J_KL
            fig1, ax1 = self.output.figure_axe()
            ax1.set_ylabel(r"$-J_{KL}(r)/r\epsilon$", fontsize="x-large")
            ax1.plot(r_store, Jl_k_comp, "b", label="Numerical result")
            if "b" in keys_state_phys:
                ax1.plot(r_store, Jl_p_comp, "g", label="$J_P$")
            ax1.plot(r_store, Jl_k_th, "r--", label="4/3 theoretical")
            ax1.set_title(
                f"$-J_{{KL}}(r)/r\\epsilon$, {title}", fontsize="x-large"
            )
            ax1.set_xlabel("$r/\\eta$", fontsize="x-large")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.legend()
            if save:
                plt.savefig("Jk_r_compensate.pdf")
            plt.show()

            # Plot div(J_KL)
            fig2, ax2 = self.output.figure_axe()
            ax2.set_ylabel(
                r"$-\nabla \cdot J_{KL}(r)/4\epsilon$", fontsize="x-large"
            )
            ax2.plot(r_store, -divJ_k / 4, "b", label="Numerical result")
            if "b" in keys_state_phys:
                ax2.plot(r_store, -divJ_p / 4, "g", label="$\\nabla \\cdot J_P$")
            ax2.plot(r_store, np.ones_like(r_store), "r--", label="1 theoretical")
            ax2.set_title(
                f"$-\\nabla \\cdot J_{{KL}}(r)/4\\epsilon$, {title}",
                fontsize="x-large",
            )
            ax2.set_xlabel("$r/\\eta$", fontsize="x-large")
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.legend()
            if save:
                plt.savefig("divJk_r_comp.pdf")
            plt.show()

            # Plot S2_K
            fig3, ax3 = self.output.figure_axe()
            ax3.set_ylabel(
                r"$S_2^K(r)/(r^{2/3}\epsilon^{2/3})$", fontsize="x-large"
            )
            ax3.plot(r_store, S2_k_comp, "b", label="Numerical result")
            if "b" in keys_state_phys:
                ax3.plot(r_store, S2_p_comp, "g", label="$S_2^P$")
            ax3.plot(r_store, S2_k_th, "r--", label="22/3 theoretical")
            ax3.set_title(
                f"$S_2^K(r)/(r^{{2/3}}\\epsilon^{{2/3}})$, {title}",
                fontsize="x-large",
            )
            ax3.set_xlabel("$r/\\eta$", fontsize="x-large")
            ax3.set_xscale("log")
            ax3.set_yscale("log")
            ax3.legend()
            if save:
                plt.savefig("S2k_r_comp.pdf")
            plt.show()

    def plot_hv_dependencies(self, tmin=None, tmax=None, save=False):
        """Plot azimuthal (rho, z) dependencies of Kolmogorov law quantities."""
        state = self.sim.state
        keys_state_phys = state.keys_state_phys
        params = self.sim.params

        key_list = ["Jl_k_hv", "divJ_k_hv"]
        if "b" in keys_state_phys:
            key_list.extend(["Jl_p_hv", "divJ_p_hv"])

        to_plot = self.load_temp_average(key_list, tmin, tmax)

        rh_store = self.spatial_avg.rho_centers
        rv_store = self.spatial_avg.z_centers
        RH, RV = np.meshgrid(rv_store, rh_store)

        # Compute radius for normalization
        radius = np.sqrt(RH**2 + RV**2)

        Jk_l_comp = -to_plot["Jl_k_hv"] / (radius + 1e-14)
        divJk_hv = to_plot["divJ_k_hv"]

        if "b" in keys_state_phys:
            Jp_l_comp = -to_plot["Jl_p_hv"] / (radius + 1e-14)
            divJp_hv = to_plot["divJ_p_hv"]

        title = f"$n_x={params.oper.nx}$"

        if mpi.rank == 0:
            # Plot J_KL(rho, z)
            fig1, ax1 = self.output.figure_axe()
            im = ax1.pcolormesh(
                RH, RV, -Jk_l_comp, cmap="Blues", vmin=0.0, vmax=1.33
            )
            fig1.colorbar(im, ax=ax1)
            ax1.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
            ax1.set_ylabel(r"$r_v/\eta$", fontsize="x-large")
            ax1.set_title(
                f"$-J_{{KL}}(r_h,r_v)/r\\epsilon$, {title}", fontsize="x-large"
            )
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            if save:
                plt.savefig("Jk_l_hv.pdf")
            plt.show()

            # Plot div(J_KL)(rho, z)
            fig2, ax2 = self.output.figure_axe()
            im = ax2.pcolormesh(
                RH, RV, -divJk_hv / 4, cmap="Blues", vmin=0.0, vmax=1.0
            )
            fig2.colorbar(im, ax=ax2)
            ax2.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
            ax2.set_ylabel(r"$r_v/\eta$", fontsize="x-large")
            ax2.set_title(
                f"$-\\nabla \\cdot J_{{KL}}(r_h,r_v)/4\\epsilon$, {title}",
                fontsize="x-large",
            )
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            if save:
                plt.savefig("divJk_hv.pdf")
            plt.show()

            if "b" in keys_state_phys:
                # Plot J_PL(rho, z)
                fig3, ax3 = self.output.figure_axe()
                im = ax3.pcolormesh(
                    RH, RV, -Jp_l_comp, cmap="Greens", vmin=0.0, vmax=1.33
                )
                fig3.colorbar(im, ax=ax3)
                ax3.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
                ax3.set_ylabel(r"$r_v/\eta$", fontsize="x-large")
                ax3.set_title(
                    f"$-J_{{PL}}(r_h,r_v)/r\\epsilon$, {title}",
                    fontsize="x-large",
                )
                ax3.set_xscale("log")
                ax3.set_yscale("log")
                if save:
                    plt.savefig("Jp_l_hv.pdf")
                plt.show()

                # Plot div(J_PL)(rho, z)
                fig4, ax4 = self.output.figure_axe()
                im = ax4.pcolormesh(
                    RH, RV, -divJp_hv / 4, cmap="Greens", vmin=0.0, vmax=1.0
                )
                fig4.colorbar(im, ax=ax4)
                ax4.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
                ax4.set_ylabel(r"$r_v/\eta$", fontsize="x-large")
                ax4.set_title(
                    f"$-\\nabla \\cdot J_{{PL}}(r_h,r_v)/4\\epsilon$, {title}",
                    fontsize="x-large",
                )
                ax4.set_xscale("log")
                ax4.set_yscale("log")
                if save:
                    plt.savefig("divJp_hv.pdf")
                plt.show()

    def plot_Jhv_vector(self, tmin=None, tmax=None, save=False):
        """Plot vector field of J in (rho, z) plane."""
        state = self.sim.state
        keys_state_phys = state.keys_state_phys
        params = self.sim.params

        key_list = ["Jh_k_hv", "Jv_k_hv"]
        if "b" in keys_state_phys:
            key_list.extend(["Jh_p_hv", "Jv_p_hv"])

        to_plot = self.load_temp_average(key_list, tmin, tmax)

        Jk_v = to_plot["Jv_k_hv"]
        Jk_h = to_plot["Jh_k_hv"]

        rh_store = self.spatial_avg.rho_centers
        rv_store = self.spatial_avg.z_centers
        RH, RV = np.meshgrid(rv_store, rh_store)

        title = f"$n_x={params.oper.nx}$"

        if mpi.rank == 0:
            fig1, ax1 = self.output.figure_axe()
            ax1.set_title(f"$-J_{{KL}}(r_h,r_v)$, {title}", fontsize="x-large")
            ax1.quiver(RH, RV, -Jk_v, -Jk_h, width=0.005)
            ax1.set_xlabel(r"$r_h$", fontsize="x-large")
            ax1.set_ylabel(r"$r_v$", fontsize="x-large")
            if save:
                plt.savefig("Jk_vector_hv.pdf")
            plt.show()

            # Normalized version
            fig2, ax2 = self.output.figure_axe()
            ax2.set_title(
                f"Normalized $-J_{{KL}}(r_h,r_v)$, {title}", fontsize="x-large"
            )
            # Avoid division by zero
            RH_safe = np.where(RH != 0, RH, 1e-10)
            RV_safe = np.where(RV != 0, RV, 1e-10)
            ax2.quiver(RH, RV, -Jk_v / RV_safe, -Jk_h / RH_safe)
            ax2.set_xlabel(r"$r_h$", fontsize="x-large")
            ax2.set_ylabel(r"$r_v$", fontsize="x-large")
            if save:
                plt.savefig("Jk_vector_hv_normalized.pdf")
            plt.show()
