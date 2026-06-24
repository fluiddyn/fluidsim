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
from warnings import warn


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

        if params.ONLY_COARSE_OPER:
            self.coord_conv = None
            self.spatial_avg = None
            period_save = 0.0

        if period_save == 0.0:
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

    def _init_files(self, arrays_1st_time=None):
        if self.spatial_avg is None and self.coord_conv is None:
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
        if self.spatial_avg is None and self.coord_conv is None:
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

        # Compute mean kinetic energy
        if "b" in keys_state_phys:
            nrj_tot_A, nrj_tot_Kz, nrj_tot_Khr, nrj_tot_Khd = (
                self.output.compute_energies()
            )
            E_k_mean = nrj_tot_Kz + nrj_tot_Khr + nrj_tot_Khd
        else:
            E_k_mean = self.output.compute_energy()

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
        S2_k_r = 4 * E_k_mean - 2 * self.sim.oper.ifft(val)

        # If buoyancy field exists, compute J_p
        if "b" in keys_state_phys:
            b = state_phys.get_var("b")
            fft_b = state_spect.get_var("b_fft")
            b2 = b * b
            fft_b2 = fft(b2)

            # Compute mean buoyancy variance
            E_b_mean = nrj_tot_A * params.N**2

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
            S2_p_r = (4 * E_b_mean - 2 * self.sim.oper.ifft(src)) / (params.N**2)

        # Project onto coordinate system bases using CoordSystem3DConverter
        Jk_r_array = np.array(Jk_r)

        Jl_k = self.coord_conv.compute_radial_component(
            Jk_r_array[0], Jk_r_array[1], Jk_r_array[2]
        )

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
            _, avg_r = self.spatial_avg.compute_radial_average(field)
            averaged_results[f"{key}_r"] = avg_r

            _, _, avg_hv = self.spatial_avg.compute_azimuthal_average(field)
            averaged_results[f"{key}_hv"] = avg_hv

        return averaged_results

    def load_temp_average(self, keys=None, tmin=None, tmax=None):
        """Load selected data and time average."""
        results = {}

        with h5py.File(self.path_file, "r") as file:
            times = file["times"][...]

            if keys is None:
                keys = [
                    k
                    for k in file.keys()
                    if not any(
                        k.startswith(begin) for begin in ["r", "info_", "times"]
                    )
                ]

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
                if imin_plot == imax_plot:
                    if imin_plot == 0:
                        imax_plot += 1
                        tmax = times[imax_plot]
                    else:
                        imin_plot -= 1
                tmin = times[imin_plot]

            # Load and average data
            for key in keys:
                results[key] = np.mean(file[key][imin_plot:imax_plot], axis=0)

        return results, tmin, tmax

    def plot_radial_dependencies(
        self,
        tmin=None,
        tmax=None,
        coef_comp3=1,
        coef_comp2=2 / 3,
        which_plot="div_J",
        epsilon=None,
        save=False,
    ):
        """Plot radial dependencies of Kolmogorov law quantities."""
        self._raise_parallel_error("plot_radial_dependencies")

        state = self.sim.state
        params = self.sim.params
        keys_state_phys = state.keys_state_phys

        keys = [
            "S2_k_r",
            "divJ_k_r",
            "Jl_k_r",
        ]
        if "b" in keys_state_phys:
            keys.extend(["S2_p_r", "divJ_p_r", "Jl_p_r"])

        to_plot, tmin, tmax = self.load_temp_average(keys, tmin, tmax)

        dimless_num = self.sim.output.spatial_means.get_dimless_numbers_averaged(
            tmin=tmin, tmax=tmax
        )["dimensional"]
        try:
            eta = dimless_num["eta"]
        except KeyError:
            warn("KeyError: 'eta' not available; eta is set to unity")
            eta = 1
        if epsilon is None:
            epsilon = dimless_num["epsK"]
            if "b" in keys_state_phys:
                epsilon += dimless_num["epsA"]
        EK = dimless_num["EKh"] + dimless_num["EKz"]
        if "b" in keys_state_phys:
            EA = dimless_num["EA"]

        with h5py.File(self.path_file, "r") as file:
            r_store = np.array(file["r_store"])

        # Compensated plots
        Jl_k_comp = -to_plot["Jl_k_r"] / ((r_store * epsilon) ** coef_comp3)
        divJ_k = -to_plot["divJ_k_r"] / (4 * epsilon)
        S2_k_comp = to_plot["S2_k_r"] / ((r_store * epsilon) ** coef_comp2)

        # Theoretical values
        Jl_k_th = 4 / 3 * np.ones_like(r_store)
        S2_k_th = 22 / 3 * np.ones_like(r_store)
        EK_array = EK * np.ones_like(r_store)
        if "b" in keys_state_phys:
            EA_array = EA * np.ones_like(r_store)

        if "b" in keys_state_phys:
            Jl_p_comp = -to_plot["Jl_p_r"] / (r_store**coef_comp3)
            divJ_p = -to_plot["divJ_p_r"] / (4 * epsilon)
            S2_p_comp = to_plot["S2_p_r"] / (r_store**coef_comp2)

        title = f"$N_x={params.oper.nx}$"

        match which_plot:
            case "J":
                fig1, ax1 = self.output.figure_axe()
                ax1.set_ylabel(r"$-J_L(r)/r\epsilon$", fontsize="x-large")
                ax1.plot(r_store[1:] / eta, Jl_k_comp[1:], "b", label="$J_{K,L}$")
                if "b" in keys_state_phys:
                    ax1.plot(
                        r_store[1:] / eta,
                        Jl_p_comp[1:],
                        "g",
                        label="$J_{P,L}$",
                    )
                    ax1.plot(
                        r_store[1:] / eta,
                        Jl_k_comp[1:] + Jl_p_comp[1:],
                        "k",
                        label="$J_L = J_{K,L} + J_{P,L}$",
                    )
                ax1.plot(
                    r_store[1:] / eta, Jl_k_th[1:], "r--", label="4/3 theoretical"
                )
                ax1.set_title(
                    f"$-J_L(r)/r\\epsilon$, {title}", fontsize="x-large"
                )
                ax1.set_xlabel("$r/\\eta$", fontsize="x-large")
                ax1.set_xscale("log")
                ax1.set_yscale("log")
                ax1.set_xlim(xmax=1e3)
                ax1.set_ylim(ymin=1e-2)
                ax1.legend()
                plt.tight_layout()
                if save:
                    plt.savefig("J_L_r_compensate.png", dpi=300)

            case "div_J":
                fig2, ax2 = self.output.figure_axe()
                ax2.set_ylabel(
                    r"$-\nabla \cdot J_L(r)/4\epsilon$", fontsize="x-large"
                )
                ax2.plot(
                    r_store[1:] / eta,
                    divJ_k[1:],
                    "b",
                    label="$\\nabla \cdot J_{K,L}$",
                )
                if "b" in keys_state_phys:
                    ax2.plot(
                        r_store[1:] / eta,
                        divJ_p[1:],
                        "g",
                        label="$\\nabla \\cdot J_{P,L}$",
                    )
                    ax2.plot(
                        r_store[1:] / eta,
                        divJ_p[1:] + divJ_k[1:],
                        "k",
                        label="$\\nabla \\cdot J_L = \\nabla \cdot (J_{K,L} + J_{P,L})$",
                    )
                ax2.plot(
                    r_store[1:] / eta,
                    np.ones_like(r_store[1:]),
                    "r--",
                    label="1 theoretical",
                )
                ax2.set_title(
                    f"$-\\nabla \\cdot J_L(r)/4\\epsilon$, {title}",
                    fontsize="x-large",
                )
                ax2.set_xlabel("$r/\\eta$", fontsize="x-large")
                ax2.set_xscale("log")
                ax2.set_yscale("log")
                ax2.set_xlim(xmax=1e3)
                ax2.set_ylim(ymin=1e-2)
                ax2.legend()
                plt.tight_layout()
                if save:
                    plt.savefig("divJ_L_r_comp.png", dpi=300)

            case "S2":
                fig3, ax3 = self.output.figure_axe()
                ax3.set_ylabel(
                    r"$S_2(r)/(r^{2/3}\epsilon^{2/3})$", fontsize="x-large"
                )
                ax3.plot(r_store[1:] / eta, S2_k_comp[1:], "b", label="$S_2^K$")
                if "b" in keys_state_phys:
                    ax3.plot(
                        r_store[1:] / eta, S2_p_comp[1:], "g", label="$S_2^P$"
                    )
                    ax3.plot(
                        r_store[1:] / eta, EA_array[1:], "gray--", label=r"$E_A$"
                    )
                ax3.plot(
                    r_store[1:] / eta,
                    S2_k_th[1:],
                    "r--",
                    label="22/3 theoretical",
                )
                ax3.plot(r_store[1:] / eta, EK_array[1:], "k--", label=r"$E_K$")
                ax3.set_title(
                    f"$S_2(r)/(r^{{2/3}}\\epsilon^{{2/3}})$, {title}",
                    fontsize="x-large",
                )
                ax3.set_xlabel("$r/\\eta$", fontsize="x-large")
                ax3.set_xscale("log")
                ax3.set_yscale("log")
                ax3.set_xlim(xmax=1e3)
                ax3.set_ylim(ymin=1e-2)
                ax3.legend()
                plt.tight_layout()
                if save:
                    plt.savefig("S2_r_comp.png", dpi=300)

            case _:
                raise ValueError(
                    f"Field {which_plot} not available. "
                    f"Available fields: {', '.join(keys)}"
                )
        plt.show()

    def plot_hv_dependencies(
        self,
        tmin=None,
        tmax=None,
        vmin=None,
        vmax=None,
        which_plot="div_JK",
        logscale=True,
        epsilon=None,
        cmap="plasma",
        save=False,
    ):
        """Plot azimuthal (rho, z) dependencies of Kolmogorov law quantities."""
        self._raise_parallel_error("plot_hv_dependencies")

        state = self.sim.state
        keys_state_phys = state.keys_state_phys
        params = self.sim.params

        keys = ["Jl_k_hv", "divJ_k_hv"]
        if "b" in keys_state_phys:
            keys.extend(["Jl_p_hv", "divJ_p_hv"])

        to_plot, tmin, tmax = self.load_temp_average(keys, tmin, tmax)

        dimless_num = self.sim.output.spatial_means.get_dimless_numbers_averaged(
            tmin=tmin, tmax=tmax
        )["dimensional"]
        try:
            eta = dimless_num["eta"]
        except KeyError:
            warn("KeyError: 'eta' not available; eta is set to unity")
            eta = 1
        if epsilon is None:
            epsilon = dimless_num["epsK"]
            if "b" in keys_state_phys:
                epsilon += dimless_num["epsA"]

        with h5py.File(self.path_file, "r") as file:
            rh_store = np.array(file["rh_store"])
            rv_store = np.array(file["rv_store"])

        RH, RV = np.meshgrid(rh_store, rv_store)

        # Compute radius for normalization
        radius = np.sqrt(RH**2 + RV**2)

        Jk_l_comp = -to_plot["Jl_k_hv"] / ((radius + 1e-14) * epsilon)
        divJk_hv = -to_plot["divJ_k_hv"] / (4 * epsilon)

        if "b" in keys_state_phys:
            Jp_l_comp = -to_plot["Jl_p_hv"] / ((radius + 1e-14) * epsilon)
            divJp_hv = -to_plot["divJ_p_hv"] / (4 * epsilon)

        title = f"$N_x={params.oper.nx}$"

        def _plot(j_l, cmap, vmin, vmax, type_plot="K", divergence=False):
            coma = ""
            if type_plot != "":
                coma = ","
            full_title = (
                f"$-J_{{{type_plot}{coma}L}}(r_h,r_v)/r\\epsilon$, {title}"
            )
            save_name_file = f"J{type_plot}_L_hv.png"
            if divergence:
                full_title = f"$-\\nabla \\cdot J_{{{type_plot}}}(r_h,r_v)/4\\epsilon$, {title}"
                save_name_file = f"divJ{type_plot}_hv.png"
            fig, ax = self.output.figure_axe()
            im = ax.pcolormesh(
                RH[1:] / eta,
                RV[1:] / eta,
                j_l,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
            ax.set_ylabel(r"$r_v/\eta$", fontsize="x-large")
            ax.set_title(full_title, fontsize="x-large")
            ax.set_aspect("equal", "box")
            if logscale:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(xmin=1, xmax=400)
                ax.set_ylim(ymin=1, ymax=400)
            else:
                ax.set_xlim(xmin=0, xmax=400)
                ax.set_ylim(ymin=0, ymax=400)
            plt.tight_layout()
            if save:
                plt.savefig(save_name_file, dpi=300)
            plt.show()

        # Reference vmin = -0.5, vmax = 1.2

        match which_plot:
            case "JK":
                _plot(
                    Jk_l_comp[1:],
                    cmap,
                    vmin,
                    vmax,
                    type_plot="K",
                    divergence=False,
                )

            case "div_JK":
                _plot(
                    divJk_hv[1:],
                    cmap,
                    vmin,
                    vmax,
                    type_plot="K",
                    divergence=True,
                )

            case "JP" | "div_JP" | "J" | "div_J" as what_plot:
                if "b" not in keys_state_phys:
                    raise ValueError(
                        f"Cannot plot '{which_plot}': buoyancy field 'b' is not present. "
                        f"Available fields: {', '.join(keys)}"
                    )

                match what_plot:
                    case "JP":
                        _plot(
                            Jp_l_comp[1:],
                            cmap,
                            vmin,
                            vmax,
                            type_plot="P",
                            divergence=False,
                        )

                    case "div_JP":
                        _plot(
                            divJp_hv[1:],
                            cmap,
                            vmin,
                            vmax,
                            type_plot="P",
                            divergence=True,
                        )

                    case "J":
                        _plot(
                            Jp_l_comp[1:] + Jk_l_comp[1:],
                            cmap,
                            vmin,
                            vmax,
                            type_plot="",
                            divergence=False,
                        )

                    case "div_J":
                        _plot(
                            divJp_hv[1:] + divJk_hv[1:],
                            cmap,
                            vmin,
                            vmax,
                            type_plot="",
                            divergence=True,
                        )
            case _:
                raise ValueError(
                    f"Field {which_plot} not available. "
                    f"Available fields: {', '.join(keys)}"
                )

    def plot_Jhv_vector(
        self,
        tmin=None,
        tmax=None,
        which_plot="JK",
        ratio_vectors=2,
        shifted=True,
        logscale=True,
        epsilon=None,
        theory=False,
        vect_theory=False,
        ani_param=1,
        cmap="plasma",
        save=False,
    ):
        """Plot vector field of J in (rho, z) plane."""
        self._raise_parallel_error("plot_Jhv_vector")

        state = self.sim.state
        keys_state_phys = state.keys_state_phys
        params = self.sim.params

        keys = ["Jh_k_hv", "Jv_k_hv"]
        if "b" in keys_state_phys:
            keys.extend(["Jh_p_hv", "Jv_p_hv"])

        to_plot, _, _ = self.load_temp_average(keys, tmin, tmax)

        Jk_v = to_plot["Jv_k_hv"][::ratio_vectors, ::ratio_vectors]
        Jk_h = to_plot["Jh_k_hv"][::ratio_vectors, ::ratio_vectors]
        if "b" in keys_state_phys:
            Jp_v = to_plot["Jv_p_hv"][::ratio_vectors, ::ratio_vectors]
            Jp_h = to_plot["Jh_p_hv"][::ratio_vectors, ::ratio_vectors]

        with h5py.File(self.path_file, "r") as file:
            rh_store = np.array(file["rh_store"])
            rv_store = np.array(file["rv_store"])

        RH, RV = np.meshgrid(rh_store, rv_store)

        dimless_num = self.sim.output.spatial_means.get_dimless_numbers_averaged(
            tmin=tmin, tmax=tmax
        )["dimensional"]
        try:
            eta = dimless_num["eta"]
        except KeyError:
            warn("KeyError: 'eta' not available; eta is set to unity")
            eta = 1
        if epsilon is None:
            epsilon = dimless_num["epsK"]
            if "b" in keys_state_phys:
                epsilon += dimless_num["epsA"]

        RH, RV = np.meshgrid(rh_store, rv_store)

        RH /= eta

        rv_min = np.min(rv_store)
        rv_zero_line = None

        if shifted:
            rv_zero_line = np.abs(rv_min) / eta
            RV = RV - rv_min
            RV_label = r"$(r_v - r_{v,min})/\eta$"
        else:
            RV_label = r"$r_v/\eta$"

        RV /= eta

        RH_sub = RH[::ratio_vectors, ::ratio_vectors]
        RV_sub = RV[::ratio_vectors, ::ratio_vectors]

        title = f"$N_x={params.oper.nx}$"

        def _plot(j_v, j_h, type_plot="_K", normalized=False, theory=False):
            full_title = f"$-J{type_plot}(r_h,r_v)/4\epsilon$, {title}"
            save_name_file = f"J{type_plot}_vector_hv.png"
            if theory:
                save_name_file = f"J{type_plot}_vector_hv_with_theory.png"
            j_v_plot = j_v.copy()
            j_h_plot = j_h.copy()
            C = np.sqrt(j_v_plot**2 + j_h_plot**2)

            if normalized:
                full_title = f"$Normalized -J{type_plot}(r_h,r_v)$, {title}"
                save_name_file = f"J{type_plot}_vector_hv_normalized.png"
                RH_safe = np.where(RH_sub != 0, RH_sub, 1e-10)
                RV_safe = np.abs(np.where(RV_sub != 0, RV_sub, 1e-10))
                j_v_plot /= RV_safe
                j_h_plot /= RH_safe

            fig, ax = self.output.figure_axe()
            ax.set_title(full_title, fontsize="x-large")
            ax.set_aspect("equal", "box")
            axis_max = 400
            axis_min = 0
            width = 0.002
            headwidth = 3
            headlength = 2.5
            headaxislength = 2.5
            if logscale:
                axis_min = 1
                ax.set_xscale("log")
                ax.set_yscale("log")
                width = 0.002
                headwidth = 2
                headlength = 1.5
                headaxislength = 1.5

            if theory:
                axis_max = 200
                axis_min = 30

            ax.set_xlim(xmin=axis_min, xmax=axis_max)
            ax.set_ylim(ymin=axis_min, ymax=axis_max)

            if theory:
                r_max = min(RH.max(), RV.max())
                r_min = max(RH.min(), RV.min())
                rh_line = np.linspace(r_min, r_max, 100)

                if ani_param < 0:
                    num_r = 50
                else:
                    num_r = 25

                rv_at_rmax = np.linspace(r_min, r_max, num_r)
                C_consts_right = rv_at_rmax / r_max**ani_param
                rh_at_rvmax = np.linspace(r_min, r_max, num_r)
                C_consts_top = r_max / rh_at_rvmax**ani_param
                C_consts = np.unique(
                    np.concatenate([C_consts_right, C_consts_top])
                )

                if not vect_theory:
                    for i, C_const in enumerate(C_consts):
                        rv_line = C_const * rh_line**ani_param
                        mask = (rv_line >= r_min) & (rv_line <= r_max)
                        if mask.sum() < 2:
                            continue
                        label_plot = (
                            rf"$r_v = \alpha\, r_h^{{{ani_param}}}$"
                            if i == 0
                            else None
                        )
                        ax.plot(
                            rh_line[mask],
                            rv_line[mask],
                            "r-",
                            linewidth=0.8,
                            alpha=0.3,
                            label=label_plot,
                        )

                offset = (r_max - r_min) * 0.003  # Léger décalage vertical
                J_h_theory = RH_sub / (ani_param + 2)
                J_v_theory = ani_param * RV_sub / (ani_param + 2)

                norm_th = np.sqrt(J_h_theory**2 + J_v_theory**2)
                norm_th = np.where(norm_th != 0, norm_th, 1e-10)
                J_h_theory /= norm_th
                J_v_theory /= norm_th
                norm = np.sqrt(j_h_plot**2 + j_v_plot**2)
                norm = np.where(norm != 0, norm, 1e-10)
                J_h_theory *= norm
                J_v_theory *= norm

                if vect_theory:
                    ax.quiver(
                        RH_sub,
                        RV_sub + offset,
                        J_h_theory,
                        J_v_theory,
                        color="k",
                        width=width,
                        headwidth=headwidth,
                        headlength=headlength,
                        headaxislength=headaxislength,
                        alpha=0.5,
                        label=rf"$\alpha = {ani_param}$",
                    )

                if not shifted:
                    ax.legend(
                        fontsize="x-large",
                        loc="upper right",
                        handlelength=0.5,
                    )

            quiv = ax.quiver(
                RH_sub,
                RV_sub,
                j_h_plot,
                j_v_plot,
                C,
                cmap=cmap,
                width=width,
                headwidth=headwidth,
                headlength=headlength,
                headaxislength=headaxislength,
            )

            cbar = fig.colorbar(quiv, ax=ax)
            cbar.set_label("Amplitude", fontsize="x-large")

            if shifted and rv_zero_line is not None:
                ax.axhline(
                    y=rv_zero_line,
                    color="r",
                    linestyle="--",
                    linewidth=1.5,
                    label="$r_v = 0$",
                )
                ax.legend(fontsize="x-large", loc="lower left")

            ax.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
            ax.set_ylabel(RV_label, fontsize="x-large")
            plt.tight_layout()

            if save:
                plt.savefig(save_name_file, dpi=300)
            plt.show()

        match which_plot:
            case "JK":
                _plot(
                    -Jk_v / (4 * epsilon),
                    -Jk_h / (4 * epsilon),
                    type_plot="_K",
                    normalized=False,
                    theory=theory,
                )

            case "JK_norm":
                _plot(
                    -Jk_v / (4 * epsilon),
                    -Jk_h / (4 * epsilon),
                    type_plot="_K",
                    normalized=True,
                )

            case "JP" | "JP_norm" | "J" | "J_norm" as what_plot:
                if "b" not in keys_state_phys:
                    raise ValueError(
                        f"Cannot plot '{which_plot}': buoyancy field 'b' is not present. "
                        f"Available fields: {', '.join(keys)}"
                    )

                match what_plot:
                    case "JP":
                        _plot(
                            -Jp_v / (4 * epsilon),
                            -Jp_h / (4 * epsilon),
                            type_plot="_P",
                            normalized=False,
                        )

                    case "JP_norm":
                        _plot(
                            -Jp_v / (4 * epsilon),
                            -Jp_h / (4 * epsilon),
                            type_plot="_P",
                            normalized=True,
                        )

                    case "J":
                        _plot(
                            -(Jp_v + Jk_v) / (4 * epsilon),
                            -(Jp_h + Jk_h) / (4 * epsilon),
                            type_plot="",
                            normalized=False,
                            theory=theory,
                        )

                    case "J_norm":
                        _plot(
                            -(Jp_v + Jk_v) / (4 * epsilon),
                            -(Jp_h + Jk_h) / (4 * epsilon),
                            type_plot="",
                            normalized=True,
                        )

            case _:
                raise ValueError(
                    f"Field {which_plot} not available. "
                    f"Available fields: {', '.join(keys)}"
                )
