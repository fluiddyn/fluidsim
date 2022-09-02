"""Spatial means regions
========================

.. autoclass:: SpatialMeansRegions
   :members:
   :private-members:

"""

from pathlib import Path
import numbers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fluiddyn.util import mpi
from fluidfft.fft3d.operators import vector_product

from fluidsim.base.output.base import SpecificOutput

from . import SimulExtender


class SpatialMeansRegions(SimulExtender, SpecificOutput):
    r"""Specific output for the MILESTONE simulations

    Notes
    -----

    .. |p| mathmacro:: \partial

    .. |vv| mathmacro:: \textbf{v}

    .. |Fh| mathmacro:: \textbf{F}_h

    .. |ez| mathmacro:: \textbf{e}_\textbf{z}

    .. |bnabla| mathmacro:: \boldsymbol{\nabla}

    .. |bomega| mathmacro:: \boldsymbol{\omega}

    This output is used with the solver
    :mod:`fluidsim.solvers.ns3d.strat.solver`, which solves the incompressible
    Navier-Stokes equations (possibly with hyper-viscosity) under the
    Boussinesq approximation with a constant Brunt-Vaisala frequency:

    .. math::

      \p_t \vv + \vv \cdot \bnabla \vv = - \bnabla p + b \ez + D_\vv + \Fh,

      \p_t b   + \vv \cdot \bnabla b = - N^2 v_z  + D_b,

    where :math:`\vv` is the non-divergent velocity (:math:`\bnabla \cdot \vv =
    0`), :math:`p` is the pressure, :math:`b` is the buoyancy, :math:`N` is the
    (constant) Brunt-Vaisala frequency and :math:`D_\vv` and :math:`D_b` are
    dissipative terms.

    The total dynamic pressure :math:`P = p + |\vv|^2/2` can be computed from
    :math:`\vv` and :math:`b` as:

    .. math::

      \bnabla^2 P = \bnabla \cdot (\vv \times \bomega) + \bnabla \cdot \Fh + \p_z b.

    We consider the energy budget

    .. math::

      \p_t |\vv|^2/2 + \bnabla \cdot (\vv P) = \vv \cdot D_\vv + \vv \cdot \Fh + v_z b

      \p_t e_A +  \bnabla \cdot (\vv e_A) = - v_z b + b D_b / N^2,

    with :math:`e_A = b^2/(2N^2)`.

    We take the average of these equations over regions delimited by surfaces
    along the y and z axis. Some terms written as a divergence can be rewritten
    as fluxes through 2 surfaces at :math:`x_{min}` and :math:`x_{max}`:

    .. math::

      \int_V \frac{dV}{V} \bnabla \cdot (\vv P) =
      \frac{S}{V} \left( \langle v_x P \rangle_{S_{min}} - \langle v_x P \rangle_{S_{max}} \right).

    """

    _tag = "spatial_means_regions"
    _module_name = "fluidsim.extend_simul.spatial_means_regions_milestone"

    def __init__(self, output):

        params = output.sim.params

        params_cls = params.output.spatial_means_regions
        self.xmin_given = params_cls.xmin
        self.xmax_given = params_cls.xmax

        if isinstance(self.xmin_given, numbers.Number):
            self.xmin_given = [self.xmin_given]

        if isinstance(self.xmax_given, numbers.Number):
            self.xmax_given = [self.xmax_given]

        self.nb_regions = len(self.xmin_given)

        if len(self.xmax_given) != self.nb_regions:
            raise ValueError("len(self.xmax_given) != len(self.xmin_given)")

        oper = output.sim.oper

        Lx = params.oper.Lx
        if not params.ONLY_COARSE_OPER:
            x_seq = oper.x_seq
        else:
            x_seq = Lx / params.oper.nx * np.arange(params.oper.nx)

        self.info_regions = []

        _, _, ix_seq_start = oper.seq_indices_first_X
        nx_loc = oper.shapeX_loc[2]

        for xmin, xmax in zip(self.xmin_given, self.xmax_given):

            xmin, xmax = Lx * xmin, Lx * xmax

            ixmin = np.argmin(abs(x_seq - xmin))
            xmin = x_seq[ixmin]

            ixmin_loc = ixmin - ix_seq_start
            if ixmin_loc < 0 or ixmin_loc > nx_loc - 1:
                # this limit is not in this process
                ixmin_loc = None

            ixmax = np.argmin(abs(x_seq - xmax))
            xmax = x_seq[ixmax]
            ixmax_loc = ixmax - ix_seq_start
            ixstop_loc = ixmax_loc + 1

            # special case when region == whole numerical domain
            if ixmax == params.oper.nx - 1 and ixmin == 0:
                xmax += x_seq[1]
                ixmax += 1

            ixmax_surf = ixmax % params.oper.nx
            ixmax_surf_loc = ixmax_surf - ix_seq_start
            if ixmax_surf_loc < 0 or ixmax_surf_loc > nx_loc - 1:
                # this limit is not in this process
                ixmax_surf_loc = None

            self.info_regions.append(
                (xmin, xmax, ixmin, ixmax, ixmin_loc, ixmax_surf_loc, ixstop_loc)
            )

        super().__init__(
            output, period_save=params.output.periods_save.spatial_means_regions
        )

        if self.period_save == 0:
            return

        self.masks = []
        self.nb_points = []

        self.nb_points_xmin = []
        self.nb_points_xmax = []

        for info_region in self.info_regions:
            (ixmin_loc, ixmax_surf_loc, ixstop_loc) = info_region[4:]
            mask_loc = np.zeros(shape=oper.shapeX_loc, dtype=np.int8)
            mask_loc[:, :, ixmin_loc:ixstop_loc] = 1
            self.masks.append(mask_loc)

            nb_points = mask_loc.sum()
            if mpi.nb_proc > 1:
                nb_points = mpi.comm.allreduce(nb_points, op=mpi.MPI.SUM)
            self.nb_points.append(nb_points)

            def compute_nb_points_surface(ixsurface_loc):
                if ixsurface_loc is not None:
                    nz_loc = oper.shapeX_loc[0]
                    ny_loc = oper.shapeX_loc[1]
                    nb_points_xsurface_loc = ny_loc * nz_loc
                else:
                    nb_points_xsurface_loc = np.int8(0)
                if mpi.nb_proc > 1:
                    nb_points_xsurface = mpi.comm.allreduce(
                        nb_points_xsurface_loc, op=mpi.MPI.SUM
                    )
                else:
                    nb_points_xsurface = nb_points_xsurface_loc

                return nb_points_xsurface

            self.nb_points_xmin.append(compute_nb_points_surface(ixmin_loc))
            self.nb_points_xmax.append(compute_nb_points_surface(ixmax_surf_loc))

        self._save_one_time()

    def _init_path_files(self):
        self.path_dir = Path(self.output.path_run) / self._tag
        self.paths = [
            self.path_dir / f"data{iregion}.csv"
            for iregion in range(self.nb_regions)
        ]

    def _init_files(self, arrays_1st_time=None):
        if mpi.rank == 0:
            self.path_dir.mkdir(exist_ok=True)
            for path, info_region in zip(self.paths, self.info_regions):
                xmin, xmax = info_region[:2]
                if not path.exists():
                    with open(path, "w") as file:
                        file.write(
                            f"# xmin = {xmin} ; xmax = {xmax}\n"
                            "time,EK,EKz,EA,epsK,epsA,conv_K2A,PK,PA,"
                            "flux_Pnl_xmin,flux_Pnl_xmax,flux_v2_xmin,flux_v2_xmax,"
                            "flux_P_dz_b_xmin,flux_P_dz_b_xmax,"
                            "flux_Pforcing_xmin,flux_Pforcing_xmax,"
                            "flux_A_xmin,flux_A_xmax\n"
                        )
                else:
                    with open(path, "r") as file:
                        words = file.readline().split()
                        xmin_file = float(words[3])
                        xmax_file = float(words[7])
                        if xmin_file != xmin or xmax_file != xmax:
                            raise ValueError(
                                "xmin_file != xmin or xmax_file != xmax\n"
                                f"{xmin_file} != {xmin} or {xmax_file} != {xmax}"
                            )

    @classmethod
    def complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)
        params.output._set_child(
            cls._tag, attribs={"xmin": [0.1, 0.4, 0.7], "xmax": [0.3, 0.6, 0.9]}
        )

    @classmethod
    def get_modif_info_solver(cls):
        """Create a function to modify ``info_solver``.

        Note that this function is called when the object ``info_solver`` has
        not yet been created (and cannot yet be modified)! This is why one
        needs to create a function that will be called later to modify
        ``info_solver``.

        """

        def modif_info_solver(info_solver):
            info_solver.classes.Output.classes._set_child(
                "SpatialMeansRegions",
                attribs={
                    "module_name": cls._module_name,
                    "class_name": cls.__name__,
                },
            )

        return modif_info_solver

    def _online_save(self):
        if self._has_to_online_save():
            self._save_one_time()

    def _compute_means_regions(self, field):
        results = []
        for nb_points, mask in zip(self.nb_points, self.masks):
            result = (mask * field).sum()
            if mpi.nb_proc > 1:
                result = mpi.comm.allreduce(result, op=mpi.MPI.SUM)
            results.append(result / nb_points)
        return np.array(results)

    def _compute_fluxes_regions(self, field, vx):
        """Compute for each region the fluxes over xmin and xmax surface"""
        fluxes = []
        for index_region, info_region in enumerate(self.info_regions):
            xmin, xmax = info_region[:2]
            length_region = xmax - xmin
            ixmin_loc, ixmax_surf_loc = info_region[4:-1]

            def compute_flux(nb_points_surfaces, ixsurface_loc):
                nb_points_surface = nb_points_surfaces[index_region]
                if ixsurface_loc is not None:
                    field_xsurface = field[:, :, ixsurface_loc]
                    vx_xsurface = vx[:, :, ixsurface_loc]
                    sum_xsurface = (vx_xsurface * field_xsurface).sum()
                else:
                    sum_xsurface = 0.0
                if mpi.nb_proc > 1:
                    sum_xsurface = mpi.comm.allreduce(
                        sum_xsurface, op=mpi.MPI.SUM
                    )
                return sum_xsurface / nb_points_surface / length_region

            flux_xmin = compute_flux(self.nb_points_xmin, ixmin_loc)
            flux_xmax = -compute_flux(self.nb_points_xmax, ixmax_surf_loc)
            fluxes.append((flux_xmin, flux_xmax))
        return np.array(fluxes)

    def _save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim

        state = self.sim.state
        oper = self.sim.oper

        ifft = oper.ifft
        fft = oper.fft
        ifft_as_arg_destroy = oper.ifft_as_arg_destroy

        get_var = state.state_phys.get_var
        b = get_var("b")
        vx = get_var("vx")
        vy = get_var("vy")
        vz = get_var("vz")

        N2b2 = 0.5 / self.sim.params.N**2 * b * b
        EAs = self._compute_means_regions(N2b2)

        vh2 = 0.5 * (vx * vx + vy * vy)
        vz2 = 0.5 * vz * vz
        v2_over_2 = vh2 + vz2

        EKhs = self._compute_means_regions(vh2)
        EKzs = self._compute_means_regions(vz2)
        del vh2, vz2

        EKs = EKhs + EKzs

        conv_K2A = -self._compute_means_regions(b * vz)

        get_var = state.state_spect.get_var
        b_fft = get_var("b_fft")
        vx_fft = get_var("vx_fft")
        vy_fft = get_var("vy_fft")
        vz_fft = get_var("vz_fft")

        f_d, _ = self.sim.compute_freq_diss()

        f_d_vx = ifft(f_d * vx_fft)
        f_d_vy = ifft(f_d * vy_fft)
        f_d_vz = ifft(f_d * vz_fft)
        epsKs = self._compute_means_regions(
            vx * f_d_vx + vy * f_d_vy + vz * f_d_vz
        )
        del f_d_vx, f_d_vy, f_d_vz

        epsAs = [
            epsA / self.sim.params.N**2
            for epsA in self._compute_means_regions(b * ifft(f_d * b_fft))
        ]

        if self.sim.params.forcing.enable:
            deltat = self.sim.time_stepping.deltat
            forcing_fft = self.sim.forcing.get_forcing()

            fx_fft = forcing_fft.get_var("vx_fft")
            fy_fft = forcing_fft.get_var("vy_fft")

            assert np.allclose(forcing_fft.get_var("vz_fft"), 0)

            fx = ifft(fx_fft)
            fy = ifft(fy_fft)

            PKs = self._compute_means_regions(
                vx * fx + vy * fy + (abs(fx) ** 2 + abs(fy) ** 2) * deltat / 2
            )

        else:
            PKs = np.zeros(3)

        # Compute spatial fluxes
        # Need to compute the pressure P = v^2/2 + p

        omegax_fft, omegay_fft, omegaz_fft = oper.rotfft_from_vecfft(
            vx_fft, vy_fft, vz_fft
        )

        ifft_as_arg_destroy = oper.ifft_as_arg_destroy

        omegax = state.fields_tmp[3]
        omegay = state.fields_tmp[4]
        omegaz = state.fields_tmp[5]

        ifft_as_arg_destroy(omegax_fft, omegax)
        ifft_as_arg_destroy(omegay_fft, omegay)
        ifft_as_arg_destroy(omegaz_fft, omegaz)

        del omegax_fft, omegay_fft, omegaz_fft

        f_nl_x, f_nl_y, f_nl_z = vector_product(
            vx, vy, vz, omegax, omegay, omegaz
        )

        del omegax, omegay, omegaz

        f_nl_x_fft = fft(f_nl_x)
        f_nl_y_fft = fft(f_nl_y)
        f_nl_z_fft = fft(f_nl_z)

        f_nl_x_fft *= 1j * oper.Kx
        dx_f_nl_x_fft = f_nl_x_fft
        del f_nl_x_fft

        f_nl_y_fft *= 1j * oper.Ky
        dy_f_nl_y_fft = f_nl_y_fft
        del f_nl_y_fft

        f_nl_z_fft *= 1j * oper.Kz
        dz_f_nl_z_fft = f_nl_z_fft
        del f_nl_z_fft

        P_nl_fft = -(dx_f_nl_x_fft + dy_f_nl_y_fft + dz_f_nl_z_fft) / oper.K2_not0
        del dx_f_nl_x_fft, dy_f_nl_y_fft, dz_f_nl_z_fft
        P_nl = ifft(P_nl_fft)
        del P_nl_fft

        if self.sim.params.forcing.enable:
            fz_fft = forcing_fft.get_var("vz_fft")
            P_forcing = -ifft(
                oper.divfft_from_vecfft(fx_fft, fy_fft, fz_fft) / oper.K2_not0
            )

        P_dz_b = -ifft(1j * oper.Kz * b_fft / oper.K2_not0)

        fluxes_P_nl = self._compute_fluxes_regions(P_nl, vx)
        fluxes_v2 = self._compute_fluxes_regions(v2_over_2, vx)
        fluxes_P_forcing = self._compute_fluxes_regions(P_forcing, vx)
        fluxes_P_dz_b = self._compute_fluxes_regions(P_dz_b, vx)
        fluxes_A = self._compute_fluxes_regions(N2b2, vx)

        if mpi.rank > 0:
            return

        for i, path in enumerate(self.paths):
            flux_Pnl_xmin, flux_Pnl_xmax = fluxes_P_nl[i]
            flux_v2_xmin, flux_v2_xmax = fluxes_v2[i]
            flux_Pforcing_xmin, flux_Pforcing_xmax = fluxes_P_forcing[i]
            flux_P_dz_b_xmin, flux_P_dz_b_xmax = fluxes_P_dz_b[i]
            flux_A_xmin, flux_A_xmax = fluxes_A[i]

            with open(path, "a") as file:
                file.write(
                    f"{tsim},{EKs[i]},{EKzs[i]},{EAs[i]},"
                    f"{epsKs[i]},{epsAs[i]},{conv_K2A[i]},{PKs[i]},0.0,"
                    f"{flux_Pnl_xmin},{flux_Pnl_xmax},{flux_v2_xmin},{flux_v2_xmax},"
                    f"{flux_P_dz_b_xmin},{flux_P_dz_b_xmax},"
                    f"{flux_Pforcing_xmin},{flux_Pforcing_xmax},"
                    f"{flux_A_xmin},{flux_A_xmax}\n"
                )

    def load(self, iregion=0):
        df = pd.read_csv(self.paths[iregion], skiprows=1)
        return df

    def print_keys(self):
        """Print the keys associated with the computed quantities"""
        df = pd.read_csv(self.paths[0], skiprows=1, nrows=1)
        print(df.columns)

    def plot(self, keys="EK", iregion=0):
        """Plot some quantities for a given region"""
        df = self.load(iregion)
        ax = df.plot(x="time", y=keys)
        xmin, xmax = self.info_regions[iregion][:2]
        ax.set_title(self.output.summary_simul + f"\nxmin={xmin}, xmax={xmax}")
        return ax

    def plot_budget(
        self, iregion=0, decompose_fluxes=False, plot_conversion=False
    ):
        """Plot the energy budget for a given region"""
        df = self.load(iregion)
        times = df["time"]

        E_tot = df["EK"] + df["EA"]
        dt_E_tot = np.gradient(E_tot, times)

        P_tot = df["PK"] + df["PA"]
        eps = df["epsK"] + df["epsA"]

        fig, ax = plt.subplots()

        ax.plot(times, dt_E_tot, "k--", label="$d_t E$", linewidth=2)

        ax.plot(times, P_tot, label="Forcing")
        ax.plot(times, -eps, label="Viscosity")

        kinds = "flux_Pnl flux_Pforcing flux_A flux_P_dz_b".split()

        flux_tot = np.zeros_like(times)

        for kind in kinds:
            flux_kind = df[kind + "_xmin"] + df[kind + "_xmax"]
            flux_tot += flux_kind.values

        ax.plot(times, flux_tot, label="Surface fluxes")
        ax.plot(times, P_tot - eps + flux_tot, label="All terms", linewidth=2)

        if decompose_fluxes:
            for kind in kinds:
                flux_kind = df[kind + "_xmin"] + df[kind + "_xmax"]
                ax.plot(times, flux_kind, ":", label=kind)

        if plot_conversion:
            ax.plot(times, df["conv_K2A"], "-.", label=r"$C_{K\rightarrow A}$")

        xmin, xmax = self.info_regions[iregion][:2]
        ax.set_title(
            self.output.summary_simul + f"\nxmin={xmin:.3f}, xmax={xmax:.3f}"
        )
        ax.set_xlabel("time")

        fig.legend()
        fig.tight_layout()

        return ax
