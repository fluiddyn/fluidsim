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

from fluiddyn.util import mpi

from fluidsim.base.output.base import SpecificOutput

from . import SimulExtender


class SpatialMeansRegions(SimulExtender, SpecificOutput):
    """Specific output for the MILESTONE simulations

    It is still a work in progress.

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
            if ixmax_loc < 0 or ixmax_loc > nx_loc - 1:
                # this limit is not in this process
                ixmax_loc = None

            self.info_regions.append(
                (xmin, xmax, ixmin, ixmax, ixmin_loc, ixmax_loc, ixstop_loc)
            )

        super().__init__(
            output,
            period_save=params.output.periods_save.spatial_means_regions,
        )

        if self.period_save == 0:
            return

        self.masks = []
        self.nb_points = []
        for info_region in self.info_regions:
            (ixmin_loc, ixmax_loc, ixstop_loc) = info_region[4:]
            mask_loc = np.zeros(shape=oper.shapeX_loc, dtype=np.int8)
            mask_loc[:, :, ixmin_loc:ixstop_loc] = 1
            self.masks.append(mask_loc)

            nb_points = mask_loc.sum()
            if mpi.nb_proc > 1:
                nb_points = mpi.comm.allreduce(nb_points, op=mpi.MPI.SUM)
            self.nb_points.append(nb_points)

        self._save_one_time()

    def _init_path_files(self):
        self.path_dir = Path(self.output.path_run) / self._tag
        self.paths = [
            self.path_dir / f"data{iregion}.csv"
            for iregion in range(self.nb_regions)
        ]

    def _init_files(self, arrays_1st_time=None):
        if mpi.rank == 0:
            self.path_dir.mkdir(exist_ok=False)
            for path, info_region in zip(self.paths, self.info_regions):
                xmin, xmax = info_region[:2]
                if not path.exists():
                    with open(path, "w") as file:
                        file.write(
                            f"# xmin = {xmin} ; xmax = {xmax}\n"
                            "time,EK,EKz,EA,epsK,epsA,PK,PA\n"
                        )
                else:
                    with open(path, "r") as file:
                        words = file.readline().split()
                        xmin_file = words[3]
                        xmax_file = words[7]
                        if xmin_file != xmin or xmax_file != xmax:
                            raise ValueError

    @classmethod
    def complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)
        params.output._set_child(cls._tag, attribs={"xmin": 0.25, "xmax": 0.75})

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
        return results

    def _save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim
        ifft = self.sim.oper.ifft

        from fluidsim.operators.operators3d import (
            compute_energy_from_1field,
            compute_energy_from_1field_with_coef,
            compute_energy_from_2fields,
        )

        get_var = self.sim.state.state_phys.get_var
        b = get_var("b")
        vx = get_var("vx")
        vy = get_var("vy")
        vz = get_var("vz")

        EAs = self._compute_means_regions(0.5 * self.sim.params.N ** 2 * b * b)
        EKzs = self._compute_means_regions(0.5 * vz * vz)
        EKxs = self._compute_means_regions(0.5 * vx * vx)
        EKys = self._compute_means_regions(0.5 * vy * vy)

        EKhs = [EKx + EKy for EKx, EKy in zip(EKxs, EKys)]
        EKs = [EKh + EKz for EKh, EKz in zip(EKhs, EKzs)]

        get_var = self.sim.state.state_spect.get_var
        b_fft = get_var("b_fft")
        vx_fft = get_var("vx_fft")
        vy_fft = get_var("vy_fft")
        vz_fft = get_var("vz_fft")

        energyA_fft = compute_energy_from_1field_with_coef(
            b_fft, 1.0 / self.sim.params.N ** 2
        )
        E_Kz_fft = compute_energy_from_1field(vz_fft)
        E_Kh_fft = compute_energy_from_2fields(vx_fft, vy_fft)

        energyK_fft = E_Kz_fft + E_Kh_fft

        f_d, _ = self.sim.compute_freq_diss()

        epsKs = self._compute_means_regions(
            ifft((2 * f_d * energyK_fft).astype(complex))
        )
        epsAs = self._compute_means_regions(
            ifft((2 * f_d * energyA_fft).astype(complex))
        )

        if self.sim.params.forcing.enable:
            deltat = self.sim.time_stepping.deltat
            forcing_fft = self.sim.forcing.get_forcing()

            fx_fft = forcing_fft.get_var("vx_fft")
            fy_fft = forcing_fft.get_var("vy_fft")
            fz_fft = forcing_fft.get_var("vz_fft")

            PK1_fft = (
                vx_fft.conj() * fx_fft
                + vy_fft.conj() * fy_fft
                + vz_fft.conj() * fz_fft
            )
            PK2_fft = abs(fx_fft) ** 2 + abs(fy_fft) ** 2 + abs(fz_fft) ** 2

            PKs = self._compute_means_regions(
                ifft(PK1_fft + PK2_fft * deltat / 2)
            )

        else:
            PKs = [0.0] * 3

        for i, path in enumerate(self.paths):
            with open(path, "a") as file:
                file.write(f"{tsim},{EKs[i]},{EKzs[i]},{EAs[i]},{epsKs[i]},{epsAs[i]},{PKs[i]},0.0\n")

    def load(self, iregion=0):
        df = pd.read_csv(self.paths[iregion], skiprows=1)
        return df

    def plot(self, keys="EK", iregion=0):
        df = self.load(iregion)
        df.plot(x="time", y=keys)
