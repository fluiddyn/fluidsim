"""Horizontal means
===================

Provides:

.. autoclass:: HorizontalMeans
   :members:
   :private-members:
   :noindex:
   :undoc-members:

"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

from fluiddyn.util import mpi

from fluidsim.extend_simul import SimulExtender, extend_simul_class

from .base import SpecificOutput


__all__ = ["extend_simul_class", "HorizontalMeans"]


class HorizontalMeans(SpecificOutput, SimulExtender):
    """Horizontal means as functions of the z coordinate

    Examples
    --------

    This is an output class to compute/save/load/plot horizontally averaged
    quantities. Since this is useful only for particular simulations, it is a
    ``SimulExtender``, meaning that it should be used this way:

    .. code-block:: python

        from fluidsim.solvers.ns3d.solver import Simul
        from fluidsim.base.output.horiz_means import HorizontalMeans, extend_simul_class

        Simul = extend_simul_class(Simul, HorizontalMeans)

        params = Simul.create_default_params()

        ...

        params.output.periods_save.horiz_means = 0.1

        sim = Simul(params)

    Then, during or after the simulation:

    .. code-block:: python

        sim.output.horiz_means.plot(tmin=10)
        data = sim.output.horiz_means.load(tmin=10)

    """

    _tag = "horiz_means"
    _module_name = "fluidsim.base.output.horiz_means"
    _name_file = _tag + ".h5"

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
                cls._tag,
                attribs={
                    "module_name": cls._module_name,
                    "class_name": cls.__name__,
                },
            )

        return modif_info_solver

    @classmethod
    def _complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)

    def _compute_hmean(self, arr3d):
        if mpi.nb_proc == 1:
            return np.mean(arr3d, axis=(1, 2))

        # here (MPI case) it gets a bit more complicated!
        sum_local = np.sum(arr3d, axis=(1, 2))
        if mpi.rank == 0:
            sum_global = np.zeros(self.nz)

        # MPI messages and summation
        for rank in range(mpi.nb_proc):
            if mpi.rank == 0:
                nz_loc = self.nzs_local[rank]

            if rank == 0 and mpi.rank == 0:
                data = sum_local
            else:
                if mpi.rank == 0:
                    data = np.empty(nz_loc)
                    mpi.comm.Recv(data, source=rank, tag=42 * rank)
                elif mpi.rank == rank:
                    mpi.comm.Send(sum_local, dest=0, tag=42 * rank)

            if mpi.rank == 0:
                iz_start = self.izs_start[rank]
                sum_global[iz_start : iz_start + nz_loc] += data

        if mpi.rank == 0:
            return sum_global / self.nh

    def _hmean_to_3d_local(self, profile):
        if mpi.nb_proc != 1:
            profile = mpi.comm.bcast(profile, root=0)
            profile = profile[self.iz_start : self.iz_start + self.nz_local]

        return np.broadcast_to(
            profile[:, np.newaxis, np.newaxis], self.shapeX_loc
        )

    def __init__(self, output):
        self.output = output
        sim = output.sim
        params = sim.params

        self.shapeX_loc = sim.oper.shapeX_loc

        if mpi.nb_proc > 1:
            self.iz_start, _, _ = sim.oper.oper_fft.get_seq_indices_first_X()
            self.izs_start = mpi.comm.gather(self.iz_start, root=0)

            nh_local = self.shapeX_loc[1] * self.shapeX_loc[2]
            self.nhs_local = mpi.comm.gather(nh_local, root=0)

            self.nz_local = self.shapeX_loc[0]
            self.nzs_local = mpi.comm.gather(self.nz_local, root=0)

            if mpi.rank == 0:
                assert len(self.izs_start) == mpi.nb_proc
                assert len(self.nhs_local) == mpi.nb_proc
                assert len(self.nzs_local) == mpi.nb_proc

                print(self.izs_start, self.nhs_local, self.nzs_local)

        self.nz = params.oper.nz
        self.nh = params.oper.nx * params.oper.ny
        dz = params.oper.Lz / self.nz
        z = dz * np.arange(self.nz)

        super().__init__(
            output,
            period_save=params.output.periods_save.horiz_means,
            arrays_1st_time={"z": z},
        )

    def compute(self):
        data = {}
        get_var = self.sim.state.state_phys.get_var
        vx = get_var("vx")
        vy = get_var("vy")
        vz = get_var("vz")

        def _extend_data_with_hmean(arr3d, name: str):
            result = data[name] = self._compute_hmean(arr3d)
            return result

        vx_mean = _extend_data_with_hmean(vx, "vx")
        vy_mean = _extend_data_with_hmean(vy, "vy")
        vz_mean = _extend_data_with_hmean(vz, "vz")

        vx_mean3d = self._hmean_to_3d_local(vx_mean)
        vy_mean3d = self._hmean_to_3d_local(vy_mean)
        vz_mean3d = self._hmean_to_3d_local(vz_mean)

        vxp = vx - vx_mean3d
        vyp = vy - vy_mean3d
        vzp = vz - vz_mean3d

        _extend_data_with_hmean(vxp**2, "vxp_vxp")
        _extend_data_with_hmean(vyp**2, "vyp_vyp")
        _extend_data_with_hmean(vzp**2, "vzp_vzp")

        _extend_data_with_hmean(vyp * vxp, "vyp_vxp")
        _extend_data_with_hmean(vzp * vxp, "vzp_vxp")
        _extend_data_with_hmean(vzp * vyp, "vzp_vyp")

        return data

    def load(self, tmin=None, tmax=None, verbose=False):
        with h5py.File(self.path_file, "r") as file:
            dset_times = file["times"]
            times = dset_times[...]
            nt = len(times)

            z = file["z"][...]

            if tmin is None:
                imin_plot = 0
            else:
                imin_plot = np.argmin(abs(times - tmin))

            if tmax is None:
                imax_plot = nt - 1
            else:
                imax_plot = np.argmin(abs(times - tmax))

            tmin = times[imin_plot]
            tmax = times[imax_plot]

            if verbose:
                print(
                    "compute time average with\n"
                    + (
                        f"tmin = {tmin:8.6g} ; tmax = {tmax:8.6g}"
                        f"imin = {imin_plot:8d} ; imax = {imax_plot:8d}"
                    )
                )

            data = {"z": z}
            for key in list(file.keys()):
                if key.startswith("v"):
                    dset_key = file[key]
                    profile = dset_key[imin_plot : imax_plot + 1].mean(0)
                    data[key] = profile
        return data

    def plot(self, tmin=None, tmax=None, verbose=False):
        data = self.load(tmin, tmax, verbose)
        z = data.pop("z")

        fig, (ax_left, ax_right) = plt.subplots(ncols=2, sharey=True)

        for letter in "xyz":
            arr = data.pop("v" + letter)
            ax_left.plot(arr, z, label=f"$<v_{letter}>_{{xy}}$")

            arr = data.pop(f"v{letter}p_v{letter}p")
            ax_right.plot(arr, z, label=f"$<v_{letter}' v_{letter}'>_{{xy}}$")

        ax_left.legend()
        ax_left.set_ylabel("$z$")

        for l0, l1 in ("yx", "zx", "zy"):
            name = f"v{l0}p_v{l1}p"
            ax_right.plot(
                data[name], z, "--", label=f"$<v_{l0}' v_{l1}'>_{{xy}}$"
            )

        ax_right.legend()

        fig.tight_layout()
