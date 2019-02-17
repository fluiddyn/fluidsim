"""Physical fields output (:mod:`fluidsim.base.output.phys_fields`)
===================================================================

Provides:

.. autoclass:: PhysFieldsBase
   :members:
   :private-members:

.. autoclass:: SetOfPhysFieldFiles
   :members:
   :private-members:

"""

import re
import os
import datetime
from glob import glob

import numpy as np
import h5py
import h5netcdf

from fluiddyn.util import mpi
from .base import SpecificOutput

cfg_h5py = h5py.h5.get_config()

if cfg_h5py.mpi:
    ext = "h5"
    h5pack = h5py
else:
    ext = "nc"
    h5pack = h5netcdf


def _create_variable(group, key, field):
    if ext == "nc":
        if field.ndim == 0:
            dimensions = tuple()
        elif field.ndim == 1:
            dimensions = ("x",)
        elif field.ndim == 2:
            dimensions = ("y", "x")
        elif field.ndim == 3:
            dimensions = ("z", "y", "x")
        try:
            group.create_variable(key, data=field, dimensions=dimensions)
        except AttributeError:
            raise ValueError(
                "Error while creating a netCDF4 variable using group"
                " of type {} for key {}".format(type(group), key)
            )

    else:
        try:
            group.create_dataset(key, data=field)
        except AttributeError:
            raise ValueError(
                "Error while creating a HDF5 dataset using group"
                " of type {} for key {}".format(type(group), key)
            )


class PhysFieldsBase(SpecificOutput):
    """Manage the output of physical fields."""

    _tag = "phys_fields"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "phys_fields"
        params.output._set_child(
            tag, attribs={"field_to_plot": "ux", "file_with_it": False}
        )

        params.output.periods_save._set_attrib(tag, 0)
        params.output.periods_plot._set_attrib(tag, 0)

    def __init__(self, output):
        params = output.sim.params
        self.output = output
        self.oper = output.oper

        if hasattr(self, "_init_skip_quiver"):
            self._init_skip_quiver()

        if hasattr(self, "_init_movies"):
            self._init_movies()
            self.animate = self.movies.animate
            self.interact = self.movies.interact

        super().__init__(
            output,
            period_save=params.output.periods_save.phys_fields,
            period_plot=params.output.periods_plot.phys_fields,
        )

        self.field_to_plot = params.output.phys_fields.field_to_plot

        self.set_of_phys_files = SetOfPhysFieldFiles(output=self.output)
        self._equation = None

        if self.period_save == 0 and self.period_plot == 0:
            self.t_last_save = -np.inf
            return

        self.t_last_save = self.sim.time_stepping.t
        self.t_last_plot = self.sim.time_stepping.t

    def _init_path_files(self):
        super()._init_path_files()

        # This if clause is required since the function _init_path_files is
        # first called by the super().__init__ function when set_of_phys_files
        # is not initialized (see above). Useful when end_of_simul is called.
        if hasattr(self, "set_of_phys_files"):
            self.set_of_phys_files.path_dir = self.output.path_run

    def _init_files(self, dict_arrays_1time=None):
        # Does nothing on purpose...
        pass

    def _init_online_plot(self):
        pass

    def _online_save(self):
        """Online save."""
        tsim = self.sim.time_stepping.t
        if self._has_to_online_save():
            self.t_last_save = tsim
            self.save()

    def _online_plot(self):
        """Online plot."""
        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_plot >= self.period_plot:
            self.t_last_plot = tsim
            itsim = self.sim.time_stepping.it
            self.plot(
                numfig=itsim,
                key_field=self.params.output.phys_fields.field_to_plot,
            )

    def save(self, state_phys=None, params=None, particular_attr=None):
        if state_phys is None:
            state_phys = self.sim.state.state_phys
        if params is None:
            params = self.params

        time = self.sim.time_stepping.t

        path_run = self.output.path_run

        if mpi.rank == 0 and not os.path.exists(path_run):
            os.mkdir(path_run)

        if (
            self.period_save < 0.001
            or self.params.output.phys_fields.file_with_it
        ):
            name_save = "state_phys_t{:07.3f}_it={}.{}".format(
                time, self.sim.time_stepping.it, ext
            )
        else:
            name_save = f"state_phys_t{time:07.3f}.{ext}"

        path_file = os.path.join(path_run, name_save)
        if os.path.exists(path_file):
            name_save = "state_phys_t{:07.3f}_it={}.{}".format(
                time, self.sim.time_stepping.it, ext
            )
            path_file = os.path.join(path_run, name_save)
        to_print = "save state_phys in file " + name_save
        self.output.print_stdout(to_print)

        # FIXME: bad condition below when run sequentially, with MPI enabled h5py
        # As a workaround the instantiation is made with h5pack
        if mpi.nb_proc == 1 or not cfg_h5py.mpi:
            if mpi.rank == 0:
                # originally:
                # h5file = h5netcdf.File(...
                h5file = h5pack.File(path_file, "w")
                group_state_phys = h5file.create_group("state_phys")
                group_state_phys.attrs["what"] = "obj state_phys for fluidsim"
                group_state_phys.attrs["name_type_variables"] = state_phys.info
                group_state_phys.attrs["time"] = time
                group_state_phys.attrs["it"] = self.sim.time_stepping.it
        else:
            # originally:
            # h5file = h5py.File(...
            h5file = h5pack.File(path_file, "w", driver="mpio", comm=mpi.comm)
            group_state_phys = h5file.create_group("state_phys")
            group_state_phys.attrs["what"] = "obj state_phys for fluidsim"
            group_state_phys.attrs["name_type_variables"] = state_phys.info

            group_state_phys.attrs["time"] = time
            group_state_phys.attrs["it"] = self.sim.time_stepping.it

        if mpi.nb_proc == 1:
            for k in state_phys.keys:
                field_seq = state_phys.get_var(k)
                _create_variable(group_state_phys, k, field_seq)
        elif not cfg_h5py.mpi:
            for k in state_phys.keys:
                field_loc = state_phys.get_var(k)
                field_seq = self.oper.gather_Xspace(field_loc)
                if mpi.rank == 0:
                    _create_variable(group_state_phys, k, field_seq)
        else:
            h5file.atomic = False
            ndim = len(self.oper.shapeX_loc)
            if ndim == 2:
                xstart, ystart = self.oper.seq_indices_first_X
            elif ndim == 3:
                xstart, ystart, zstart = self.oper.seq_indices_first_X
            else:
                raise NotImplementedError
            xend = xstart + self.oper.shapeX_loc[0]
            yend = ystart + self.oper.shapeX_loc[1]
            for k in state_phys.keys:
                field_loc = state_phys.get_var(k)
                dset = group_state_phys.create_dataset(
                    k, self.oper.shapeX_seq, dtype=field_loc.dtype
                )
                with dset.collective:
                    if field_loc.ndim == 2:
                        dset[xstart:xend, ystart:yend] = field_loc
                    elif field_loc.ndim == 3:
                        dset[xstart:xend, ystart:yend, :] = field_loc
                    else:
                        raise NotImplementedError(
                            "Unsupported number of dimensions"
                        )
            h5file.close()
            if mpi.rank == 0:
                h5file = h5pack.File(path_file, "r+")

        if mpi.rank == 0:
            h5file.attrs["date saving"] = str(datetime.datetime.now()).encode()
            h5file.attrs["name_solver"] = self.output.name_solver
            h5file.attrs["name_run"] = self.output.name_run
            h5file.attrs["axes"] = np.array(self.oper.axes, dtype="|S9")
            if particular_attr is not None:
                h5file.attrs["particular_attr"] = particular_attr

            self.sim.info._save_as_hdf5(hdf5_parent=h5file)
            gp_info = h5file["info_simul"]
            gf_params = gp_info["params"]
            gf_params.attrs["SAVE"] = 1
            gf_params.attrs["NEW_DIR_RESULTS"] = 1
            h5file.close()

    def get_field_to_plot(
        self,
        key=None,
        time=None,
        idx_time=None,
        equation=None,
        interpolate_time=True,
    ):
        """Get the field to be plotted in process 0."""

        if equation is None:
            equation = self._equation

        if (
            not self.sim.params.ONLY_COARSE_OPER
            and (time is None or time == self.sim.time_stepping.t)
            and idx_time is None
        ):
            # we get the field from the state
            field, key = self.get_field_to_plot_from_state(
                field=key, equation=equation
            )
            return field

        else:
            return self.set_of_phys_files.get_field_to_plot(
                time=time,
                idx_time=idx_time,
                key=key,
                equation=equation,
                interpolate_time=interpolate_time,
            )

    def get_field_to_plot_from_state(self, field=None, equation=None):
        """Get the field to be plotted in process 0."""

        if field is None:
            keys_state_phys = self.sim.info.solver.classes.State[
                "keys_state_phys"
            ]
            keys_computable = self.sim.info.solver.classes.State[
                "keys_computable"
            ]
            field_to_plot = self.params.output.phys_fields.field_to_plot
            if (
                field_to_plot in keys_state_phys
                or field_to_plot in keys_computable
            ):
                key_field = field_to_plot
            else:
                if "q" in keys_state_phys:
                    key_field = "q"
                elif "rot" in keys_state_phys:
                    key_field = "rot"
                else:
                    key_field = keys_state_phys[0]
            field_loc = self.sim.state.get_var(key_field)
        elif isinstance(field, np.ndarray):
            key_field = "given field"
            field_loc = field
        else:
            field_loc = self.sim.state.get_var(field)
            key_field = field

        if mpi.nb_proc > 1:
            field = self.oper.gather_Xspace(field_loc)
        else:
            field = field_loc

        if equation is None:
            return field, key_field

        elif equation.startswith("iz="):
            iz = eval(equation[len("iz=") :])
            field = field[iz, ...]
        elif equation.startswith("z="):
            z = eval(equation[len("z=") :])
            iz = abs(self.output.sim.oper.get_grid1d_seq("z") - z).argmin()
            field = field[iz, ...]
        elif equation.startswith("iy="):
            iy = eval(equation[len("iy=") :])
            field = field[:, iy, :]
        elif equation.startswith("y="):
            y = eval(equation[len("y=") :])
            iy = abs(self.output.sim.oper.get_grid1d_seq("y") - y).argmin()
            field = field[:, iy, :]
        elif equation.startswith("ix="):
            ix = eval(equation[len("ix=") :])
            field = field[..., ix]
        elif equation.startswith("x="):
            x = eval(equation[len("x=") :])
            ix = abs(self.output.sim.oper.get_grid1d_seq("x") - x).argmin()
            field = field[..., ix]
        else:
            raise NotImplementedError

        return field, key_field


def time_from_path(path):
    """Regular expression search to extract time from filename."""
    filename = os.path.basename(path)
    pattern = r"""
        (?!t)     # text after t but exclude it
        [0-9]+    # a couple of digits
        \.        # the decimal point
        [0-9]+    # a couple of digits
    """
    match = re.search(pattern, filename, re.VERBOSE)
    time = float(match.group(0))
    return time


class SetOfPhysFieldFiles:
    """A set of physical field files.

    """

    def __init__(self, path_dir=os.curdir, output=None):
        self.output = output
        self.path_dir = path_dir if output is None else output.path_run
        self.update_times()

    def update_times(self):
        """Initialize the times by globing and analyzing the file names."""
        path_files = glob(os.path.join(self.path_dir, "state_phys*.[hn]*"))

        if hasattr(self, "path_files") and len(self.path_files) == len(
            path_files
        ):
            return

        self.path_files = sorted(path_files)
        self.times = np.array([time_from_path(path) for path in self.path_files])

    def get_field_to_plot(
        self,
        time=None,
        idx_time=None,
        key=None,
        equation=None,
        interpolate_time=True,
    ):

        if time is None and idx_time is None:
            self.update_times()
            time = self.times[-1]

        # Assert files are available
        if self.times.size == 0:
            raise FileNotFoundError(
                "No state_phys files were detected in directory: "
                f"{self.path_dir}"
            )

        if not interpolate_time and time is not None:
            idx, time_closest = self.get_closest_time_file(time)
            return self.get_field_to_plot(
                idx_time=idx, key=key, equation=equation
            )

        if interpolate_time and time is not None:
            if self.times.min() > time > self.times.max():
                raise ValueError()

            idx_closest, time_closest = self.get_closest_time_file(time)

            if time == time_closest:
                return self.get_field_to_plot(
                    idx_time=idx_closest, key=key, equation=equation
                )

            if idx_closest == self.times.size - 1:
                idx0 = idx_closest - 1
                idx1 = idx_closest
            elif time_closest < time or idx_closest == 0:
                idx0 = idx_closest
                idx1 = idx_closest + 1
            elif time_closest > time:
                idx0 = idx_closest - 1
                idx1 = idx_closest

            dt_save = self.times[idx1] - self.times[idx0]
            weight0 = 1 - np.abs(time - self.times[idx0]) / dt_save
            weight1 = 1 - np.abs(time - self.times[idx1]) / dt_save

            field0 = self.get_field_to_plot(
                idx_time=idx0, key=key, equation=equation
            )
            field1 = self.get_field_to_plot(
                idx_time=idx1, key=key, equation=equation
            )

            return field0 * weight0 + field1 * weight1

        # print(idx_time, 'Using file', self.path_files[idx_time])

        with h5py.File(self.path_files[idx_time]) as f:
            dset = f["state_phys"][key]

            if equation is None:
                return dset.value

            if equation.startswith("iz="):
                iz = eval(equation[len("iz=") :])
                return dset[iz, ...]

            elif equation.startswith("z="):
                z = eval(equation[len("z=") :])
                iz = abs(self.output.sim.oper.get_grid1d_seq("z") - z).argmin()
                return dset[iz, ...]

            elif equation.startswith("iy="):
                iy = eval(equation[len("iy=") :])
                return dset[:, iy, :]

            elif equation.startswith("y="):
                y = eval(equation[len("y=") :])
                iy = abs(self.output.sim.oper.get_grid1d_seq("y") - y).argmin()
                return dset[:, iy, :]

            elif equation.startswith("ix="):
                ix = eval(equation[len("ix=") :])
                return dset[..., ix]

            elif equation.startswith("x="):
                x = eval(equation[len("x=") :])
                ix = abs(self.output.sim.oper.get_grid1d_seq("x") - x).argmin()
                return dset[..., ix]

            else:
                raise NotImplementedError

    def get_closest_time_file(self, time):
        """Find the index and value of the closest actual time of the field."""
        idx = np.abs(self.times - time).argmin()
        return idx, self.times[idx]
