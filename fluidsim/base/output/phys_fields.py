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
from glob import glob
from pathlib import Path
from math import isclose

import numpy as np
import h5py

from fluiddyn.util import mpi

from fluidsim.util.output import save_file, h5pack, ext

from .base import SpecificOutput


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

        self.key_field_to_plot = self.get_key_field_to_plot()

        self.set_of_phys_files = SetOfPhysFieldFiles(output=self.output)
        self._equation = None

        if self.period_save == 0 and self.period_plot == 0:
            self.t_last_save = -np.inf
            return

        self.t_last_save = self.sim.time_stepping.t
        self.t_last_plot = self.sim.time_stepping.t

    def get_key_field_to_plot(self, forbid_compute=False, key_field_to_plot=None):
        params = self.params
        if key_field_to_plot is None:
            key_field_to_plot = params.output.phys_fields.field_to_plot
        info_state = self.sim.info.solver.classes.State
        keys_ok = info_state.keys_state_phys.copy()

        if not params.ONLY_COARSE_OPER and not forbid_compute:
            keys_ok.extend(info_state.keys_computable)

        if key_field_to_plot not in keys_ok:
            key_field_to_plot = info_state.keys_state_phys[0]
            for key in ["q", "rot", "rotz"]:
                if key in keys_ok:
                    key_field_to_plot = key

        return key_field_to_plot

    def _init_path_files(self):
        super()._init_path_files()

        # This if clause is required since the function _init_path_files is
        # first called by the super().__init__ function when set_of_phys_files
        # is not initialized (see above). Useful when end_of_simul is called.
        if hasattr(self, "set_of_phys_files"):
            self.set_of_phys_files.path_dir = self.output.path_run

    def _init_files(self, arrays_1st_time=None):
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

        path_run = Path(self.output.path_run)

        if params.time_stepping.USE_T_END:
            # check if some state_phys files already exist
            try:
                path_test = next(path_run.glob("state_phys*"))
            except StopIteration:
                # file does not exist : get str_width from t_end
                # max number of digits = int(log10(t_end)) + 1
                # add .3f precision = 4 additional characters
                # +2 by anticipation of potential restarts
                str_width = int(np.log10(params.time_stepping.t_end)) + 7
            else:
                # file does exist : get str_width from file name
                # file name is something like 'state_phys_tYYYY.YYY.nc'
                str_width = len(path_test.name[12:-3])
        else:
            # dynamic width not implemented if USE_T_END==False
            str_width = 7

        if mpi.rank == 0:
            path_run.mkdir(exist_ok=True)

        if 0 < self.period_save < 0.001 or params.output.phys_fields.file_with_it:
            str_it = f"_it={self.sim.time_stepping.it}"
        else:
            str_it = ""

        name_save = f"state_phys_t{time:0{str_width}.3f}{str_it}.{ext}"

        path_file = path_run / name_save

        does_path_exist = None
        if mpi.rank == 0:
            does_path_exist = path_file.exists()
        if mpi.nb_proc > 1:
            does_path_exist = mpi.comm.bcast(does_path_exist, root=0)

        if does_path_exist:
            # do not save if the file corresponds to the same it
            it_file = None
            if mpi.rank == 0:
                with h5pack.File(str(path_file), "r") as file:
                    it_file = file["state_phys"].attrs["it"]
            if mpi.nb_proc > 1:
                it_file = mpi.comm.bcast(it_file, root=0)
            if it_file == self.sim.time_stepping.it:
                return
            name_save = (
                f"state_phys_t{time:07.3f}_it={self.sim.time_stepping.it}.{ext}"
            )
            path_file = path_run / name_save
        self.output.print_stdout("save state_phys in file " + name_save)

        save_file(
            path_file,
            state_phys,
            self.sim.info,
            self.output.name_run,
            self.sim.oper,
            time,
            self.sim.time_stepping.it,
            particular_attr,
        )

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
            and (time is None or isclose(time, self.sim.time_stepping.t))
            and idx_time is None
        ):
            # we get the field from the state
            field, key = self.get_field_to_plot_from_state(
                field=key, equation=equation
            )
            return field, self.sim.time_stepping.t

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
            key_field = self.key_field_to_plot
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
    """A set of physical field files."""

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

    def get_min_time(self):
        if hasattr(self, "times"):
            return self.times.min()
        return 0.0

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

            idx_closest, time_closest = self.get_closest_time_file(time)

            if isclose(time, time_closest) or self.times.size == 1:
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

            field0, time0 = self.get_field_to_plot(
                idx_time=idx0, key=key, equation=equation
            )
            field1, time1 = self.get_field_to_plot(
                idx_time=idx1, key=key, equation=equation
            )

            return field0 * weight0 + field1 * weight1, time

        # print(idx_time, 'Using file', self.path_files[idx_time])

        with h5py.File(self.path_files[idx_time], "r") as file:
            time = file["state_phys"].attrs["time"]
            dset = file["state_phys"][key]

            if equation is None:
                return dset[...], time

            if equation.startswith("iz="):
                iz = eval(equation[len("iz=") :])
                return dset[iz, ...], time

            elif equation.startswith("z="):
                z = eval(equation[len("z=") :])
                iz = abs(self.output.sim.oper.get_grid1d_seq("z") - z).argmin()
                return dset[iz, ...], time

            elif equation.startswith("iy="):
                iy = eval(equation[len("iy=") :])
                return dset[:, iy, :], time

            elif equation.startswith("y="):
                y = eval(equation[len("y=") :])
                iy = abs(self.output.sim.oper.get_grid1d_seq("y") - y).argmin()
                return dset[:, iy, :], time

            elif equation.startswith("ix="):
                ix = eval(equation[len("ix=") :])
                return dset[..., ix], time

            elif equation.startswith("x="):
                x = eval(equation[len("x=") :])
                ix = abs(self.output.sim.oper.get_grid1d_seq("x") - x).argmin()
                return dset[..., ix], time

            else:
                raise NotImplementedError

    def get_closest_time_file(self, time):
        """Find the index and value of the closest actual time of the field."""
        idx = np.abs(self.times - time).argmin()
        return idx, self.times[idx]
