"""Abstract base classes for Physical fields output
---------------------------------------------------

These classes can be used to define classes working together with
:class:`fluidsim_core.output.movies.MoviesBasePhysFields`.

.. autoclass:: PhysFieldsABC
   :members:
   :private-members:
   :undoc-members:

.. autoclass:: SetOfPhysFieldFilesABC
   :members:
   :private-members:
   :undoc-members:

"""

from abc import ABCMeta, abstractmethod
from typing import Optional
from pathlib import Path
import math

import numpy as np


class SetOfPhysFieldFilesABC(metaclass=ABCMeta):
    times: list

    @abstractmethod
    def update_times(self):
        "Update the times of the files"

    @abstractmethod
    def get_min_time(self):
        "Get minimum time"


class PhysFieldsABC(metaclass=ABCMeta):
    _equation: Optional[str]
    _can_plot_quiver: Optional[bool]
    set_of_phys_files: SetOfPhysFieldFilesABC

    @abstractmethod
    def get_key_field_to_plot(self, forbid_compute=False, key_prefered=None):
        "Get the key corresponding to the field to be plotted"

    @abstractmethod
    def get_field_to_plot(
        self,
        key=None,
        time=None,
        idx_time=None,
        equation=None,
        interpolate_time=True,
        skip_vars=(),
    ):
        """Get the field to be plotted in process 0."""

    @abstractmethod
    def get_vector_for_plot(
        self, from_state=False, time=None, interpolate_time=True
    ):
        "Get the vector components"

    def _set_title(self, ax, key, time, vmax=None):
        title = f"{key}, $t = {time:.3f}$"
        if vmax is not None:
            title += r", $|\vec{v}|_{max} = $" + f"{vmax:.3f}"
        ax.set_title(title)

    @abstractmethod
    def _get_axis_data(equation=None):
        "Get x and y axes data"

    def set_equation_crosssection(self, equation):
        """Set the equation defining the cross-section.

        Parameters
        ----------

        equation : str

          The equation can be of the shape 'iz=2', 'z=1', ...

        """
        self._equation = equation

    def _init_skip_quiver(self):
        params = self.output.sim.params
        Lx = params.oper.Lx
        Ly = params.oper.Ly
        x, y = self._get_axis_data()
        # 4% of the Lx it is a great separation between vector arrows.
        try:
            delta_quiver = 0.04 * min(Lx, Ly)
        except AttributeError:
            skip = 1
        else:
            skip = (len(x) / Lx) * delta_quiver
            skip = int(np.round(skip))
            if skip < 1:
                skip = 1
        self._skip_quiver = skip
        return skip


class SetOfPhysFieldFilesBase(SetOfPhysFieldFilesABC):
    @staticmethod
    @abstractmethod
    def time_from_path(path):
        "Get the time corresponding to a file"

    def __init__(self, path_dir=None, output=None):
        self.output = output
        if path_dir is None:
            if output is None:
                path_dir = Path.cwd()
            else:
                path_dir = Path(output.path_run)
        self.path_dir = Path(path_dir)
        self._glob_pattern = self._get_glob_pattern()
        self.update_times()

    def _get_glob_pattern(self):
        return "state_phys*.[hn]*"

    def update_times(self):
        """Initialize the times by globing and analyzing the file names."""
        path_files = sorted(self.path_dir.glob(self._glob_pattern))

        if hasattr(self, "path_files") and len(self.path_files) == len(
            path_files
        ):
            return

        self.path_files = sorted(path_files)
        self.times = np.array(
            [self.time_from_path(path) for path in self.path_files]
        )

    def get_min_time(self):
        if hasattr(self, "times"):
            return self.times.min()
        return 0.0

    def get_max_time(self):
        return self.times.max()

    def get_field_to_plot(
        self,
        time=None,
        idx_time=None,
        key=None,
        equation=None,
        interpolate_time=True,
        skip_vars=(),
    ):

        if time is None and idx_time is None:
            self.update_times()
            time = self.times[-1]

        # Assert files are available
        if self.times.size == 0:
            raise FileNotFoundError(
                f"No phys files were detected in directory: {self.path_dir}"
            )

        if not interpolate_time and time is not None:
            idx, time_closest = self.get_closest_time_file(time)
            return self.get_field_to_plot(
                idx_time=idx, key=key, equation=equation, skip_vars=skip_vars
            )

        if interpolate_time and time is not None:

            idx_closest, time_closest = self.get_closest_time_file(time)

            if math.isclose(time, time_closest) or self.times.size == 1:
                return self.get_field_to_plot(
                    idx_time=idx_closest,
                    key=key,
                    equation=equation,
                    skip_vars=skip_vars,
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
                idx_time=idx0, key=key, equation=equation, skip_vars=skip_vars
            )
            field1, time1 = self.get_field_to_plot(
                idx_time=idx1, key=key, equation=equation, skip_vars=skip_vars
            )

            return field0 * weight0 + field1 * weight1, time

        return self._get_field_to_plot_from_file(
            self.path_files[idx_time], key, equation=equation, skip_vars=skip_vars
        )

    def get_closest_time_file(self, time):
        """Find the index and value of the closest actual time of the field."""
        idx = np.abs(self.times - time).argmin()
        return idx, self.times[idx]

    @abstractmethod
    def _get_field_to_plot_from_file(
        self, path_file, key, equation, skip_vars=()
    ):
        "Get a 2d field from a file"
