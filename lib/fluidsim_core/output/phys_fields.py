"""Abstract base classes for Physical fields output

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
        ...

    @abstractmethod
    def get_min_time(self):
        ...


class PhysFieldsABC(metaclass=ABCMeta):
    _equation: Optional[str]
    _can_plot_quiver: Optional[bool]
    set_of_phys_files: SetOfPhysFieldFilesABC

    @abstractmethod
    def get_key_field_to_plot(self, forbid_compute=False, key_field_to_plot=None):
        ...

    @abstractmethod
    def get_field_to_plot(
        self,
        key=None,
        time=None,
        idx_time=None,
        equation=None,
        interpolate_time=True,
    ):
        """Get the field to be plotted in process 0."""
        ...

    @abstractmethod
    def get_vector_for_plot(self):
        ...

    @abstractmethod
    def _quiver_plot(self, ax, vecx="ux", vecy="uy", XX=None, YY=None, skip=None):
        """Superimposes a quiver plot of velocity vectors with a given axis
        object corresponding to a 2D contour plot.

        """
        ...

    @abstractmethod
    def _set_title(self, ax, key, time, vmax=None):
        ...

    @abstractmethod
    def _get_axis_data(equation=None):
        ...


class SetOfPhysFieldFilesBase(SetOfPhysFieldFilesABC):
    @staticmethod
    @abstractmethod
    def time_from_path(path):
        ...

    def __init__(self, path_dir=None, output=None):
        self.output = output
        if path_dir is None:
            if output is None:
                path_dir = Path.cwd()
            else:
                path_dir = Path(output.path_run)
        self.path_dir = Path(path_dir)
        self.update_times()

    def update_times(self):
        """Initialize the times by globing and analyzing the file names."""
        path_files = sorted(self.path_dir.glob("state_phys*.[hn]*"))

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
                idx_time=idx, key=key, equation=equation
            )

        if interpolate_time and time is not None:

            idx_closest, time_closest = self.get_closest_time_file(time)

            if math.isclose(time, time_closest) or self.times.size == 1:
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

        return self._get_field_to_plot_from_file(
            self.path_files[idx_time], key, equation
        )

    def get_closest_time_file(self, time):
        """Find the index and value of the closest actual time of the field."""
        idx = np.abs(self.times - time).argmin()
        return idx, self.times[idx]

    @abstractmethod
    def _get_field_to_plot_from_file(self, path_file, key, equation):
        ...
