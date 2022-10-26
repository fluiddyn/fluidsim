"""Spatial means
================

Provides:

.. autoclass:: SpatialMeansBase
   :members:
   :private-members:
   :noindex:
   :undoc-members:

.. autoclass:: SpatialMeansJSON
   :members:
   :private-members:
   :noindex:
   :undoc-members:

"""

import os
import numpy as np
import json
from typing import Dict

import pandas as pd
import xarray as xr
from fluiddyn.util import mpi

from .base import SpecificOutput


def inner_prod(a_fft, b_fft):
    return np.real(a_fft.conj() * b_fft)


class SpatialMeansBase(SpecificOutput):
    """A :class:`SpatialMean` object handles the saving of .

    This class uses the particular functions defined by some solvers
    :func:`` and
    :func``. If the solver doesn't has these
    functions, this class does nothing.
    """

    _tag = "spatial_means"
    _name_file = _tag + ".txt"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "spatial_means"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag, attribs={"HAS_TO_PLOT_SAVED": False})

    def __init__(self, output):
        params = output.sim.params

        self.sum_wavenumbers = output.sum_wavenumbers
        try:
            self.vecfft_from_rotfft = output.oper.vecfft_from_rotfft
        except AttributeError:
            pass

        super().__init__(
            output,
            period_save=params.output.periods_save.spatial_means,
            has_to_plot_saved=params.output.spatial_means.HAS_TO_PLOT_SAVED,
        )

        if self.period_save != 0:
            # saved for each initialization to help detecting bugs
            self._save_one_time()
            self.t_last_save = self.sim.time_stepping.t

    def _init_files(self, arrays_1st_time=None):

        if mpi.rank == 0:
            if not os.path.exists(self.path_file):
                self.file = open(self.path_file, "w")
            else:
                self.file = open(self.path_file, "r+")
                # to go to the end of the file
                self.file.seek(0, os.SEEK_END)

    def _online_save(self):
        self()

    def __call__(self):
        """Save the values at one time."""
        tsim = self.sim.time_stepping.t
        if (
            tsim + 1e-15
        ) // self.period_save > self.t_last_save // self.period_save:
            self.t_last_save = self.sim.time_stepping.t
            self._save_one_time()

    def _save_one_time(self, *args):
        self.t_last_save = self.sim.time_stepping.t

    def _init_online_plot(self):
        if mpi.rank == 0:
            width_axe = 0.85
            height_axe = 0.4
            x_left_axe = 0.12
            z_bottom_ax = 0.55

            size_ax = [x_left_axe, z_bottom_ax, width_axe, height_axe]
            fig, ax = self.output.figure_axe(size_axe=size_ax, numfig=3_000_000)
            self.ax_a = ax
            ax.set_xlabel("$t$")
            ax.set_ylabel("$E(t)$")
            ax.set_title("mean quantities\n" + self.output.summary_simul)

            z_bottom_ax = 0.08
            size_ax[1] = z_bottom_ax
            ax = fig.add_axes(size_ax)
            self.axe_b = ax
            ax.set_xlabel("$t$")
            ax.set_ylabel(r"$\epsilon(t)$")

    def _get_nb_points_from_lines(self, lines_t, *liness):
        nt = len(lines_t)
        liness = [lines for lines in liness if lines]
        if all(len(lines) == nt for lines in liness):
            return nt
        else:
            # the last line for a quantity may not have been saved yet
            return nt - 1

    def load(self):
        dict_results = {}
        return dict_results

    def load_dataset(self, dims=("t",)):
        """Loads results as a xarray dataset."""
        dict_results = self.load()
        # NOTE: format specified in
        # http://xarray.pydata.org/en/stable/generated/xarray.Dataset.from_dict.html
        dset = {"coords": {}, "attrs": {}, "dims": dims, "data_vars": {}}
        for key, value in dict_results.items():
            if isinstance(value, np.ndarray):
                target = "coords" if key in dims else "data_vars"
                dset[target].update({key: {"dims": dims, "data": value}})
            else:
                dset["attrs"].update({key: value})

        return xr.Dataset.from_dict(dset)

    def plot(self):
        pass

    def compute_time_means(self, tstatio=0.0, tmax=None):
        """compute the temporal means."""
        dict_results = self.load()
        times = dict_results["t"]
        imin_mean = np.argmin(abs(times - tstatio))
        imax_mean = None if tmax is None else np.argmin(abs(times - tmax)) + 1

        dict_time_means = {}
        for key, value in dict_results.items():
            if isinstance(value, np.ndarray):
                dict_time_means[key] = np.mean(value[imin_mean:imax_mean])
        return dict_time_means, dict_results

    def _close_file(self):
        try:
            self.file.close()
        except AttributeError:
            pass

    def time_first_saved(self) -> float:
        with open(self.path_file) as file_means:
            line = ""
            while not line.startswith("time ="):
                line = file_means.readline()

        words = line.split()
        return float(words[2])

    def time_last_saved(self) -> float:
        with open(self.path_file, "rb") as file_means:
            nb_char = file_means.seek(0, os.SEEK_END)  # go to the end
            nb_char_to_read = min(nb_char, 1000)
            file_means.seek(-nb_char_to_read, 2)
            line = file_means.readline()
            while line != b"":
                if line.startswith(b"time ="):
                    line_time = line
                line = file_means.readline()

        words = line_time.split()
        return float(words[2])


class SpatialMeansJSON(SpatialMeansBase):
    """Save and load as line-delimited JSON."""

    _tag = "spatial_means"
    _name_file = _tag + ".json"

    def _save_one_time(self, result: Dict[str, float], delimiter: str = "\n"):
        if mpi.rank == 0:
            json.dump(result, self.file)
            self.file.write(delimiter)

        super()._save_one_time()

    def _file_exists(self):
        return self.path_file.endswith(".json") and os.path.exists(self.path_file)

    def load(self):
        if not os.path.exists(self.path_file):
            raise FileNotFoundError(
                f"Spatial means file is missing: {self.path_file}"
            )

        try:
            df = pd.read_json(self.path_file, orient="records", lines=True)
            return df
        except ValueError:
            with open(self.path_file) as fp:
                for line_nb, line in enumerate(fp):
                    line_nb += 1
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        break

            raise IOError(
                f"Error reading spatial means file {self.path_file} in\n\tline"
                f"number {line_nb}:\n\t{line}"
            )

    def load_dataset(self, dims=("t",)):
        df = self.load()
        if isinstance(df, dict):
            return super().load_dataset(dims)
        else:
            df.index = df[dims[0]]

            return xr.Dataset.from_dataframe(df)

    def compute_time_means(self, tstatio=0.0, tmax=None):
        """compute the temporal means."""
        if not self._file_exists():
            return super().compute_time_means(tstatio, tmax)

        df = self.load()
        times = df.t
        imin_mean = abs(times - tstatio).idxmin()
        imax_mean = None if tmax is None else (abs(times - tmax)).idxmin() + 1

        df_mean = df.iloc[imin_mean:imax_mean].mean()
        return df_mean, df

    def time_first_saved(self) -> float:
        if not self._file_exists():
            self.path_file = self.path_file.replace(".json", ".txt")
            return super().time_first_saved()

        with open(self.path_file) as file_means:
            line = file_means.readline()

        result = json.loads(line)
        return result["t"]

    def time_last_saved(self) -> float:
        if not self._file_exists():
            self.path_file = self.path_file.replace(".json", ".txt")
            return super().time_last_saved()

        with open(self.path_file, "rb") as file_means:
            nb_char = file_means.seek(0, os.SEEK_END)  # go to the end
            nb_char_to_read = min(nb_char, 1000)
            file_means.seek(-nb_char_to_read, 2)
            line = file_means.readline()
            while line != b"":
                line_prev = line
                line = file_means.readline()

        result = json.loads(line_prev)
        return result["t"]
