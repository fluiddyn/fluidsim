import os

import numpy as np
import h5py

from fluiddyn.util import mpi

from .base import SpecificOutput
from .movies import MoviesBase1D


class MoviesSpectra(MoviesBase1D):
    def __init__(self, output, spectra):
        self.spectra = spectra
        super().__init__(output)

    def init_animation(self, *args, **kwargs):
        if "xmax" not in kwargs:
            kwargs["xmax"] = self.oper.khE[-1:][0]
        if "ymax" not in kwargs:
            kwargs["ymax"] = 1.0

        with h5py.File(self.spectra.path_file2D) as f:
            self.times = f["times"][...]

        super().init_animation(*args, **kwargs)

    def get_field_to_plot(self, time, key=None):
        if key is None:
            key = self.key_field
        idx, t_file = self.get_closest_time_file(time)
        with h5py.File(self.spectra.path_file2D) as f:
            y = f["spectrum2D_" + key][idx]
        y[abs(y) < 10e-16] = 0
        return y

    def get_closest_time_file(self, time):
        """Find the index and value of the closest actual time of the field."""
        idx = np.abs(self.times - time).argmin()
        return idx, self.times[idx]

    def _init_labels(self, xlabel="x"):
        """Initialize the labels."""
        self.ax.set_xlabel(xlabel, fontdict=self.font)
        self.ax.set_ylabel(self.key_field, fontdict=self.font)
        self.ax.set_yscale("log")

    def _get_axis_data(self):
        """Get axis data.

        Returns
        -------

        x : array
          x-axis data.

        """
        with h5py.File(self.spectra.path_file2D) as f:
            x = f["khE"][...]

        return x

    def _set_key_field(self, key_field):
        """
        Defines key_field default.
        """
        if key_field is None:
            key_field = "E"
        self.key_field = key_field


class Spectra(SpecificOutput):
    """Used for the saving of spectra."""

    _tag = "spectra"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "spectra"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag, attribs={"HAS_TO_PLOT_SAVED": False})

    def _init_movies(self):
        self.movies = MoviesSpectra(self.output, self)

    def __init__(self, output):
        self.output = output

        if hasattr(self, "_init_movies"):
            self._init_movies()

        params = output.sim.params
        self.nx = int(params.oper.nx)

        self.spectrum2D_from_fft = output.sim.oper.spectrum2D_from_fft
        self.spectra1D_from_fft = output.sim.oper.spectra1D_from_fft

        super().__init__(
            output,
            period_save=params.output.periods_save.spectra,
            has_to_plot_saved=params.output.spectra.HAS_TO_PLOT_SAVED,
        )

    def _init_path_files(self):
        path_run = self.output.path_run
        self.path_file1D = path_run + "/spectra1D.h5"
        self.path_file2D = path_run + "/spectra2D.h5"

    def _init_files(self, dict_arrays_1time=None):
        dict_spectra1D, dict_spectra2D = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file1D):
                dict_arrays_1time = {
                    "kxE": self.sim.oper.kxE,
                    "kyE": self.sim.oper.kyE,
                }
                self._create_file_from_dict_arrays(
                    self.path_file1D, dict_spectra1D, dict_arrays_1time
                )
                dict_arrays_1time = {"khE": self.sim.oper.khE}
                self._create_file_from_dict_arrays(
                    self.path_file2D, dict_spectra2D, dict_arrays_1time
                )
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file1D, "r") as f:
                    dset_times = f["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                # save the spectra in the file spectra1D.h5
                self._add_dict_arrays_to_file(self.path_file1D, dict_spectra1D)
                # save the spectra in the file spectra2D.h5
                self._add_dict_arrays_to_file(self.path_file2D, dict_spectra2D)

        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Save the values at one time. """
        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_save >= self.period_save:
            self.t_last_save = tsim
            dict_spectra1D, dict_spectra2D = self.compute()
            if mpi.rank == 0:
                # save the spectra in the file spectra1D.h5
                self._add_dict_arrays_to_file(self.path_file1D, dict_spectra1D)
                # save the spectra in the file spectra2D.h5
                self._add_dict_arrays_to_file(self.path_file2D, dict_spectra2D)
                self.nb_saved_times += 1
                if self.has_to_plot:
                    self._online_plot_saving(dict_spectra1D, dict_spectra2D)

                    if tsim - self.t_last_show >= self.period_show:
                        self.t_last_show = tsim
                        self.axe.get_figure().canvas.draw()

    def compute(self):
        """compute the values at one time."""
        if mpi.rank == 0:
            dict_results = {}
            return dict_results

    def _init_online_plot(self):
        if mpi.rank == 0:
            fig, axe = self.output.figure_axe(numfig=1_000_000)
            self.axe = axe
            axe.set_xlabel("$k_h$")
            axe.set_ylabel("$E(k_h)$")
            axe.set_title(
                "spectra, solver "
                + self.output.name_solver
                + f", nh = {self.nx:5d}"
            )

    def _online_plot_saving(self):
        pass

    def load2d_mean(self, tmin=None, tmax=None):
        with h5py.File(self.path_file2D, "r") as f:
            dset_times = f["times"]
            times = dset_times[...]
            nt = len(times)

            kh = f["khE"][...]

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

            print(
                "compute mean of 2D spectra\n"
                + (
                    "tmin = {0:8.6g} ; tmax = {1:8.6g}"
                    "imin = {2:8d} ; imax = {3:8d}"
                ).format(tmin, tmax, imin_plot, imax_plot)
            )

            dict_results = {"kh": kh}
            for key in list(f.keys()):
                if key.startswith("spectr"):
                    dset_key = f[key]
                    spect = dset_key[imin_plot : imax_plot + 1].mean(0)
                    dict_results[key] = spect
        return dict_results

    def load1d_mean(self, tmin=None, tmax=None):
        with h5py.File(self.path_file1D, "r") as f:
            dset_times = f["times"]
            times = dset_times[...]
            nt = len(times)

            kx = f["kxE"][...]
            # ky = f['kyE'][...]
            kh = kx

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

            print(
                "compute mean of 1D spectra"
                + (
                    "tmin = {0:8.6g} ; tmax = {1:8.6g}\n"
                    "imin = {2:8d} ; imax = {3:8d}\n"
                ).format(tmin, tmax, imin_plot, imax_plot)
            )

            dict_results = {"kh": kh}
            for key in list(f.keys()):
                if key.startswith("spectr"):
                    dset_key = f[key]
                    spect = dset_key[imin_plot : imax_plot + 1].mean(0)
                    dict_results[key] = spect
        return dict_results

    def plot1d(self):
        pass

    def plot2d(self):
        pass
