import os
from textwrap import dedent

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
            kwargs["xmax"] = self.oper.k_spectra3d[-1]
        if "ymax" not in kwargs:
            kwargs["ymax"] = 1.0

        with h5py.File(self.spectra.path_file3d) as file:
            self.times = file["times"][...]

        super().init_animation(*args, **kwargs)

    def get_field_to_plot(self, time, key=None):
        if key is None:
            key = self.key_field
        idx, t_file = self.get_closest_time_file(time)
        with h5py.File(self.spectra.path_file3d) as file:
            y = file["spectra_" + key][idx]
        y[abs(y) < 10e-16] = 0
        return y, t_file

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
        with h5py.File(self.spectra.path_file3d) as file:
            x = file["k_spectra3d"][...]

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
        p_spectra = params.output._set_child(
            tag, attribs={"HAS_TO_PLOT_SAVED": False, "kzkh_periodicity": 0}
        )
        p_spectra._set_doc(
            dedent(
                """
                    HAS_TO_PLOT_SAVED : bool (False)

                      If True, some curves can be plotted during the run.

                    kzkh_periodicity : int (0)

                      Periodicity of saving of (kz, kh) spectra (compared to standard spectra).
        """
            )
        )

    def _init_movies(self):
        self.movies = MoviesSpectra(self.output, self)

    def __init__(self, output):
        self.output = output

        if hasattr(self, "_init_movies"):
            self._init_movies()

        params = output.sim.params
        self.kzkh_periodicity = params.output.spectra.kzkh_periodicity
        self.nx = int(params.oper.nx)
        self.nb_saved_times = 0

        super().__init__(
            output,
            period_save=params.output.periods_save.spectra,
            has_to_plot_saved=params.output.spectra.HAS_TO_PLOT_SAVED,
        )

    def _init_path_files(self):
        path_run = self.output.path_run
        self.path_file1d = path_run + "/spectra1d.h5"
        self.path_file3d = path_run + "/spectra3d.h5"
        self.path_file_kzkh = path_run + "/spectra_kzkh.h5"

    def _init_files(self, arrays_1st_time=None):
        dict_spectra1d, dict_spectra3d, dict_kzkh = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file1d):
                oper = self.sim.oper
                kx = oper.deltakx * np.arange(oper.nkx_spectra)
                ky = oper.deltaky * np.arange(oper.nky_spectra)
                kz = oper.deltakz * np.arange(oper.nkz_spectra)
                self._create_file_from_dict_arrays(
                    self.path_file1d,
                    dict_spectra1d,
                    {"kx": kx, "ky": ky, "kz": kz},
                )
                self._create_file_from_dict_arrays(
                    self.path_file3d,
                    dict_spectra3d,
                    {"k_spectra3d": self.sim.oper.k_spectra3d},
                )
                if self.kzkh_periodicity:
                    self._create_file_from_dict_arrays(
                        self.path_file_kzkh,
                        dict_kzkh,
                        {"kz": kz, "kh_spectra": self.sim.oper.kh_spectra},
                    )

                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file1d, "r") as file:
                    dset_times = file["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                # save the spectra in the file spectra1d.h5
                self._add_dict_arrays_to_file(self.path_file1d, dict_spectra1d)
                # save the spectra in the file spectra3d.h5
                self._add_dict_arrays_to_file(self.path_file3d, dict_spectra3d)

                if self.has_to_save_kzkh(only_rank0=True):
                    self._add_dict_arrays_to_file(self.path_file_kzkh, dict_kzkh)

        self.t_last_save = self.sim.time_stepping.t

    def has_to_save_kzkh(self, only_rank0=False):
        if mpi.rank == 0:
            answer = (
                self.kzkh_periodicity
                and self.nb_saved_times % self.kzkh_periodicity == 0
            )
        else:
            answer = None

        if only_rank0 or mpi.nb_proc == 1:
            return answer

        return mpi.comm.bcast(answer, root=0)

    def _online_save(self):
        """Save the values at one time."""
        tsim = self.sim.time_stepping.t
        if self._has_to_online_save():
            self.t_last_save = tsim
            dict_spectra1d, dict_spectra3d, dict_kzkh = self.compute()
            if mpi.rank == 0:
                # save the spectra in the file spectra1D.h5
                self._add_dict_arrays_to_file(self.path_file1d, dict_spectra1d)
                # save the spectra in the file spectra2D.h5
                self._add_dict_arrays_to_file(self.path_file3d, dict_spectra3d)
                if self.has_to_save_kzkh(only_rank0=True):
                    self._add_dict_arrays_to_file(self.path_file_kzkh, dict_kzkh)
                self.nb_saved_times += 1
                if self.has_to_plot:
                    self._online_plot_saving(dict_spectra1d, dict_spectra3d)

                    if tsim - self.t_last_show >= self.period_show:
                        self.t_last_show = tsim
                        self.axe.get_figure().canvas.draw()

    def compute(self):
        """compute the values at one time."""
        raise NotImplementedError

    def _init_online_plot(self):
        if mpi.rank == 0:
            fig, axe = self.output.figure_axe(numfig=1_000_000)
            self.axe = axe
            axe.set_xlabel("$k$")
            axe.set_ylabel("$E(k)$")
            axe.set_title("spectra\n" + self.output.summary_simul)

    def _online_plot_saving(self, dict_spectra1d, dict_spectra3d):
        pass

    def _load_mean_file(self, path, tmin=None, tmax=None, key_to_load=None):
        results = {}
        with h5py.File(path, "r") as h5file:
            times = h5file["times"][...]
            nt = len(times)
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
                "compute mean of spectra\n"
                f"tmin = {tmin:8.6g} ; tmax = {tmax:8.6g}\n"
                f"imin = {imin_plot:8d} ; imax = {imax_plot:8d}"
            )

            if key_to_load is not None:
                if isinstance(key_to_load, str):
                    keys = [key_to_load]
                else:
                    keys = key_to_load
                for key in keys:
                    if key not in h5file.keys():
                        raise ValueError(f"{key} not in {h5file.keys()}")
                    spect = h5file[key][imin_plot : imax_plot + 1].mean(0)
                    results[key] = spect
                return results

            for key in list(h5file.keys()):
                if key.startswith("spectr"):
                    dset_key = h5file[key]
                    spect = dset_key[imin_plot : imax_plot + 1].mean(0)
                    results[key] = spect
        return results

    def load3d_mean(self, tmin=None, tmax=None):
        results = self._load_mean_file(self.path_file3d, tmin, tmax)
        with h5py.File(self.path_file3d, "r") as h5file:
            results["k"] = h5file["k_spectra3d"][...]
        return results

    def load1d_mean(self, tmin=None, tmax=None):
        results = self._load_mean_file(self.path_file1d, tmin, tmax)
        with h5py.File(self.path_file1d, "r") as h5file:
            for key in ("kx", "ky", "kz"):
                results[key] = h5file[key][...]
        return results

    def load_kzkh_mean(self, tmin=None, tmax=None, key_to_load=None):

        if not os.path.exists(self.path_file_kzkh):
            raise RuntimeError(
                self.path_file_kzkh
                + " does not exist. Can't load values from it."
            )

        results = self._load_mean_file(
            self.path_file_kzkh, tmin, tmax, key_to_load
        )
        with h5py.File(self.path_file_kzkh, "r") as h5file:
            for key in ("kz", "kh_spectra"):
                results[key] = h5file[key][...]

        return results

    def plot1d(self):
        pass

    def plot3d(self):
        pass
