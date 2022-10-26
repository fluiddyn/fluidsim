"""Spectra 3d
=============

Provides:

.. autoclass:: MoviesSpectra
   :members:
   :private-members:
   :noindex:
   :undoc-members:

.. autoclass:: BaseSpectra
   :members:
   :private-members:
   :noindex:
   :undoc-members:

.. autoclass:: Spectra
   :members:
   :private-members:
   :noindex:
   :undoc-members:

"""


import os
from textwrap import dedent
from math import sqrt, tau

import numpy as np
import h5py

from fluiddyn.util import mpi

from .base import SpecificOutput

from .spectra import MoviesSpectra


class MoviesSpectra(MoviesSpectra):
    _name_attr_path = "path_file3d"
    _half_key = "spectra_"
    _key_axis = "k_spectra3d"

    def init_animation(self, *args, **kwargs):
        if "xmax" not in kwargs:
            kwargs["xmax"] = self.oper.k_spectra3d[-1]
        if "ymax" not in kwargs:
            kwargs["ymax"] = 1.0

        super().init_animation(*args, **kwargs)


class BaseSpectra(SpecificOutput):
    """Used for the saving of spectra."""

    _tag = "spectra"
    _cls_movies = MoviesSpectra

    @classmethod
    def _complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)
        p_spectra = params.output._set_child(
            cls._tag, attribs={"HAS_TO_PLOT_SAVED": False, "kzkh_periodicity": 0}
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

    def __init__(self, output):
        self.output = output

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
        self.path_file1d = path_run + f"/{self._tag}1d.h5"
        self.path_file3d = path_run + f"/{self._tag}3d.h5"
        self.path_file_kzkh = path_run + f"/{self._tag}_kzkh.h5"

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
                        self.ax.get_figure().canvas.draw()

    def compute(self):
        """compute the values at one time."""
        raise NotImplementedError

    def _init_online_plot(self):
        if mpi.rank == 0:
            fig, ax = self.output.figure_axe(numfig=1_000_000)
            self.ax = ax
            ax.set_xlabel("$k$")
            ax.set_ylabel("$E(k)$")
            ax.set_title(f"{self._tag}\n{self.output.summary_simul}")

    def _online_plot_saving(self, dict_spectra1d, dict_spectra3d):
        pass

    def _load_mean_file(
        self, path, tmin=None, tmax=None, key_to_load=None, verbose=True
    ):
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

            if verbose:
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

    def load3d_mean(self, tmin=None, tmax=None, verbose=True):
        results = self._load_mean_file(
            self.path_file3d, tmin, tmax, verbose=verbose
        )
        with h5py.File(self.path_file3d, "r") as h5file:
            results["k"] = h5file["k_spectra3d"][...]
        return results

    def load1d_mean(self, tmin=None, tmax=None, verbose=True):
        results = self._load_mean_file(
            self.path_file1d, tmin, tmax, verbose=verbose
        )
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

    def plot_kzkh(
        self, tmin=0, tmax=None, key="xz", ax=None, vmin=None, vmax=None
    ):
        data = self.load_kzkh_mean(tmin, tmax, key)
        if self._tag == "cross_corr":
            plotted = np.sign(data[key]) * np.log10(abs(data[key]))
        else:
            plotted = np.log10(data[key])

        kz = data["kz"]
        kh = data["kh_spectra"]

        if ax is None:
            fig, ax = self.output.figure_axe()
        else:
            fig = ax.figure

        ax.set_xlabel(r"$\kappa_h$")
        ax.set_ylabel("$k_z$")
        ax.set_title("log10 spectra\n" + self.output.summary_simul)

        qmesh = ax.pcolormesh(
            kh, kz, plotted, shading="nearest", vmin=vmin, vmax=vmax
        )
        fig.colorbar(qmesh)


class Spectra(BaseSpectra):
    def compute_isotropy_velocities(
        self, tmin=None, tmax=None, verbose=False, data=None
    ):
        if data is None:
            data = self.load1d_mean(tmin, tmax, verbose)
        kz = data["kz"]
        delta_kz = kz[1]
        EKx_kz = data["spectra_vx_kz"] * delta_kz
        EKy_kz = data["spectra_vy_kz"] * delta_kz
        EKz_kz = data["spectra_vz_kz"] * delta_kz

        EKx_kz[0] = 0
        EKy_kz[0] = 0

        EKz = EKz_kz.sum()
        EK = EKx_kz.sum() + EKy_kz.sum() + EKz

        return 3 * EKz / EK

    def compute_length_scales(
        self, tmin=None, tmax=None, verbose=False, data=None
    ):
        if data is None:
            data = self.load1d_mean(tmin, tmax, verbose)

        lengths = {}

        def add_lengths_i(letter):
            ki = data[f"k{letter}"]
            EK_ki = data[f"spectra_E_k{letter}"]
            sum0i = EK_ki.sum()
            sum1i = (EK_ki * ki).sum()
            sum2i = (EK_ki * ki**2).sum()
            # tau == 2*pi
            lengths[f"l{letter}1"] = tau * sum0i / sum1i
            lengths[f"l{letter}2"] = tau * sqrt(sum0i / sum2i)

        add_lengths_i("x")
        add_lengths_i("y")
        add_lengths_i("z")

        return lengths
