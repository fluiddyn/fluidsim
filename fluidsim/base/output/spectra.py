"""Spectra
==========

Provides:

.. autoclass:: MoviesSpectra
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

import numpy as np
import h5py

from fluiddyn.util import mpi

from .base import SpecificOutput
from fluidsim_core.output.movies import MoviesBase1D


class MoviesSpectra(MoviesBase1D):
    _name_attr_path = "path_file2D"
    _half_key = "spectrum2D_"
    _key_axis = "khE"

    def __init__(self, output, spectra):
        self.spectra = spectra
        super().__init__(output)

    @property
    def path(self):
        # needed to follow the simulation dir which can be moved
        return getattr(self.spectra, self._name_attr_path)

    def _init_ani_times(self, tmin, tmax, dt_equations):
        """Initialization of the variable ani_times for one animation."""

        if tmax is None:
            tmax = self.times.max()

        if tmin is None:
            tmin = self.times.min()

        if dt_equations is None:
            dt_equations = self.params.periods_save.spectra
            if dt_equations == 0:
                if len(self.times) > 1:
                    dt_equations = self.times[1] - self.times[0]
                else:
                    dt_equations = tmax / 2

        super()._init_ani_times(tmin, tmax, dt_equations)

    def init_animation(self, *args, **kwargs):
        if "xmax" not in kwargs:
            kwargs["xmax"] = self.oper.khE[-1:][0]
        if "ymax" not in kwargs:
            kwargs["ymax"] = 1.0

        with h5py.File(self.path) as file:
            self.times = file["times"][...]

        super().init_animation(*args, **kwargs)

    def get_field_to_plot(self, time, key=None):
        if key is None:
            key = self.key_field
        idx, t_file = self.get_closest_time_file(time)
        with h5py.File(self.path) as file:
            y = file[self._half_key + key][idx]
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
        with h5py.File(self.path) as file:
            return file[self._key_axis][...]

    def _set_key_field(self, key_field):
        """
        Defines key_field default.
        """
        if key_field is None:
            key_field = "E"
        self.key_field = key_field


class SpectraBase(SpecificOutput):
    _tag = "spectra"
    _cls_movies = MoviesSpectra
    _possible_ndims = (1, 2)

    @staticmethod
    def _complete_params_with_default(params):
        tag = "spectra"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag, attribs={"HAS_TO_PLOT_SAVED": False})

    def _get_styleline(self, ndim, direction, kind=None):
        style_line = "r"
        if ndim == 1:
            if direction == "y":
                return "g"
            elif direction == "z":
                return "b"
        return style_line

    def _get_key_spectrum(self, ndim, direction=None, kind=None):
        if ndim == 1:
            return f"spectrum1Dk{direction}_E"
        else:
            return "spectrum2D_E"

    def _get_key_wavenumber(self, ndim, direction=None):
        if ndim == 1:
            return f"k{direction}E"
        else:
            return "khE"

    def _get_pathfile_from_ndim(self, ndim):
        return getattr(self, f"path_file{ndim}D")

    def _get_default_directions(self, ndim):
        if ndim != 1:
            return (None,)
        else:
            return ("x", "y")

    def _plot_ndim(
        self,
        tmin=0,
        tmax=None,
        delta_t=None,
        with_average=True,
        coef_compensate=0,
        coef_plot_k3=None,
        coef_plot_k53=None,
        coef_plot_k43=None,
        coef_plot_k2=None,
        xlim=None,
        ylim=None,
        ndim=1,
        directions=None,
    ):
        if ndim not in self._possible_ndims:
            raise ValueError

        if directions is None:
            directions = self._get_default_directions(ndim)

        if ndim != 1:
            directions = (None,)

        path_file = self._get_pathfile_from_ndim(ndim)

        key_k = self._get_key_wavenumber(ndim, directions[0])
        if ndim == 1:
            key_k_label = r",\ ".join(["k_" + letter for letter in directions])
        else:
            key_k_label = "k"

        with h5py.File(path_file, "r") as h5file:
            times = h5file["times"][...]
            ks = h5file[key_k][...]

        if tmax is None:
            tmax = times.max()

        imin_plot = np.argmin(abs(times - tmin))
        imax_plot = np.argmin(abs(times - tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]

        print(
            f"plot{ndim}d(tmin={tmin:8.6g}, tmax={tmax:8.6g},"
            f" coef_compensate={coef_compensate:.3f})\n"
            f"""plot {ndim}D spectra
tmin = {tmin_plot:8.6g} ; tmax = {tmax_plot:8.6g}
imin = {imin_plot:8d} ; imax = {imax_plot:8d}"""
        )

        if delta_t == 0:
            delta_t = None

        if delta_t is not None:
            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))
            delta_t = delta_t_save * delta_i_plot
            if delta_i_plot == 0:
                delta_i_plot = 1
        else:
            delta_i_plot = None

        fig, ax = self.output.figure_axe()
        ax.set_xlabel(f"${key_k_label}$")
        ax.set_ylabel("spectra")
        ax.set_title(
            f"{ndim}D spectra (tmin={tmin:.2g}, tmax={tmax:.2g})\n"
            + self.output.summary_simul
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        for direction in directions:
            self._plot_ndim_direction(
                ax,
                ndim,
                direction,
                imin_plot,
                imax_plot,
                delta_i_plot,
                with_average,
                coef_compensate,
            )

        ks = np.linspace(10 * ks[1], 0.6 * ks[-1], 4)

        ks_no0 = ks.copy()
        ks_no0[ks == 0] = np.nan
        coef_norm = ks_no0**coef_compensate

        if coef_plot_k3 is not None:
            to_plot = coef_plot_k3 * ks_no0 ** (-3) * coef_norm
            ax.plot(ks, to_plot, "k--", label=r"$\propto k^{-3}$")

        if coef_plot_k53 is not None:
            to_plot = coef_plot_k53 * ks_no0 ** (-5.0 / 3) * coef_norm
            ax.plot(ks, to_plot, "k-.", label=r"$\propto k^{-5/3}$")

        if coef_plot_k2 is not None:
            to_plot = coef_plot_k2 * ks_no0 ** (-2) * coef_norm
            ax.plot(ks, to_plot, "k:", label=r"$\propto k^{-2}$")

        if coef_plot_k43 is not None:
            to_plot = coef_plot_k43 * ks_no0 ** (-4.0 / 3) * coef_norm
            ax.plot(ks, to_plot, "k:", label=r"$\propto k^{-4/3}$")

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if ndim == 1:
            ax.legend(loc="lower left")

        return ax

    def _plot_ndim_direction(
        self,
        ax,
        ndim,
        direction,
        imin_plot,
        imax_plot,
        delta_i_plot,
        with_average,
        coef_compensate,
        linewidth=2,
    ):
        path_file = self._get_pathfile_from_ndim(ndim)
        key_k = self._get_key_wavenumber(ndim, direction)

        with h5py.File(path_file, "r") as h5file:
            ks = h5file[key_k][...]

        ks_no0 = ks.copy()
        ks_no0[ks == 0] = np.nan
        coef_norm = ks_no0 ** (coef_compensate)

        style_line = self._get_styleline(ndim, direction)

        if delta_i_plot is not None:
            alpha_min = 0.1
            alpha_max = 1.0
            slope = (alpha_max - alpha_min) / (imax_plot - imin_plot)
            alpha0 = alpha_min - slope * imin_plot

            with h5py.File(path_file, "r") as h5file:
                dset_spectra = h5file[self._get_key_spectrum(ndim, direction)]
                for i_spect in range(imin_plot, imax_plot + 1, delta_i_plot):
                    spectrum = dset_spectra[i_spect]
                    spectrum[spectrum < 10e-16] = np.nan
                    alpha = alpha0 + slope * i_spect

                    if i_spect == imin_plot and not with_average:
                        label = f"$E(k_{direction})$"
                    else:
                        label = None

                    ax.plot(
                        ks,
                        spectrum * coef_norm,
                        style_line,
                        linewidth=1,
                        label=label,
                        alpha=alpha,
                    )

        if with_average:
            self._ax_add_average(
                ax,
                path_file,
                ndim,
                direction,
                ks,
                imin_plot,
                imax_plot,
                coef_norm,
            )

    def _get_averaged_spectrum(
        self, path_file, ndim, direction, imin_plot, imax_plot, kind=None
    ):
        with h5py.File(path_file, "r") as h5file:
            dset_spectra = h5file[self._get_key_spectrum(ndim, direction, kind)]
            spectra = dset_spectra[imin_plot : imax_plot + 1]
            spectrum = spectra.mean(0)
            spectrum[spectrum < 10e-16] = np.nan
        return spectrum

    def _ax_add_average(
        self, ax, path_file, ndim, direction, ks, imin_plot, imax_plot, coef_norm
    ):
        spectrum = self._get_averaged_spectrum(
            path_file, ndim, direction, imin_plot, imax_plot
        )
        style_line = self._get_styleline(ndim, direction)
        ax.plot(
            ks,
            spectrum * coef_norm,
            style_line,
            linewidth=2,
            label=f"$E(k_{direction})$",
        )


class Spectra(SpectraBase):
    """Used for the saving of spectra."""

    def __init__(self, output):
        self.output = output

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

    def _init_files(self, arrays_1st_time=None):
        dict_spectra1D, dict_spectra2D = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file1D):
                arrays_1st_time = {
                    "kxE": self.sim.oper.kxE,
                    "kyE": self.sim.oper.kyE,
                }
                self._create_file_from_dict_arrays(
                    self.path_file1D, dict_spectra1D, arrays_1st_time
                )
                arrays_1st_time = {"khE": self.sim.oper.khE}
                self._create_file_from_dict_arrays(
                    self.path_file2D, dict_spectra2D, arrays_1st_time
                )
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file1D, "r") as file:
                    dset_times = file["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                # save the spectra in the file spectra1D.h5
                self._add_dict_arrays_to_file(self.path_file1D, dict_spectra1D)
                # save the spectra in the file spectra2D.h5
                self._add_dict_arrays_to_file(self.path_file2D, dict_spectra2D)

        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Save the values at one time."""
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
                        self.ax.get_figure().canvas.draw()

    def compute(self):
        """compute the values at one time."""
        if mpi.rank == 0:
            dict_results = {}
            return dict_results

    def _init_online_plot(self):
        if mpi.rank == 0:
            fig, ax = self.output.figure_axe(numfig=1_000_000)
            self.ax = ax
            ax.set_xlabel("$k_h$")
            ax.set_ylabel("$E(k_h)$")
            ax.set_title("spectra\n" + self.output.summary_simul)

    def _online_plot_saving(self):
        pass

    def load2d_mean(self, tmin=None, tmax=None):
        with h5py.File(self.path_file2D, "r") as file:
            dset_times = file["times"]
            times = dset_times[...]
            nt = len(times)

            kh = file["khE"][...]

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
            for key in list(file.keys()):
                if key.startswith("spectr"):
                    dset_key = file[key]
                    spect = dset_key[imin_plot : imax_plot + 1].mean(0)
                    dict_results[key] = spect
        return dict_results

    def load1d_mean(self, tmin=None, tmax=None):
        with h5py.File(self.path_file1D, "r") as file:
            dset_times = file["times"]
            times = dset_times[...]
            nt = len(times)

            kx = file["kxE"][...]
            # ky = file['kyE'][...]
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
            for key in list(file.keys()):
                if key.startswith("spectr"):
                    dset_key = file[key]
                    spect = dset_key[imin_plot : imax_plot + 1].mean(0)
                    dict_results[key] = spect
        return dict_results

    def plot1d(self):
        raise NotImplementedError

    def plot2d(self):
        raise NotImplementedError
