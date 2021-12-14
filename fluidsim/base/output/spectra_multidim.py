import os

import numpy as np
import h5py

from fluiddyn.util import mpi

from .base import SpecificOutput


class SpectraMultiDim(SpecificOutput):
    """Saving the multidimensional spectra."""

    _tag = "spectra_multidim"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "spectra_multidim"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag, attribs={"HAS_TO_PLOT_SAVED": False})

    def __init__(self, output):
        self.output = output

        params = output.sim.params
        self.nx = int(params.oper.nx)

        if not params.output.HAS_TO_SAVE:
            params.output.periods_save.spectra_multidim = False

        super().__init__(
            output,
            period_save=params.output.periods_save.spectra_multidim,
            has_to_plot_saved=params.output.spectra_multidim.HAS_TO_PLOT_SAVED,
        )

    def _init_path_files(self):
        path_run = self.output.path_run
        self.path_file = path_run + "/spectra_multidim.h5"

    def _init_files(self, arrays_1st_time=None):
        dict_spectra = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file):
                arrays_1st_time = {
                    "kxE": self.sim.oper.kxE,
                    "kyE": self.sim.oper.kyE,
                }
                self._create_file_from_dict_arrays(
                    self.path_file, dict_spectra, arrays_1st_time
                )
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file, "r") as file:
                    dset_times = file["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                # save the spectra in the file spectra_multidim.h5
                self._add_dict_arrays_to_file(self.path_file, dict_spectra)

        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Save the values at one time."""
        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_save >= self.period_save:
            self.t_last_save = tsim
            dict_spectra = self.compute()
            if mpi.rank == 0:
                # save the spectra in the file spectra_multidim.h5
                self._add_dict_arrays_to_file(self.path_file, dict_spectra)
                self.nb_saved_times += 1
                if self.has_to_plot:
                    self._online_plot_saving(dict_spectra)

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
            axe.set_xlabel("$k_x$")
            axe.set_ylabel("$k_y$")
            axe.set_title(
                "Multidimensional spectra\n" + self.output.summary_simul
            )

    def _online_plot_saving(self, arg):
        pass

    def load_mean(self, tmin=None, tmax=None):
        """Loads time averaged data between tmin and tmax."""

        results = {}

        # Load data
        with h5py.File(self.path_file, "r") as file:
            for key in file.keys():
                if not key.startswith("info"):
                    results[key] = file[key][...]

        # Time average spectra
        times = results["times"]
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
            "compute mean of multidimensional spectra\n"
            f"tmin = {tmin:8.6g} ; tmax = {tmax:8.6g}\n"
            f"imin = {imin_plot:8d} ; imax = {imax_plot:8d}"
        )

        for key in results.keys():
            if key.startswith("spectr"):
                spect = results[key]
                spect_averaged = spect[imin_plot : imax_plot + 1].mean(0)
                results[key] = spect_averaged

        return results

    def plot(self):
        pass
