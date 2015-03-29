import h5py

import os
import numpy as np

from fluiddyn.util import mpi


from .base import SpecificOutput


class Spectra(SpecificOutput):
    """Used for the saving of spectra.


    """

    _tag = 'spectra'

    @staticmethod
    def _complete_params_with_default(params):
        tag = 'spectra'

        params.output.periods_save.set_attrib(tag, 0)
        params.output.set_child(tag,
                                attribs={'HAS_TO_PLOT_SAVED': False})

    def __init__(self, output):
        params = output.sim.params
        self.nx = params.oper.nx

        self.spectrum2D_from_fft = output.sim.oper.spectrum2D_from_fft
        self.spectra1D_from_fft = output.sim.oper.spectra1D_from_fft

        super(Spectra, self).__init__(
            output,
            period_save=params.output.periods_save.spectra,
            has_to_plot_saved=params.output.spectra.HAS_TO_PLOT_SAVED)

    def init_path_files(self):
        path_run = self.output.path_run
        self.path_file1D = path_run + '/spectra1D.h5'
        self.path_file2D = path_run + '/spectra2D.h5'

    def init_files(self, dico_arrays_1time=None):
        dico_spectra1D, dico_spectra2D = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file1D):
                dico_arrays_1time = {'kxE': self.sim.oper.kxE,
                                     'kyE': self.sim.oper.kyE}
                self.create_file_from_dico_arrays(
                    self.path_file1D, dico_spectra1D, dico_arrays_1time)
                dico_arrays_1time = {'khE': self.sim.oper.khE}
                self.create_file_from_dico_arrays(
                    self.path_file2D, dico_spectra2D, dico_arrays_1time)
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file1D, 'r') as f:
                    dset_times = f['times']
                    self.nb_saved_times = dset_times.shape[0]+1
                # save the spectra in the file spectra1D.h5
                self.add_dico_arrays_to_file(self.path_file1D,
                                             dico_spectra1D)
                # save the spectra in the file spectra2D.h5
                self.add_dico_arrays_to_file(self.path_file2D,
                                             dico_spectra2D)

        self.t_last_save = self.sim.time_stepping.t

    def online_save(self):
        """Save the values at one time. """
        tsim = self.sim.time_stepping.t
        if (tsim-self.t_last_save >= self.period_save):
            self.t_last_save = tsim
            dico_spectra1D, dico_spectra2D = self.compute()
            if mpi.rank == 0:
                # save the spectra in the file spectra1D.h5
                self.add_dico_arrays_to_file(self.path_file1D,
                                             dico_spectra1D)
                # save the spectra in the file spectra2D.h5
                self.add_dico_arrays_to_file(self.path_file2D,
                                             dico_spectra2D)
                self.nb_saved_times += 1
                if self.has_to_plot:
                    self._online_plot(dico_spectra1D, dico_spectra2D)

                    if (tsim-self.t_last_show >= self.period_show):
                        self.t_last_show = tsim
                        self.axe.get_figure().canvas.draw()

    def compute(self):
        """compute the values at one time."""
        if mpi.rank == 0:
            dico_results = {}
            return dico_results

    def init_online_plot(self):
        fig, axe = self.output.figure_axe(numfig=1000000)
        self.axe = axe
        axe.set_xlabel('$k_h$')
        axe.set_ylabel('$E(k_h)$')
        axe.set_title('spectra, solver '+self.output.name_solver +
                      ', nh = {0:5d}'.format(self.nx))
        axe.hold(True)

    def _online_plot(self):
        pass

    def load2D_mean(self, tmin=None, tmax=None):
        f = h5py.File(self.path_file2D, 'r')
        dset_times = f['times']
        times = dset_times[...]
        nt = len(times)

        kh = f['khE'][...]

        if tmin is None:
            imin_plot = 0
        else:
            imin_plot = np.argmin(abs(times-tmin))

        if tmax is None:
            imax_plot = nt-1
        else:
            imax_plot = np.argmin(abs(times-tmax))

        tmin = times[imin_plot]
        tmax = times[imax_plot]

        print('compute mean of 2D spectra\n'
              ('tmin = {0:8.6g} ; tmax = {1:8.6g}'
               'imin = {2:8d} ; imax = {3:8d}').format(
                  tmin, tmax, imin_plot, imax_plot))

        dico_results = {'kh': kh}
        for key in f.keys():
            if key.startswith('spectr'):
                dset_key = f[key]
                spect = dset_key[imin_plot:imax_plot+1].mean(0)
                dico_results[key] = spect
        return dico_results

    def load1D_mean(self, tmin=None, tmax=None):
        f = h5py.File(self.path_file1D, 'r')
        dset_times = f['times']
        times = dset_times[...]
        nt = len(times)

        kx = f['kxE'][...]
        # ky = f['kyE'][...]
        kh = kx

        if tmin is None:
            imin_plot = 0
        else:
            imin_plot = np.argmin(abs(times-tmin))

        if tmax is None:
            imax_plot = nt-1
        else:
            imax_plot = np.argmin(abs(times-tmax))

        tmin = times[imin_plot]
        tmax = times[imax_plot]

        print('compute mean of 1D spectra'
              ('tmin = {0:8.6g} ; tmax = {1:8.6g}\n'
               'imin = {2:8d} ; imax = {3:8d}\n').format(
                   tmin, tmax, imin_plot, imax_plot))

        dico_results = {'kh': kh}
        for key in f.keys():
            if key.startswith('spectr'):
                dset_key = f[key]
                spect = dset_key[imin_plot:imax_plot+1].mean(0)
                dico_results[key] = spect
        return dico_results

    def plot1D(self):
        pass

    def plot2D(self):
        pass
