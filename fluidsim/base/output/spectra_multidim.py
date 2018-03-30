from __future__ import print_function

import os

import numpy as np
import h5py

from fluiddyn.util import mpi

from .base import SpecificOutput


class SpectraMultiDim(SpecificOutput):
    """Saving the multidimensional spectra."""

    _tag = 'spectra_multidim'

    @staticmethod
    def _complete_params_with_default(params):
        tag = 'spectra_multidim'

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag,
                                 attribs={'HAS_TO_PLOT_SAVED': False})
        
    def __init__(self, output):
        self.output = output
        
        params = output.sim.params
        self.nx = int(params.oper.nx)

        if not params.output.HAS_TO_SAVE:
            params.output.periods_save.spectra_multidim = False


        super(SpectraMultiDim, self).__init__(
            output,
            period_save=params.output.periods_save.spectra_multidim,
            has_to_plot_saved=params.output.spectra_multidim.HAS_TO_PLOT_SAVED)

    def _init_path_files(self):
        path_run = self.output.path_run
        self.path_file = path_run + '/spectra_multidim.h5'
        
    def _init_files(self, dico_arrays_1time=None):
        dico_spectra = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file):
                dico_arrays_1time = {'kxE': self.sim.oper.kxE,
                                     'kyE': self.sim.oper.kyE}
                self.create_file_from_dico_arrays(
                    self.path_file, dico_spectra, dico_arrays_1time)
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file, 'r') as f:
                    dset_times = f['times']
                    self.nb_saved_times = dset_times.shape[0] + 1
                # save the spectra in the file spectra_multidim.h5
                self.add_dico_arrays_to_file(self.path_file, dico_spectra)

        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Save the values at one time. """
        tsim = self.sim.time_stepping.t
        if (tsim-self.t_last_save >= self.period_save):
            self.t_last_save = tsim
            dico_spectra = self.compute()
            if mpi.rank == 0:
                # save the spectra in the file spectra_multidim.h5
                self.add_dico_arrays_to_file(self.path_file, dico_spectra)
                self.nb_saved_times += 1
                if self.has_to_plot:
                    self._online_plot_saving(dico_spectra)

                    if (tsim-self.t_last_show >= self.period_show):
                        self.t_last_show = tsim
                        self.axe.get_figure().canvas.draw()

    def compute(self):
        """compute the values at one time."""
        if mpi.rank == 0:
            dico_results = {}
            return dico_results

    def _init_online_plot(self):
        if mpi.rank == 0:
            fig, axe = self.output.figure_axe(numfig=1000000)
            self.axe = axe
            axe.set_xlabel('$k_x$')
            axe.set_ylabel('$k_y$')
            axe.set_title('Multidimensional spectra, solver ' +
                          self.output.name_solver +
                          ', nh = {0:5d}'.format(self.nx))

    def _online_plot_saving(self):
        pass

    def load_mean(self, tmin=None, tmax=None):
        """Loads time averaged data between tmin and tmax."""
        
        dico_results = {}

        # Load data
        with h5py.File(self.path_file, 'r') as f:
            for key in f.keys():
                if not key.startswith('info'):
                    dico_results[key] = f[key].value
                    
        # Time average spectra
        times = dico_results['times']
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

        print('compute mean of multidimensional spectra\n' +
              ('tmin = {0:8.6g} ; tmax = {1:8.6g}\n'
               'imin = {2:8d} ; imax = {3:8d}').format(
                  tmin, tmax, imin_plot, imax_plot))
        
        for key in dico_results.keys():
            if key.startswith('spectr'):
                spect = dico_results[key]
                spect_averaged = spect[imin_plot:imax_plot+1].mean(0)
                dico_results[key] = spect_averaged    

        return dico_results

    def plot(self):
        pass