"""
Correl freq (:mod:`fluidsim.solvers.plate2d.output.correlations_freq`)
============================================================================

.. currentmodule:: fluidsim.solvers.plate2d.output.correlations_freq

Provides:

.. autoclass:: CorrelationsFreq
   :members:
   :private-members:

"""
import h5py

import os
import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.output.base import SpecificOutput


class CorrelationsFreq(SpecificOutput):
    """Compute, save, load and plot correlations of frequencies.

    """

    _tag = 'correl'

    @staticmethod
    def _complete_params_with_default(params):
        tag = 'correl'

        params.output.periods_save.set_attrib(tag, 0)
        params.output.set_child(tag,
                                attribs={
                                    'HAS_TO_PLOT_SAVED': False,
                                    'nb_times': 1000})

    def __init__(self, output):
        params = output.sim.params

        self.nb_times = params.output.correl.nb_times
        self.nb_omegas = self.nb_times/4

        super(CorrelationsFreq, self).__init__(
            output,
            period_save=params.output.periods_save.correl,
            has_to_plot_saved=params.output.correl.HAS_TO_PLOT_SAVED)

    def init_path_files(self):
        path_run = self.output.path_run
        self.path_file = path_run + '/correlations_freq.h5'

    def init_files(self, dico_arrays_1time=None):
        correlations = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file):
                dico_arrays_1time = {'kxE': self.sim.oper.kxE,
                                     'kyE': self.sim.oper.kyE}
                self.create_file_from_dico_arrays(
                    self.path_file, correlations, dico_arrays_1time)
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file, 'r') as f:
                    dset_times = f['times']
                    self.nb_saved_times = dset_times.shape[0]+1
                # save the spectra in the file spectra1D.h5
                self.add_dico_arrays_to_file(self.path_file1D,
                                             correlations)

        self.t_last_save = self.sim.time_stepping.t

    def online_save(self):
        """Save the values at one time. """
        tsim = self.sim.time_stepping.t
        if (tsim-self.t_last_save >= self.period_save):
            self.t_last_save = tsim
            correlations = self.compute()
            if mpi.rank == 0:
                # save the spectra in the file correlation_freq.h5
                self.add_dico_arrays_to_file(self.path_file,
                                             correlations)
                self.nb_saved_times += 1
                # if self.has_to_plot:
                #     self._online_plot(dico_spectra1D, dico_spectra2D)

                #     if (tsim-self.t_last_show >= self.period_show):
                #         self.t_last_show = tsim
                #         self.axe.get_figure().canvas.draw()

    def compute(self):
        """compute the values at one time."""
        if mpi.rank == 0:
            dico_results = {}
            return dico_results

    def init_online_plot(self):
        fig, axe = self.output.figure_axe(numfig=4000000)
        self.axe = axe
        axe.set_xlabel('?')
        axe.set_ylabel('?')
        axe.set_title('Correlation, solver '+self.output.name_solver +
                      ', nh = {0:5d}'.format(self.nx))
        axe.hold(True)

    def _online_plot(self):
        pass

    # def load2D_mean(self, tmin=None, tmax=None):
    #     f = h5py.File(self.path_file2D, 'r')
    #     dset_times = f['times']
    #     times = dset_times[...]
    #     nt = len(times)

    #     kh = f['khE'][...]

    #     if tmin is None:
    #         imin_plot = 0
    #     else:
    #         imin_plot = np.argmin(abs(times-tmin))

    #     if tmax is None:
    #         imax_plot = nt-1
    #     else:
    #         imax_plot = np.argmin(abs(times-tmax))

    #     tmin = times[imin_plot]
    #     tmax = times[imax_plot]

    #     print('compute mean of 2D spectra\n'
    #           ('tmin = {0:8.6g} ; tmax = {1:8.6g}'
    #            'imin = {2:8d} ; imax = {3:8d}').format(
    #               tmin, tmax, imin_plot, imax_plot))

    #     dico_results = {'kh': kh}
    #     for key in f.keys():
    #         if key.startswith('spectr'):
    #             dset_key = f[key]
    #             spect = dset_key[imin_plot:imax_plot+1].mean(0)
    #             dico_results[key] = spect
    #     return dico_results

    def plot(self):
        pass

    def _compute_correl4(self, w):
        r"""Compute the correlations 4.

        .. math::
           C_4(\omega_1, \omega_2, \omega_3, \omega_4) =
           \langle
           \tilde w(\omega_1, \mathbf{x}) +
           \tilde w(\omega_2, \mathbf{x}) +
           \tilde w(\omega_3, \mathbf{x})^* +
           \tilde w(\omega_4, \mathbf{x})^* +
           \rangle_\mathbf{x},

        where

        .. math::
           \omega_2 = \omega_3 + \omega_4 - \omega_1

        and :math:`\omega_1 > 0`, :math:`\omega_3 > 0` and
        :math:`\omega_4 > 0`. Thus, this function produces an array
        :math:`C_4(\omega_1, \omega_3, \omega_4)`.

        """

        nt, ny, nx = w.shape

        w_fftt = self.oper_fft1.fft(w).reshape([nt, nx*ny])
        w_fftt_conj = w_fftt.conj()

        nb_omegas = self.nb_omegas

        iomegas1 = self.iomegas1

        corr4 = np.empty([len(iomegas1), nb_omegas, nb_omegas])

        for i1, io1 in enumerate(iomegas1):
            # this loop could be parallelized (OMP)
            for io3 in range(nb_omegas):
                # we use the symmetry omega_3 <--> omega_4
                for io4 in range(0, io3+1):
                    tmp = (w_fftt[io1, :] *
                           w_fftt_conj[io3, :] *
                           w_fftt_conj[io4, :])
                    io2 = io3 + io4 - io1
                    if io2 < 0:
                        io2 = abs(io2)
                        corr4[i1, io3, io4] = np.mean(tmp*w_fftt_conj[io2, :])
                    else:
                        corr4[i1, io3, io4] = np.mean(tmp*w_fftt[io3, :])
                # symmetry omega_3 <--> omega_4:
                corr4[i1, io4, io3] = corr4[i1, io3, io4]

        # if mpi.nb_proc > 1:
        #     # reduce for mean:
        #     mpi.comm.

        return corr4
