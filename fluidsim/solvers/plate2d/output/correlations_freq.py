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
from fluidsim.operators.fft.easypyfft import FFTW1DReal2Complex


class CorrelationsFreq(SpecificOutput):
    """Compute, save, load and plot correlations of frequencies.

    """

    _tag = 'correl_freq'

    @staticmethod
    def _complete_params_with_default(params):
        tag = 'correl_freq'

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag,
                                attribs={
                                    'HAS_TO_PLOT_SAVED': False,
                                    'nb_times_compute': 100,
                                    'coef_decimate': 10,
                                    'key_quantity': 'w'})

    def __init__(self, output):
        params = output.sim.params

        self.nb_times_compute = params.output.correl_freq.nb_times_compute
        self.coef_decimate = params.output.correl_freq.coef_decimate
        self.key_quantity = params.output.correl_freq.key_quantity
        self.periods_fill = params.output.periods_save.correl_freq

        n0 = len(range(0, output.sim.oper.shapeX_loc[0], self.coef_decimate))
        n1 = len(range(0, output.sim.oper.shapeX_loc[1], self.coef_decimate))
        shape_spatio_temp = [n0*n1, self.nb_times_compute]
        self.oper_fft1 = FFTW1DReal2Complex(shape_spatio_temp)
        self.nb_omegas = self.oper_fft1.shapeK[-1]

        self.spatio_temp = np.empty(shape_spatio_temp)
        self.nb_times_in_spatio_temp = 0

        super(CorrelationsFreq, self).__init__(
            output,
            period_save=params.output.periods_save.correl_freq,
            has_to_plot_saved=params.output.correl_freq.HAS_TO_PLOT_SAVED)

        if os.path.exists(self.path_file4):
            with h5py.File(self.path_file, 'r') as f:
                if self.sim.time_stepping.deltat != f.attrs['deltat']:
                    raise ValueError()

    def init_path_files(self):
        path_run = self.output.path_run
        # self.path_file = path_run + '/spectra_temporal.h5'
        self.path_file4 = path_run + '/correl4_freq.h5'

    def init_files(self, dico_arrays_1time=None):
        # we can not do anything when this function is called.
        pass

    def init_files2(self, dcorrel4):
        if mpi.rank == 0:
            if not os.path.exists(self.path_file):
                time_tot = (
                    self.sim.time_stepping.deltat * self.nb_times_compute *
                    self.period_fill)
                omegas = 2*np.pi/time_tot * np.arange(self.nb_omegas)
                dico_arrays_1time = {
                    'omegas': omegas,
                    'deltat': self.sim.time_stepping.deltat,
                    'nb_times_compute': self.nb_times_compute,
                    'period_fill': self.period_fill}
                self.create_file_from_dico_arrays(
                    self.path_file, scorrel4, dico_arrays_1time)
                self.nb_saved_times = 1

        self.t_last_save = self.sim.time_stepping.t

    def online_save(self):
        """Save the values at one time. """
        itsim = self.sim.time_stepping.t/self.sim.time_stepping.deltat
        if (itsim-self.it_last_run >= self.period_save):
            self.it_last_run = itsim
            field = self.sim.state.state_phys.get_var(self.key_quantity)
            field = field[::self.coef_decimate, ::self.coef_decimate]
            self.spatio_temp[:, self.nb_times_in_spatio_temp] = (
                field.reshape([field.size]))
            self.nb_times_in_spatio_temp += 1
            if self.nb_times_in_spatio_temp == self.nb_times_compute:
                self.nb_times_in_spatio_temp = 0
                self.t_last_save = tsim
                correlations = self.compute()
                if mpi.rank == 0:
                    # save the spectra in the file correlation_freq.h5
                    self.add_dico_arrays_to_file(self.path_file4,
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
            state_fft = self.sim.state.state_fft
            w_fft = state_fft.get_var('w_fft')
            #w = self
            dico_results = {}
            return dico_results

    def init_online_plot(self):
        fig, axe = self.output.figure_axe(numfig=4000000)
        self.axe = axe
        axe.set_xlabel('?')
        axe.set_ylabel('?')
        #axe.set_title('Correlation, solver '+self.output.name_solver +
        #              ', nh = {0:5d}'.format(self.nx))
        axe.hold(True)

    def _online_plot(self):
        pass

    def plot(self):
        pass

    def _compute_correl4(self, q_fftt):
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

        #nt, ny, nx = w.shape

        #q_fftt = self.oper_fft1.fft(w).reshape([nt, nx*ny])
        q_fftt_conj = q_fftt.conj()

        nb_omegas = self.nb_omegas

        iomegas1 = self.iomegas1

        corr4 = np.empty([len(iomegas1), nb_omegas, nb_omegas])

        for i1, io1 in enumerate(iomegas1):
            # this loop could be parallelized (OMP)
            for io3 in range(nb_omegas):
                # we use the symmetry omega_3 <--> omega_4
                for io4 in range(0, io3+1):
                    tmp = (q_fftt[io1, :] *
                           q_fftt_conj[io3, :] *
                           q_fftt_conj[io4, :])
                    io2 = io3 + io4 - io1
                    if io2 < 0:
                        io2 = abs(io2)
                        corr4[i1, io3, io4] = np.mean(tmp*q_fftt_conj[io2, :])
                    else:
                        corr4[i1, io3, io4] = np.mean(tmp*q_fftt[io3, :])
                # symmetry omega_3 <--> omega_4:
                corr4[i1, io4, io3] = corr4[i1, io3, io4]

        # if mpi.nb_proc > 1:
        #     # reduce for mean:
        #     mpi.comm.

        return corr4
