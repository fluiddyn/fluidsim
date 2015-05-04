"""
Correl freq (:mod:`fluidsim.solvers.plate2d.output.correlations_freq`)
============================================================================

.. currentmodule:: fluidsim.solvers.plate2d.output.correlations_freq

Provides:

.. autoclass:: CorrelationsFreq
   :members:
   :private-members:

"""
import os
import numpy as np
import h5py

from fluiddyn.util import mpi

from fluidsim.base.output.base import SpecificOutput
from fluidsim.operators.fft.easypyfft import FFTW1DReal2Complex
from fluidsim.operators.miscellaneous import compute_correl4, compute_correl2


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
                                     'it_start': 10,
                                     'nb_times_compute': 100,
                                     'coef_decimate': 10,
                                     'key_quantity': 'w',
                                     'iomegas1': [1]})

    def __init__(self, output):
        params = output.sim.params

        self.nb_times_compute = params.output.correl_freq.nb_times_compute
        self.coef_decimate = params.output.correl_freq.coef_decimate
        self.key_quantity = params.output.correl_freq.key_quantity
        self.periods_fill = params.output.periods_save.correl_freq
        self.iomegas1 = np.array(params.output.correl_freq.iomegas1)
        self.it_last_run = params.output.correl_freq.it_start
        n0 = len(range(0, output.sim.oper.shapeX_loc[0], self.coef_decimate))
        n1 = len(range(0, output.sim.oper.shapeX_loc[1], self.coef_decimate))
        nb_xs = n0 * n1
        shape_spatio_temp = [nb_xs, self.nb_times_compute]
        self.oper_fft1 = FFTW1DReal2Complex(shape_spatio_temp)
        self.nb_omegas = self.oper_fft1.shapeK[-1]
        duration = self.nb_times_compute*self.sim.time_stepping.deltat

        dt = self.periods_fill*self.sim.time_stepping.deltat
        delta_frequency = 1./duration
        frequency_max = delta_frequency*self.iomegas1.max()

        if (0.5/dt) <= frequency_max:
            raise ValueError('freq_max > fe/2')

        if mpi.nb_proc > 1:
            nb_xs = mpi.comm.reduce(nb_xs, op=mpi.MPI.SUM, root=0)

        if mpi.rank > 0:
            nb_xs = 0

        self.nb_xs_seq = nb_xs

        self.spatio_temp = np.empty(shape_spatio_temp)
        self.nb_times_in_spatio_temp = 0

        super(CorrelationsFreq, self).__init__(
            output,
            period_save=params.output.periods_save.correl_freq,
            has_to_plot_saved=params.output.correl_freq.HAS_TO_PLOT_SAVED)
        if os.path.exists(self.path_file4):
            with h5py.File(self.path_file4, 'r') as f:
                link_corr4 = f['corr4']
                link_corr2 = f['corr2']
                link_nb_means = f['nb_means']
                self.corr4 = link_corr4[-1]
                self.corr2 = link_corr2[-1]
                self.nb_means_times = link_nb_means[-1]
        else:
            self.corr4 = np.zeros([len(self.iomegas1),
                                   self.nb_omegas, self.nb_omegas])
            self.corr2 = np.zeros([self.nb_omegas, self.nb_omegas])
            self.nb_means_times = 0

#        if os.path.exists(self.path_file4):
#            with h5py.File(self.path_file4, 'r') as f:
#                if self.sim.time_stepping.deltat != f.attrs['deltat']:
#                    raise ValueError()

    def init_path_files(self):
        path_run = self.output.path_run
        # self.path_file = path_run + '/spectra_temporal.h5'
        self.path_file4 = path_run + '/correl4_freq.h5'

    def init_files(self, dico_arrays_1time=None):
        # we can not do anything when this function is called.
        pass

    def init_files2(self, correlations):
        time_tot = (
            self.sim.time_stepping.deltat * self.nb_times_compute *
            self.periods_fill)
        omegas = 2*np.pi/time_tot * np.arange(self.nb_omegas)
        dico_arrays_1time = {
            'omegas': omegas,
            'deltat': self.sim.time_stepping.deltat,
            'nb_times_compute': self.nb_times_compute,
            'periods_fill': self.periods_fill}
        self.create_file_from_dico_arrays(
            self.path_file4, correlations, dico_arrays_1time)

        self.t_last_save = self.sim.time_stepping.t

    def online_save(self):
        """Save the values at one time. """
        itsim = self.sim.time_stepping.t/self.sim.time_stepping.deltat
        periods_save = self.sim.params.output.periods_save.correl_freq
        if (itsim-self.it_last_run >= periods_save-1):
            self.it_last_run = itsim
            field = self.sim.state.state_phys.get_var(self.key_quantity)
            field = field[::self.coef_decimate, ::self.coef_decimate]
            self.spatio_temp[:, self.nb_times_in_spatio_temp] = (
                field.reshape([field.size]))
            self.nb_times_in_spatio_temp += 1
            if self.nb_times_in_spatio_temp == self.nb_times_compute:
                self.nb_times_in_spatio_temp = 0
                self.t_last_save = self.sim.time_stepping.t
                spatio_fft = self.oper_fft1.fft(self.spatio_temp)
                new_corr4 = compute_correl4(
                    spatio_fft, self.iomegas1, self.nb_omegas, self.nb_xs_seq)
                new_corr2 = compute_correl2(
                    spatio_fft, self.iomegas1, self.nb_omegas, self.nb_xs_seq)

                if mpi.rank == 0:
                    self.corr4 = (1./(self.nb_means_times+1))*(
                        self.nb_means_times*self.corr4 + new_corr4)
                    self.corr2 = (1./(self.nb_means_times+1))*(
                        self.nb_means_times*self.corr2 + new_corr2)
                    self.nb_means_times += 1
                    if np.mod(self.nb_means_times, 2) == 0:
                        correlations = {'corr4': self.corr4,
                                        'corr2': self.corr2,
                                        'nb_means': self.nb_means_times}
                        if not os.path.exists(self.path_file4):
                            self.init_files2(correlations)
                        else:
                            # save the spectra in the file correlation_freq.h5
                            self.add_dico_arrays_to_file(self.path_file4,
                                                         correlations)
                        if self.has_to_plot:
                            self._online_plot(correlations)
                    #     if (tsim-self.t_last_show >= self.period_show):
                    #         self.t_last_show = tsim
                    #         self.axe.get_figure().canvas.draw()

    def init_online_plot(self):
        fig, axe = self.output.figure_axe(numfig=4100000)
        self.axe = axe
        axe.set_xlabel('Frequency')
        axe.set_ylabel('Correlations')
        axe.set_title('Correlation, solver '+self.output.name_solver +
                      ', nh = {0:5d}'.format(self.params.oper.nx))
        axe.hold(True)

    def _online_plot(self, dico_results):
        nb_omegas = self.nb_omegas
        duration = self.nb_times_compute*self.sim.time_stepping.deltat
        delta_frequency = 1./duration

        corr4 = dico_results['corr4']
        corr2 = dico_results['corr2']
        corr = np.empty(corr4.shape)
        fy, fx = np.mgrid[slice(0, delta_frequency*(self.nb_times_compute/2+1),
                                delta_frequency),
                          slice(0, delta_frequency*(self.nb_times_compute/2+1),
                                delta_frequency)]
        for i1, io1 in enumerate(self.iomegas1):
            for io3 in range(nb_omegas):
                for io4 in range(io3+1):
                    io2 = io3 + io4 - io1
                    if io2 < 0:
                        io2 = -io2
                    elif io2 >= nb_omegas:
                        io2 = 2*nb_omegas-1-io2
                    corr[i1, io3, io4] = corr4[i1, io3, io4]/np.sqrt(
                        corr2[io1, io1] * corr2[io3, io3] * corr2[io4, io4] *
                        corr2[io2, io2])
                    corr[i1, io4, io3] = corr[i1, io3, io4]

        fig, axe = self.output.figure_axe(numfig=4100000)
        self.axe = axe
        axe.set_xlabel('Frequency')
        axe.set_xlabel('Correlation')
        axe.plot(corr2[:, :], 'k.')
        # axe.set_title('Correlation, solver '+self.output.name_solver +
        #               ', nh = {0:5d}'.format(self.nx))
        axe.hold(True)
        fig, ax = self.output.figure_axe(numfig=2333)
        self.ax = ax
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Frequency')
        pc = ax.pcolormesh(fx, fy, corr[4, :, :])
        fig.colorbar(pc)
        ax.axis([0, delta_frequency*self.nb_times_compute/2,
                 0, delta_frequency*self.nb_times_compute/2])
        ax.hold(True)

        fig1, ax1 = self.output.figure_axe(numfig=2334)
        self.ax1 = ax1
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Frequency')
        pc1 = ax1.pcolormesh(fx, fy, corr[3, :, :])
        fig1.colorbar(pc1)
        ax1.axis([0, delta_frequency*self.nb_times_compute/2,
                 0, delta_frequency*self.nb_times_compute/2])
        ax1.hold(True)

    def plot_corr4(self):
        import matplotlib.pyplot as plt

        plt.close('all')

        f = h5py.File(self.path_file4, 'r')
        corr4_full = f['corr4']
        corr2_full = f['corr2']
        corr4 = corr4_full[-1]
        corr2 = corr2_full[-1]

        size_io = self.iomegas1.shape[0]
        nb_omegas = self.nb_omegas
        duration = self.nb_times_compute*self.sim.time_stepping.deltat
        delta_frequency = 1./duration
        fy, fx = np.mgrid[slice(0, delta_frequency*(self.nb_times_compute/2+1),
                                delta_frequency),
                          slice(0, delta_frequency*(self.nb_times_compute/2+1),
                                delta_frequency)]

        corr = np.empty(corr4.shape)

        for i1, io1 in enumerate(self.iomegas1):
            for io3 in range(nb_omegas):
                for io4 in range(io3+1):
                    io2 = io3 + io4 - io1
                    if io2 < 0:
                        io2 = -io2
                    elif io2 >= nb_omegas:
                        io2 = 2*nb_omegas-1-io2
                    corr[i1, io3, io4] = corr4[i1, io3, io4]/np.sqrt(
                        corr2[io1, io1] * corr2[io3, io3] * corr2[io4, io4] *
                        corr2[io2, io2])
                    corr[i1, io4, io3] = corr[i1, io3, io4]

        plt.figure(num=21)
        for i1, io1 in enumerate(self.iomegas1):
            plt.subplot(int(np.sqrt(size_io)), int(round(size_io /
                        float(int(np.sqrt(size_io)))+0.5)), i1+1)
            plt.xlabel('Frequency')
            plt.ylabel('Frequency')
            plt.pcolormesh(fx, fy, corr[i1, :, :],
                           vmin=corr.min(), vmax=corr.max())
            plt.colorbar()
            plt.axis([fx.min(), fx.max(), fy.min(), fy.max()])
            
        plt.figure(num=22)
        plt.pcolormesh(fx, fy, corr2[:, :], vmin=corr2.min(), vmax=10*corr2.min())
        plt.colorbar()
        plt.axis([fx.min(), fx.max(), fy.min(), fy.max()])
    def _compute_correl4(self, q_fftt):
        r"""Compute the correlations 4.

        .. math::
           C_4(\omega_1, \omega_2, \omega_3, \omega_4) =
           \langle
           \tilde w(\omega_1, \mathbf{x})
           \tilde w(\omega_2, \mathbf{x})
           \tilde w(\omega_3, \mathbf{x})^*
           \tilde w(\omega_4, \mathbf{x})^*
           \rangle_\mathbf{x},

        where

        .. math::
           \omega_2 = \omega_3 + \omega_4 - \omega_1

        and :math:`\omega_1 > 0`, :math:`\omega_3 > 0` and

        :math:`\omega_4 > 0`. Thus, this function produces an array
        :math:`C_4(\omega_1, \omega_3, \omega_4)`.

        """

        q_fftt_conj = q_fftt.conj()

        nb_omegas = self.nb_omegas

        corr4 = np.empty([len(self.iomegas1), nb_omegas, nb_omegas])
        for i1, io1 in enumerate(self.iomegas1):
            # this loop could be parallelized (OMP)
            for io3 in range(nb_omegas):
                # we use the symmetry omega_3 <--> omega_4
                for io4 in range(io3+1):
                    tmp = (q_fftt[:, io1] *
                           q_fftt_conj[:, io3] *
                           q_fftt_conj[:, io4])
                    io2 = io3 + io4 - io1
                    if io2 < 0:
                        io2 = -io2
                        corr4[i1, io3, io4] = np.sum(
                            np.absolute(tmp*q_fftt_conj[:, io2]))
                    elif io2 >= nb_omegas:
                        io2 = 2*nb_omegas-1-io2
                        corr4[i1, io3, io4] = np.sum(
                            np.absolute(tmp*q_fftt_conj[:, io2]))
                    else:
                        corr4[i1, io3, io4] = np.sum(
                            np.absolute(tmp*q_fftt[:, io2]))
                # symmetry omega_3 <--> omega_4:
                    corr4[i1, io4, io3] = corr4[i1, io3, io4]

        if mpi.nb_proc > 1:
            # reduce SUM for mean:
            corr4 = mpi.comm.reduce(corr4, op=mpi.MPI.SUM, root=0)

        if mpi.rank == 0:
            corr4 /= self.nb_xs_seq
            return corr4

    def _compute_correl2(self, q_fftt):
        r"""Compute the correlations 2.

        .. math::
           C_2(\omega_1, \omega_2) =
           \langle
           \tilde w(\omega_1, \mathbf{x})
           \tilde w(\omega_2, \mathbf{x})^*
           \rangle_\mathbf{x}.

        """
#        where :math:`\omega_1 = \omega_2`. Thus, this function
#        produces an array :math:`C_2(\omega)`.

        nb_omegas = self.nb_omegas

        corr2 = np.empty([nb_omegas, nb_omegas])

        q_fftt_conj = q_fftt.conj()
        for io3 in range(nb_omegas):
            for io4 in range(io3+1):
                tmp = (q_fftt[:, io3] *
                       q_fftt_conj[:, io4])
                corr2[io3, io4] = np.sum(np.absolute(tmp))
                corr2[io4, io3] = corr2[io3, io4]

        if mpi.nb_proc > 1:
            # reduce SUM for mean:
            corr2 = mpi.comm.reduce(corr2, op=mpi.MPI.SUM, root=0)

        if mpi.rank == 0:
            corr2 /= self.nb_xs_seq
            return corr2
