"""
Correl freq (:mod:`fluidsim.solvers.plate2d.output.correlations_freq`)
============================================================================

.. currentmodule:: fluidsim.solvers.plate2d.output.correlations_freq

Provides:

.. autoclass:: CorrelationsFreq
   :members:
   :private-members:

"""

from __future__ import division, print_function

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from fluiddyn.util import mpi

from fluidsim.base.output.base import SpecificOutput
from fluidsim.operators.fft.easypyfft import FFTW1DReal2Complex
from fluidsim.operators.miscellaneous import compute_correl4, compute_correl2


class CorrelationsFreq(SpecificOutput):
    """Compute, save, load and plot correlations of frequencies.

    """

    _tag = 'correl_freq'
    _name_file = _tag + '.h5'

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
        pcorrel_freq = params.output.correl_freq
        super(CorrelationsFreq, self).__init__(
            output,
            period_save=params.output.periods_save.correl_freq,
            has_to_plot_saved=pcorrel_freq.HAS_TO_PLOT_SAVED)

        self.nb_times_compute = pcorrel_freq.nb_times_compute
        self.coef_decimate = pcorrel_freq.coef_decimate
        self.key_quantity = pcorrel_freq.key_quantity
        self.iomegas1 = np.array(pcorrel_freq.iomegas1)
        self.it_last_run = pcorrel_freq.it_start
        n0 = len(range(0, output.sim.oper.shapeX_loc[0], self.coef_decimate))
        n1 = len(range(0, output.sim.oper.shapeX_loc[1], self.coef_decimate))
        nb_xs = n0 * n1

        self.spatio_temp = np.empty([nb_xs, self.nb_times_compute])
        self.oper_fft1 = FFTW1DReal2Complex(self.spatio_temp.shape)
        self.nb_omegas = self.oper_fft1.shapeK[-1]
        self.hamming = np.hanning(self.nb_times_compute)

        if mpi.nb_proc > 1:
            nb_xs = mpi.comm.reduce(nb_xs, op=mpi.MPI.SUM, root=0)
        if mpi.rank > 0:
            nb_xs = 0
        self.nb_xs_seq = nb_xs

        self.nb_times_in_spatio_temp = 0

        if os.path.exists(self.path_file):
            with h5py.File(self.path_file, 'r') as f:
                link_corr4 = f['corr4']
                link_corr2 = f['corr2']
                link_nb_means = f['nb_means']
                self.corr4 = link_corr4[-1]
                self.corr2 = link_corr2[-1]
                self.nb_means_times = link_nb_means[-1]
                self.periods_fill = f['periods_fill'][...]
                if self.sim.time_stepping.deltat != f['deltat'][...]:
                    raise ValueError()
        else:
            self.periods_fill = params.output.periods_save.correl_freq
            self.corr4 = np.zeros([len(self.iomegas1),
                                   self.nb_omegas, self.nb_omegas],
                                  dtype=np.complex128)
            self.corr2 = np.zeros([self.nb_omegas, self.nb_omegas],
                                  dtype=np.complex128)
            self.nb_means_times = 0

        if self.periods_fill > 0:
            dt_output = self.periods_fill * output.sim.time_stepping.deltat
            duration = self.nb_times_compute * dt_output
            delta_omega = 2*np.pi/duration
            self.omegas = delta_omega * np.arange(self.nb_omegas)

            self.omega_Nyquist = np.pi/dt_output

            omega1_max = self.iomegas1.max() * delta_omega
            if self.omega_Nyquist <= omega1_max:
                raise ValueError('omega_1 max is larger than omega_Nyquist.')

            self.omega_dealiasing = (
                self.params.oper.coef_dealiasing * np.pi *
                self.params.oper.nx / self.params.oper.Lx)**2

            if self.omega_dealiasing > self.omega_Nyquist:
                print('Warning: omega_dealiasing > omega_Nyquist')

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
            self.path_file, correlations, dico_arrays_1time)

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
                spatio_fft = self.oper_fft1.fft(self.hamming*self.spatio_temp)
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

                    if ((self.nb_means_times % 128 == 0 or
                         np.log(self.nb_means_times)/np.log(2) % 1 == 0) and
                            self.nb_means_times != 1):

                        correlations = {'corr4': self.corr4,
                                        'corr2': self.corr2,
                                        'nb_means': self.nb_means_times}
                        if not os.path.exists(self.path_file):
                            self.init_files2(correlations)
                        else:
                            # save the spectra in the file correl_freq.h5
                            self.add_dico_arrays_to_file(self.path_file,
                                                         correlations)
                        if self.has_to_plot:
                            self._online_plot(correlations)
                    #     if (tsim-self.t_last_show >= self.period_show):
                    #         self.t_last_show = tsim
                    #         self.axe.get_figure().canvas.draw()

    def init_online_plot(self):
        fig, axe = self.output.figure_axe(numfig=4100000)
        self.axe = axe
        axe.set_xlabel('Omega')
        axe.set_ylabel('Correlations')
        axe.set_title('Correlation, solver '+self.output.name_solver +
                      ', nh = {0:5d}'.format(self.params.oper.nx))
        axe.hold(True)

    def _online_plot(self, dico_results):
        nb_omegas = self.nb_omegas

        corr4 = dico_results['corr4']
        corr2 = dico_results['corr2']
        corr = np.empty(corr4.shape)
        fy, fx = np.meshgrid(self.omegas, self.omegas)
        for i1, io1 in enumerate(self.iomegas1):
            for io3 in range(nb_omegas):
                for io4 in range(io3+1):
                    io2 = io3 + io4 - io1
                    if io2 < 0:
                        io2 = -io2
                    elif io2 >= nb_omegas:
                        io2 = 2*nb_omegas-1-io2
                    corr[i1, io3, io4] = corr4[i1, io3, io4]/np.sqrt(abs(
                        corr2[io1, io1] * corr2[io3, io3] * corr2[io4, io4] *
                        corr2[io2, io2]))
                    corr[i1, io4, io3] = corr[i1, io3, io4]

        fig, axe = self.output.figure_axe(numfig=4100000)
        self.axe = axe
        axe.set_xlabel('Omega')
        axe.set_xlabel('Correlation')
        axe.plot(corr2[:, :], 'k.')
        # axe.set_title('Correlation, solver '+self.output.name_solver +
        #               ', nh = {0:5d}'.format(self.nx))
        axe.hold(True)
        fig, ax = self.output.figure_axe(numfig=2333)
        self.ax = ax
        ax.set_xlabel('Omega')
        ax.set_ylabel('Omega')
        pc = ax.pcolormesh(fx, fy, abs(corr[4, :, :]))
        fig.colorbar(pc)
        ax.hold(True)

        fig1, ax1 = self.output.figure_axe(numfig=2334)
        self.ax1 = ax1
        ax1.set_xlabel('Omega')
        ax1.set_ylabel('Omega')
        pc1 = ax1.pcolormesh(fx, fy, corr[3, :, :])
        fig1.colorbar(pc1)
        ax1.hold(True)

    def compute_corr4_norm(self, it=-1):

        with h5py.File(self.path_file, 'r') as f:
            corr4 = f['corr4'][it]
            corr2 = f['corr2'][it]
            nb_means = f['nb_means'][it]

        nb_omegas = self.nb_omegas

        corr_norm = np.empty_like(corr4)
        cum_norm = np.empty(corr4.shape)
        norm = np.empty(corr4.shape)
        tmp1 = np.empty((nb_omegas, nb_omegas), dtype=np.complex128)
        tmp2 = np.empty((nb_omegas, nb_omegas), dtype=np.complex128)
        for i1, io1 in enumerate(self.iomegas1):
            for io3 in range(nb_omegas):
                for io4 in range(io3+1):
                    io2 = io3 + io4 - io1
                    if io2 < 0:
                        io2 = -io2
                    elif io2 >= nb_omegas:
                        io2 = 2*nb_omegas-1-io2

                    norm[i1, io3, io4] = np.sqrt(abs(
                        corr2[io1, io1] * corr2[io3, io3] *
                        corr2[io4, io4] * corr2[io2, io2]))
                    norm[i1, io4, io3] = norm[i1, io3, io4]

                    tmp1[io3, io4] = corr2[io4, io2].conj() * corr2[io1, io3]
                    tmp2[io3, io4] = corr2[io1, io4] * corr2[io3, io2].conj()

                    cum_norm[i1, io3, io4] = abs(
                        corr4[i1, io3, io4] - tmp1[io3, io4] - tmp2[io3, io4]
                    )/norm[i1, io3, io4]
                    cum_norm[i1, io4, io3] = cum_norm[i1, io3, io4]

                    corr_norm[i1, io3, io4] = abs(
                        corr4[i1, io3, io4]/norm[i1, io3, io4])
                    corr_norm[i1, io4, io3] = corr_norm[i1, io3, io4]

        return norm, corr_norm, cum_norm, nb_means

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
                            tmp*q_fftt_conj[:, io2])
                    elif io2 >= nb_omegas:
                        io2 = 2*nb_omegas-1-io2
                        corr4[i1, io3, io4] = np.sum(
                            tmp*q_fftt_conj[:, io2])
                    else:
                        corr4[i1, io3, io4] = np.sum(
                            tmp*q_fftt[:, io2])
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
        nb_omegas = self.nb_omegas
        corr2 = np.empty([nb_omegas, nb_omegas])

        q_fftt_conj = q_fftt.conj()
        for io3 in range(nb_omegas):
            for io4 in range(io3+1):
                corr2[io3, io4] = np.sum(q_fftt[:, io3] *
                                         q_fftt_conj[:, io4])
                corr2[io4, io3] = corr2[io3, io4].conj()

        if mpi.nb_proc > 1:
            # reduce SUM for mean:
            corr2 = mpi.comm.reduce(corr2, op=mpi.MPI.SUM, root=0)

        if mpi.rank == 0:
            corr2 /= self.nb_xs_seq
            return corr2

    def _compute_norm_pick_corr4_from_corr4(self, corr4, i1=0, delta_io=10):
        io1 = self.iomegas1[i1]
        io3 = 4*io1
        io4 = 8*io1
        delta_io = min(delta_io, io3)
        if corr4.ndim == 3:
            return np.abs(np.mean(
                corr4[i1,
                      io3-delta_io:io3+delta_io+1,
                      io4-delta_io:io4+delta_io+1]))
        elif corr4.ndim == 4:
            corr4_mini = corr4[:, i1,
                               io3-delta_io:io3+delta_io+1,
                               io4-delta_io:io4+delta_io+1]
            return np.abs(corr4_mini.mean(1).mean(1))

    def _compute_norm_pick_corr4(self):
        with h5py.File(self.path_file, 'r') as f:
            corr4 = f['corr4'][:]
            nb_means = f['nb_means'][:]
        fcorr4 = self._compute_norm_pick_corr4_from_corr4(corr4)

        return nb_means, fcorr4

    def _compute_dnormpickC4_over_dnbmean(self):
        with h5py.File(self.path_file, 'r') as f:
            corr4 = f['corr4'][:]
            nb_means = f['nb_means'][:]

        fcorr4 = self._compute_norm_pick_corr4_from_corr4(corr4)

        return (nb_means,
                np.absolute(np.diff(fcorr4) / np.diff(nb_means) *
                            nb_means[1:] / fcorr4[1:]))

    def plot_norm_pick_corr4(self):
        nb_means, fcorr4 = self._compute_norm_pick_corr4()
        plt.figure()
        ax = plt.gca()
        ax.loglog(nb_means, fcorr4, 'x-')
        ax.set_ylabel('a kind of norm of corr4')
        ax.set_xlabel('number of averages')

    def plot_convergence(self):

        nb_means, dnormpickC4 = self._compute_dnormpickC4_over_dnbmean()

        fig = plt.figure()
        fig.suptitle('"convergence"')
        ax = plt.gca()
        ax.loglog(nb_means[1:], dnormpickC4, 'x-')
        ax.set_xlabel('number of averages')

    def plot_corr2(self, nonorm=False, it=-1):

        with h5py.File(self.path_file, 'r') as f:
            corr2_in_file = f['corr2']
            corr2 = corr2_in_file[it]
            nb_means = f['nb_means'][it]

        fy, fx = np.meshgrid(self.omegas, self.omegas)

        if nonorm:
            corr2_norm = corr2
        else:
            corr2_norm = np.empty((self.nb_omegas, self.nb_omegas),
                                  dtype=np.complex128)
            for io3 in range(self.nb_omegas):
                for io4 in range(io3+1):
                    corr2_norm[io3, io4] = corr2[io3, io4]/np.sqrt(
                        np.absolute(corr2[io3, io3] *
                                    corr2[io4, io4]))
                    corr2_norm[io4, io3] = corr2_norm[io3, io4].conj()

        log10corr2 = np.log10(abs(corr2_norm))
        plt.figure()
        ax = plt.gca()
        ax.set_title('log10(abs(corr2)); nb_means: ' +
                     str(nb_means))
        plt.xlabel('Omega')
        plt.ylabel('Omega')
        plt.pcolormesh(fx, fy, log10corr2, vmin=log10corr2.min(),
                       vmax=log10corr2.max())
        plt.colorbar()
        plt.axis([fx.min(), fx.max(), fy.min(), fy.max()])

    def plot_corr2_1d(self, it=-1):

        with h5py.File(self.path_file, 'r') as f:
            corr2 = f['corr2'][it]
            nb_means = f['nb_means'][it]

        corr2_diag = np.empty(self.nb_omegas, dtype=np.complex128)
        for io3 in range(self.nb_omegas):
            corr2_diag[io3] = corr2[io3, io3]

        fig = plt.figure(num=18)
        fig.clf()
        ax = plt.gca()
        ax.set_title('abs(corr2_diag); nb_means: ' +
                     str(nb_means))
        plt.xlabel('Omega')
        plt.ylabel('abs(corr2)')
        #ax.loglog(self.omegas, abs(corr2_diag))
        ax.plot(self.omegas, np.log10(abs(corr2_diag)))

    def plot_corr4(self, it=-1):

        nb_omegas1 = self.iomegas1.shape[0]
        nb_omegas = self.nb_omegas

        corr_norm = np.empty((nb_omegas1, nb_omegas, nb_omegas))
        cum_norm = np.empty(corr_norm.shape)
        norm = np.empty(corr_norm.shape)
        norm, corr_norm, cum_norm, nb_means = self.compute_corr4_norm(it)

        fy, fx = np.meshgrid(self.omegas, self.omegas)

        fig = plt.figure()
        fig.suptitle('log10(abs(corr_norm)); nb_means: ' +
                     str(nb_means))
        nb_subplot_vert = int(np.sqrt(nb_omegas1))
        nb_subplot_horiz = int(round(nb_omegas1/nb_subplot_vert))
        for i1, io1 in enumerate(self.iomegas1):
            plt.subplot(nb_subplot_vert, nb_subplot_horiz, i1+1)
            plt.xlabel('Omega')
            plt.ylabel('Omega')
            plt.pcolormesh(fx, fy, np.log10(abs(corr_norm[i1, :, :])),
                           vmin=np.log10(abs(corr_norm.min())),
                           vmax=np.log10(abs(corr_norm.max())))
            plt.colorbar()
            plt.axis([fx.min(), fx.max(), fy.min(), fy.max()])

        fig = plt.figure()
        fig.suptitle('log10(abs(cum_norm)); nb_means: ' +
                     str(nb_means))
        for i1, io1 in enumerate(self.iomegas1):
            plt.subplot(nb_subplot_vert, nb_subplot_horiz, i1+1)
            plt.xlabel('Omega')
            plt.ylabel('Omega')
            plt.pcolormesh(fx, fy, np.log10(abs(cum_norm[i1, :, :])),
                           vmin=np.log10(abs(cum_norm.min())),
                           vmax=np.log10(abs(cum_norm.max())))
            plt.colorbar()

            plt.axis([fx.min(), fx.max(), fy.min(), fy.max()])
        fig = plt.figure()
        fig.suptitle('log10(abs(norm)); nb_means: ' +
                     str(nb_means))
        for i1, io1 in enumerate(self.iomegas1):
            plt.subplot(nb_subplot_vert, nb_subplot_horiz, i1+1)
            plt.xlabel('Omega')
            plt.ylabel('Omega')
            plt.pcolormesh(fx, fy, np.log10(abs(norm[i1, :, :])),
                           vmin=np.log10(abs(norm.min())),
                           vmax=np.log10(abs(norm.max())))
            plt.colorbar()
            plt.axis([fx.min(), fx.max(), fy.min(), fy.max()])

    def plot_convergence2(self):
        ws = 10
        with h5py.File(self.path_file, 'r') as f:
            corr4_full = f['corr4']
            nb_means = f['nb_means']
            means = np.empty(corr4_full.shape[0])
            f_conv = np.empty(corr4_full.shape[0:2])
            for inb in range(corr4_full.shape[0]):
                corr4_nb = corr4_full[inb]
                means[inb] = nb_means[inb]
                for i1, io1 in enumerate(self.iomegas1):
                    if (2*io1+ws < self.nb_omegas) & (2*io1-ws > io1):
                        f_conv[inb, i1] = np.abs(np.mean(
                            corr4_nb[i1, 2*io1-ws:2*io1+ws+1,
                                     2*io1-ws:2*io1+ws+1]))
        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel('nb_means')
        # ax1.set_ylabel('convergence')
        ax1.set_title('Trispectra Convergence')
        ax1.hold(True)
        for i1, io1 in enumerate(self.iomegas1):
            f_conv[:, i1] = f_conv[:, i1]/f_conv[-1, i1]
            ax1.plot(means, f_conv[:, i1], linewidth=1)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
