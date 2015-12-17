import h5py

import os
import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.output.base import SpecificOutput

from fluidsim.operators.fft import easypyfft


class TimeSignalsK(SpecificOutput):
    """A :class:`TimeSignalK` object handles the saving of time signals
    in spectral space.

    This class uses the particular functions defined by some solvers
    :func:`linear_eigenmode_from_values_1k` and
    :func`omega_from_wavenumber`.
    """

    _tag = 'time_signals_fft'
    _name_file = 'time_sigK.h5'

    @staticmethod
    def _complete_params_with_default(params):
        tag = 'time_signals_fft'

        params.output.periods_save._set_attrib(tag, 0)
        params.output.periods_plot._set_attrib(tag, 0)

        params.output._set_child(tag,
                                 attribs={'nb_shells_time_sigK': 4,
                                          'nb_k_per_shell_time_sigK': 4})

    def __init__(self, output):
        self.output = output
        sim = output.sim
        params = sim.params

        self.params = params
        self.c2 = params.c2
        self.f = params.f
        self.nx = params.oper.nx

        if not params.output.HAS_TO_SAVE:
            params.output.periods_save.time_signals_fft = False

        if params.output.periods_save.time_signals_fft:
            self._init_save(sim)

        super(TimeSignalsK, self).__init__(
            output,
            period_save=params.output.periods_save.time_signals_fft,
            period_plot=params.output.periods_plot.time_signals_fft)

    def _init_save(self, sim):
        """
        Sets the attribs determining how many wavenumbers are selected from
        each shell, for which the time series is saved.
        """
        params = self.params
        self.nb_shells = params.output.time_signals_fft.nb_shells_time_sigK
        self.nb_k_per_shell = \
            params.output.time_signals_fft.nb_k_per_shell_time_sigK
        self.nb_k_tot = self.nb_shells*self.nb_k_per_shell

        i_shift = 3
        deltalogk = np.log(
            params.oper.nx / 2 *
            params.oper.coef_dealiasing)/(self.nb_shells+i_shift)

        deltakh = sim.oper.deltakh

        self.kh_shell = deltakh*np.exp(
            deltalogk*np.arange(i_shift, self.nb_shells+i_shift))

        self.kh_shell = deltakh*np.round(self.kh_shell/deltakh)

        for i_s in xrange(1, self.nb_shells):
            if self.kh_shell[i_s-1] == self.kh_shell[i_s]:
                self.kh_shell[i_s] += deltakh

        # if mpi.rank == 0:
        #     print 'self.kh_shell/deltakh'
        #     print self.kh_shell/sim.oper.deltakh

        # hypothese dispersion relation only function of the module
        # of the wavenumber ("shells")
        self.omega_shell = self.output.omega_from_wavenumber(self.kh_shell)

        kx_array_ik_approx = np.empty([self.nb_k_tot])
        ky_array_ik_approx = np.empty([self.nb_k_tot])

        delta_angle = np.pi/(self.nb_k_per_shell-1)
        for ishell, kh_s in enumerate(self.kh_shell):
            angle = -np.pi/2
            for ikps in xrange(self.nb_k_per_shell):
                kx_array_ik_approx[ishell*self.nb_shells+ikps] = \
                    kh_s*np.cos(angle)
                ky_array_ik_approx[ishell*self.nb_shells+ikps] = \
                    kh_s*np.sin(angle)
                angle += delta_angle

        self.ik0_array_ik = np.empty([self.nb_k_tot], dtype=np.int32)
        self.ik1_array_ik = np.empty([self.nb_k_tot], dtype=np.int32)
        if mpi.nb_proc > 1:
            self.rank_array_ik = np.empty([self.nb_k_tot], dtype=np.int32)

        for ik in xrange(self.nb_k_tot):
            kx_approx = kx_array_ik_approx[ik]
            ky_approx = ky_array_ik_approx[ik]
            rank_ik, ik0, ik1 = \
                sim.oper.where_is_wavenumber(kx_approx, ky_approx)
            if mpi.nb_proc > 1:
                self.rank_array_ik[ik] = rank_ik
            self.ik0_array_ik[ik] = ik0
            self.ik1_array_ik[ik] = ik1

        self.kx_array_ik = np.empty([self.nb_k_tot])
        self.ky_array_ik = np.empty([self.nb_k_tot])

        for ik in xrange(self.nb_k_tot):
            ik0_ik = self.ik0_array_ik[ik]
            ik1_ik = self.ik1_array_ik[ik]

            if mpi.nb_proc > 1:
                rank_ik = self.rank_array_ik[ik]
            else:
                rank_ik = 0

            if mpi.rank == rank_ik:
                kx_1k = sim.oper.KX[ik0_ik, ik1_ik]
                ky_1k = sim.oper.KY[ik0_ik, ik1_ik]

            if rank_ik != 0:
                if mpi.rank == rank_ik:
                    data = np.array([kx_1k, ky_1k])
                    mpi.comm.Send(
                        [data, mpi.MPI.DOUBLE],
                        dest=0, tag=ik)
                elif mpi.rank == 0:
                    data = np.empty([2], np.float64)
                    mpi.comm.Recv(
                        [data, mpi.MPI.DOUBLE],
                        source=rank_ik, tag=ik)
                    kx_1k = data[0]
                    ky_1k = data[1]

            if mpi.rank == 0:
                self.kx_array_ik[ik] = kx_1k
                self.ky_array_ik[ik] = ky_1k

        if mpi.rank == 0:
            self.kh_array_ik = np.sqrt(self.kx_array_ik**2 +
                                       self.ky_array_ik**2)

            self.omega_array_ik = self.output.omega_from_wavenumber(
                self.kh_array_ik)

            self.period_save = np.pi/(8*self.omega_array_ik.max())
        else:
            self.period_save = 0.

        if mpi.nb_proc > 1:
            self.period_save = mpi.comm.bcast(self.period_save)

    def init_files(self, dico_arrays_1time=None):
        if (not os.path.exists(self.path_file)):
            dico_results = self.compute()
            if mpi.rank == 0:
                dico_arrays_1time = {
                    'kh_shell': self.kh_shell,
                    'omega_shell': self.omega_shell,
                    'kx_array_ik': self.kx_array_ik,
                    'ky_array_ik': self.ky_array_ik,
                    'kh_array_ik': self.kh_array_ik,
                    'omega_array_ik': self.omega_array_ik}
                self.create_file_from_dico_arrays(
                    self.path_file, dico_results, dico_arrays_1time)

        if mpi.rank == 0:
            self.file = h5py.File(self.path_file, 'r+')
            self.file.attrs['nb_shells'] = self.nb_shells
            self.file.attrs['nb_k_per_shell'] = self.nb_k_per_shell
            self.file.attrs['nb_k_tot'] = self.nb_k_tot
            # the file is kept open during all the simulation
            self.nb_saved_times = 1

        self.t_last_save = self.sim.time_stepping.t

    def online_save(self):
        """Save the values at one time. """
        tsim = self.sim.time_stepping.t
        if (tsim-self.t_last_save >= self.period_save):
            self.t_last_save = tsim
            dico_results = self.compute()
            if mpi.rank == 0:
                self.add_dico_arrays_to_open_file(self.file,
                                                  dico_results,
                                                  self.nb_saved_times)
                self.nb_saved_times += 1

    def compute(self):
        """compute the values at one time."""

        ux_fft = self.sim.state('ux_fft')
        uy_fft = self.sim.state('uy_fft')
        eta_fft = self.sim.state('eta_fft')

        if mpi.rank == 0:
            q_array_ik = np.empty([self.nb_k_tot], dtype=np.complex128)
            d_array_ik = np.empty([self.nb_k_tot], dtype=np.complex128)
            a_array_ik = np.empty([self.nb_k_tot], dtype=np.complex128)

        for ik in xrange(self.nb_k_tot):
            ik0_ik = self.ik0_array_ik[ik]
            ik1_ik = self.ik1_array_ik[ik]

            if mpi.rank == 0:
                kx_ik = self.kx_array_ik[ik]
                ky_ik = self.ky_array_ik[ik]

            if mpi.nb_proc > 1:
                rank_ik = self.rank_array_ik[ik]
            else:
                rank_ik = 0

            if mpi.rank == rank_ik:
                ux_1k = ux_fft[ik0_ik, ik1_ik]
                uy_1k = uy_fft[ik0_ik, ik1_ik]
                eta_1k = eta_fft[ik0_ik, ik1_ik]

            if rank_ik != 0:
                if mpi.rank == rank_ik:
                    data = np.array([ux_1k, uy_1k, eta_1k])
                    mpi.comm.Send(
                        [data, mpi.MPI.COMPLEX],
                        dest=0, tag=ik)
                elif mpi.rank == 0:
                    data = np.empty([3], np.complex128)
                    mpi.comm.Recv(
                        [data, mpi.MPI.COMPLEX],
                        source=rank_ik, tag=ik)
                    ux_1k = data[0]
                    uy_1k = data[1]
                    eta_1k = data[2]

            if mpi.rank == 0:
                q_1k, d_1k, a_1k = (
                    self.output.linear_eigenmode_from_values_1k(
                        ux_1k, uy_1k, eta_1k, kx_ik, ky_ik))
                q_array_ik[ik] = q_1k
                d_array_ik[ik] = d_1k
                a_array_ik[ik] = a_1k

        if mpi.rank == 0:
            dico_results = {'q_array_ik': q_array_ik,
                            'd_array_ik': d_array_ik,
                            'a_array_ik': a_array_ik}
            return dico_results

    def load(self):

        if (not os.path.exists(self.path_file)):
            raise ValueError(
                'no file time_sigK.h5 in\n'+self.output.dir_save_run)

        with h5py.File(self.path_file, 'r+') as f:

            dset_times = f['times']
            times = dset_times[...]

            dico_results = {}
            dico_results['times'] = times

            dico_results['nb_shells'] = f.attrs['nb_shells']
            dico_results['nb_k_per_shell'] = f.attrs['nb_k_per_shell']
            dico_results['nb_k_tot'] = f.attrs['nb_k_tot']

            keys_1time = [
                'kh_shell',
                'omega_shell',
                'kx_array_ik',
                'ky_array_ik',
                'kh_array_ik',
                'omega_array_ik']

            for key in keys_1time:
                dset_temp = f[key]
                dico_results[key] = dset_temp[...]

            keys_linear_eigenmodes = \
                self.sim.info.solver.classes.State.keys_linear_eigenmodes

            for key in keys_linear_eigenmodes:
                dset_temp = f[key[:-3]+'array_ik']
                A = dset_temp[...]
                dico_results['sig_'+key] = np.ascontiguousarray(A.transpose())
        return dico_results

    def plot(self):
        dico_results = self.load()

        t = dico_results['times']

        nb_shells = dico_results['nb_shells']
        nb_k_per_shell = dico_results['nb_k_per_shell']

        sig_q_fft = dico_results['sig_q_fft']
        sig_a_fft = dico_results['sig_a_fft']
        sig_d_fft = dico_results['sig_d_fft']

        kh_shell = dico_results['kh_shell']
        omega_shell = dico_results['omega_shell']
        period_shell = 2*np.pi/omega_shell

        for ish in xrange(nb_shells):

            fig, ax1 = self.output.figure_axe()
            ax1.set_xlabel('$t/T$')
            ax1.set_ylabel('signals (s$^{-1}$)')
            title = (
                'signals eigenmodes, ikh = {0:.2f}, solver '.format(
                    kh_shell[ish]/sim.oper.deltakh) +
                self.output.name_solver +
                ', nh = {0:5d}'.format(self.nx) +
                ', c2 = {0:.4g}, f = {1:.4g}'.format(self.c2, self.f)
                )
            ax1.set_title(title)
            ax1.hold(True)

            coef_norm_a = self.c2/omega_shell[ish]

            T = period_shell[ish]

            for ikps in xrange(nb_k_per_shell):
                isig = ish*nb_k_per_shell+ikps

                ax1.plot(t/T, sig_q_fft[isig].real, 'k', linewidth=1)
                ax1.plot(t/T, coef_norm_a*sig_a_fft[isig].real,
                         'c', linewidth=1)
                ax1.plot(t/T, sig_d_fft[isig].real, 'y', linewidth=1)

        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel('$\omega$')
        ax1.set_ylabel('kh_shell')
        ax1.loglog(kh_shell, omega_shell, 'o', linewidth=2)

    def time_spectrum(self, sig_long):

        Nt = sig_long.size
        stepit0 = int(np.fix(self.nt/2))

        nb_spectra = 0
        it0 = 0
        spect = np.zeros([self.nt/2+1])
        while it0+self.nt < Nt:
            nb_spectra += 1
            sig = sig_long[it0:it0+self.nt]
            spect_raw = (
                abs(self.opfft1d.fft(self.hann*sig))**2 / 2 / self.deltaomega)
            spect += spect_raw[:self.nt/2+1]
            if self.nt % 2 == 0:
                spect[1:self.nt/2] += spect_raw[self.nt-1:self.nt/2:-1]
            else:
                spect[1:self.nt/2+1] += spect_raw[self.nt-1:self.nt/2:-1]
            it0 += stepit0

        return spect/nb_spectra

    def compute_spectra(self):
        dico_results = self.load()

        t = dico_results['times']
        Nt = t.size
        nt = 2**int(np.fix(np.log2(Nt/10)))
        # if nt%2 == 1:
        #     nt -= 1
        self.nt = nt

        if not hasattr(self, 'opfft1d'):
            self.opfft1d = easypyfft.FFTW1D(nt)

        T = t[nt-1] - t[0]
        # deltat = T/nt
        self.deltaomega = 2*np.pi/T
        # self.omega = self.deltaomega*np.concatenate(
        #     (np.arange(nt/2+1), np.arange(-nt/2+1, 0)))

        self.omega = self.deltaomega*np.arange(nt/2+1)

        self.hann = np.hanning(nt)

        nb_shells = dico_results['nb_shells']
        nb_k_per_shell = dico_results['nb_k_per_shell']
        # nb_k_tot = dico_results['nb_k_tot']

        sig_q_fft = dico_results['sig_q_fft']
        sig_a_fft = dico_results['sig_a_fft']
        sig_d_fft = dico_results['sig_d_fft']

        # kh_shell = dico_results['kh_shell']
        omega_shell = dico_results['omega_shell']
        # period_shell = 2*np.pi/omega_shell

        time_spectra_q = np.zeros([nb_shells, nt/2+1])
        time_spectra_a = np.zeros([nb_shells, nt/2+1])
        time_spectra_d = np.zeros([nb_shells, nt/2+1])

        for ish in xrange(nb_shells):
            coef_norm_a = self.c2/omega_shell[ish]
            for ikps in xrange(nb_k_per_shell):
                isig = ish*nb_k_per_shell+ikps
                sig_a_fft[isig] *= coef_norm_a
                time_spectra_q[ish] += self.time_spectrum(sig_q_fft[isig])
                time_spectra_a[ish] += self.time_spectrum(sig_a_fft[isig])
                time_spectra_d[ish] += self.time_spectrum(sig_d_fft[isig])

        time_spectra_q /= nb_k_per_shell
        time_spectra_a /= nb_k_per_shell
        time_spectra_d /= nb_k_per_shell

        dico_spectra = {
            'omega': self.omega,
            'time_spectra_q': time_spectra_q,
            'time_spectra_a': time_spectra_a,
            'time_spectra_d': time_spectra_d}
        return dico_spectra, dico_results

    def plot_spectra(self):
        dico_spectra, dico_results = self.compute_spectra()

        omega = dico_spectra['omega']
        time_spectra_q = dico_spectra['time_spectra_q']
        time_spectra_a = dico_spectra['time_spectra_a']
        time_spectra_d = dico_spectra['time_spectra_d']
        omega_shell = dico_results['omega_shell']

        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel('r$\omega/\omega_{lin}$')
        ax1.set_ylabel('r$E(\omega)$)')
        title = (
            'time spectra, solver ' +
            self.output.name_solver +
            ', nh = {0:5d}'.format(self.nx) +
            ', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f)
            )
        ax1.set_title(title)
        ax1.hold(True)

        nb_shells = dico_results['nb_shells']
        for ish in xrange(nb_shells):
            ax1.loglog(omega/omega_shell[ish],
                       time_spectra_q[ish], 'k', linewidth=1)
            ax1.loglog(omega/omega_shell[ish],
                       time_spectra_a[ish], 'b', linewidth=1)
            ax1.loglog(omega/omega_shell[ish],
                       time_spectra_d[ish], 'r', linewidth=1)

    def close_file(self):
        try:
            self.file.close()
        except AttributeError:
            pass


# if __name__ == '__main__':
#     pass
