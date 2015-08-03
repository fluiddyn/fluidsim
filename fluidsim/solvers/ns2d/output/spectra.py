"""Spectra output (:mod:`fluidsim.solvers.ns2d.output.spectra`)
===============================================================

.. autoclass:: SpectraNS2D
   :members:
   :private-members:

"""

import h5py

import numpy as np

from fluidsim.base.output.spectra import Spectra


class SpectraNS2D(Spectra):
    """Save and plot spectra."""

    def compute(self):
        """compute the values at one time."""
        energy_fft = self.output.compute_energy_fft()
        # compute the spectra 1D
        spectrum1Dkx_E, spectrum1Dky_E = self.spectra1D_from_fft(energy_fft)
        dico_spectra1D = {'spectrum1Dkx_E': spectrum1Dkx_E,
                          'spectrum1Dky_E': spectrum1Dky_E}
        # compute the spectra 2D
        spectrum2D_E = self.spectrum2D_from_fft(energy_fft)
        dico_spectra2D = {'spectrum2D_E': spectrum2D_E}
        return dico_spectra1D, dico_spectra2D

    def _online_plot(self, dico_spectra1D, dico_spectra2D):
        if (self.nx == self.params.oper.ny and
                self.params.oper.Lx == self.params.oper.Ly):
            spectrum2D = dico_spectra2D['spectrum2D_E']
            khE = self.oper.khE
            coef_norm = khE**(3.)
            self.axe.loglog(khE, spectrum2D*coef_norm, 'k')
            lin_inf, lin_sup = self.axe.get_ylim()
            if lin_inf < 10e-6:
                lin_inf = 10e-6
            self.axe.set_ylim([lin_inf, lin_sup])
        else:
            print('you need to implement the ploting '
                  'of the spectra for this case')

    def plot1d(self, tmin=0, tmax=1000, delta_t=2,
               coef_compensate=3):

        f = h5py.File(self.path_file1D, 'r')
        dset_times = f['times']

        dset_kxE = f['kxE']
        # dset_kyE = f['kyE']
        kh = dset_kxE[...]

        dset_spectrum1Dkx = f['spectrum1Dkx_E']
        dset_spectrum1Dky = f['spectrum1Dky_E']
        # nb_spectra = dset_times.shape[0]
        times = dset_times[...]
        # nt = len(times)

        delta_t_save = np.mean(times[1:]-times[0:-1])
        delta_i_plot = int(np.round(delta_t/delta_t_save))
        delta_t = delta_t_save*delta_i_plot
        if delta_i_plot == 0:
            delta_i_plot = 1

        imin_plot = np.argmin(abs(times-tmin))
        imax_plot = np.argmin(abs(times-tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]

        print(
            'plot1d(tmin={0}, tmax={1}, delta_t={2:.2f},'.format(
                tmin, tmax, delta_t) +
            ' coef_compensate={0:.3f})'.format(coef_compensate))

        print('''plot 1D spectra
tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}'''.format(
    tmin_plot, tmax_plot, delta_t,
    imin_plot, imax_plot, delta_i_plot))

        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel('$k_h$')
        ax1.set_ylabel('spectra')
        ax1.set_title('1D spectra, solver '+self.output.name_solver +
                      ', nh = {0:5d}'.format(self.nx))
        ax1.hold(True)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        coef_norm = kh**(coef_compensate)
        if delta_t != 0.:
            for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
                EK = (dset_spectrum1Dkx[it]+dset_spectrum1Dky[it])
                EK[EK < 10e-16] = 0.
                ax1.plot(kh, EK*coef_norm, 'k', linewidth=2)

        EK = (dset_spectrum1Dkx[imin_plot:imax_plot+1] +
              dset_spectrum1Dky[imin_plot:imax_plot+1]).mean(0)

        ax1.plot(kh, kh**(-3)*coef_norm, 'k', linewidth=1)
        ax1.plot(kh, 0.01*kh**(-5/3)*coef_norm, 'k--', linewidth=1)

    def plot2d(self, tmin=0, tmax=1000, delta_t=2,
               coef_compensate=3):
        f = h5py.File(self.path_file2D, 'r')
        dset_times = f['times']
        # nb_spectra = dset_times.shape[0]
        times = dset_times[...]
        # nt = len(times)

        kh = f['khE'][...]

        dset_spectrum = f['spectrum2D_E']

        delta_t_save = np.mean(times[1:]-times[0:-1])
        delta_i_plot = int(np.round(delta_t/delta_t_save))
        if delta_i_plot == 0 and delta_t != 0.:
            delta_i_plot = 1
        delta_t = delta_i_plot*delta_t_save

        imin_plot = np.argmin(abs(times-tmin))
        imax_plot = np.argmin(abs(times-tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]

        print(
            'plot2s(tmin={0}, tmax={1}, delta_t={2:.2f},'.format(
                tmin, tmax, delta_t) +
            ' coef_compensate={0:.3f})'.format(coef_compensate))

        print('''plot 2D spectra
tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}'''.format(
tmin_plot, tmax_plot, delta_t,
imin_plot, imax_plot, delta_i_plot))

        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel('$k_h$')
        ax1.set_ylabel('2D spectra')
        ax1.set_title('2D spectra, solver ' + self.output.name_solver +
                      ', nh = {0:5d}'.format(self.nx))
        ax1.hold(True)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        coef_norm = kh**coef_compensate

        if delta_t != 0.:
            for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
                EK = dset_spectrum[it]
                EK[EK < 10e-16] = 0.
                ax1.plot(kh, EK*coef_norm, 'k', linewidth=1)

        EK = dset_spectrum[imin_plot:imax_plot+1].mean(0)
        EK[EK < 10e-16] = 0.
        ax1.plot(kh, EK*coef_norm, 'k', linewidth=2)

        ax1.plot(kh, kh**(-3)*coef_norm, 'k--', linewidth=1)
        ax1.plot(kh, 0.01*kh**(-5./3)*coef_norm, 'k-.', linewidth=1)
