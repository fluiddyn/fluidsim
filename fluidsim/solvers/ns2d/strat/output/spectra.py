"""Spectra output (:mod:`fluidsim.solvers.ns2d.strat.output.spectra`)
===============================================================

.. autoclass:: SpectraNS2DStrat
   :members:
   :private-members:

"""

import h5py

import numpy as np

from fluidsim.base.output.spectra import Spectra

class SpectraNS2DStrat(Spectra):
    """Save and plot spectra."""

    def compute(self):
        """compute the values at one time."""
        # energy_fft = self.output.compute_energy_fft()
        energyK_fft, energyA_fft = self.output.compute_energies_fft()
        energyK_ux_fft, energyK_uy_fft = self.output.compute_energies2_fft()

        # Compute the kinetic energy spectra 1D for the two velocity components
        # and two directions
        spectrum1Dkx_EK_ux, spectrum1Dky_EK_ux = self.spectra1D_from_fft(energyK_ux_fft)
        spectrum1Dkx_EK_uy, spectrum1Dky_EK_uy = self.spectra1D_from_fft(energyK_uy_fft)

        # Compute the potential energy spectra 1D two directions
        spectrum1Dkx_EA, spectrum1Dky_EA = self.spectra1D_from_fft(energyA_fft)
        
        # Dictionary with the 1D kinetic energy spectra
        dico_spectra1D = {'spectrum1Dkx_EK_ux': spectrum1Dkx_EK_ux,
                          'spectrum1Dky_EK_ux': spectrum1Dky_EK_ux,
                          'spectrum1Dkx_EK_uy': spectrum1Dkx_EK_uy,
                          'spectrum1Dky_EK_uy': spectrum1Dky_EK_uy,
                          'spectrum1Dkx_EA': spectrum1Dkx_EA,
                          'spectrum1Dky_EA': spectrum1Dky_EA}

        # spectrum1Dkx_EA, spectrum1Dky_EA = self.spectra1D_from_fft(energyA_fft)
        # dico_spectra1D = {'spectrum1Dkx_EK': spectrum1Dkx_EK,
        #                  'spectrum1Dky_EK': spectrum1Dky_EK,
        #                  'spectrum1Dkx_EA': spectrum1Dkx_EA,
        #                  'spectrum1Dky_EA': spectrum1Dky_EA}

        # # compute the spectra 1D
        # spectrum1Dkx_E, spectrum1Dky_E = self.spectra1D_from_fft(energy_fft)
        # dico_spectra1D = {'spectrum1Dkx_E': spectrum1Dkx_E,
        #                   'spectrum1Dky_E': spectrum1Dky_E}

        # compute the kinetic energy spectra 2D
        spectrum2D_EK = self.spectrum2D_from_fft(energyK_fft)
        spectrum2D_EA = self.spectrum2D_from_fft(energyA_fft)
        dico_spectra2D = {'spectrum2D_EK': spectrum2D_EK,
                          'spectrum2D_EA': spectrum2D_EA}
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
        dset_kyE = f['kyE']
        kh = dset_kxE.value
        kv = dset_kyE.value


        # Open data set 1D kinetic energy spectra
        dset_spectrum1Dkx_EK_ux = f['spectrum1Dkx_EK_ux']
        dset_spectrum1Dky_EK_ux = f['spectrum1Dky_EK_ux']
        dset_spectrum1Dkx_EK_uy = f['spectrum1Dkx_EK_uy']
        dset_spectrum1Dky_EK_uy = f['spectrum1Dky_EK_uy']
        dset_spectrum1Dky_EA = f['spectrum1Dky_EA']

        # dset_spectrum1Dkx = f['spectrum1Dkx_E']
        # dset_spectrum1Dky = f['spectrum1Dky_E']


        # nb_spectra = dset_times.shape[0]
        times = dset_times.value
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
        ax1.set_ylabel('horizontal EK spectra')
        ax1.set_title('1D spectra, solver '+self.output.name_solver +
                      ', nh = {0:5d}'.format(self.nx))
        ax1.hold(True)
        ax1.set_xscale('log')
        ax1.set_yscale('log')


        # coef_norm = kh**(coef_compensate)
        # if delta_t != 0.:
        #     for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
        #          EK = dset_spectrum1Dkx_EK_ux[it]
        #          EK[EK < 10e-16] = 0.
        #          ax1.plot(kv, EK*coef_norm, 'k', linewidth=2)

        EK_ux_kx = (dset_spectrum1Dkx_EK_ux[imin_plot:imax_plot+1]).mean(0)
        EK_uy_kx = (dset_spectrum1Dkx_EK_uy[imin_plot:imax_plot+1]).mean(0)
        # print 'EK = ', EK
        ax1.plot(kh, EK_ux_kx, 'b--', label='u_x', linewidth=3)
        ax1.plot(kh, EK_uy_kx, 'r--', label='u_y', linewidth=3)
        ax1.plot(kh[1:], 0.01*kh[1:]**(-5/3), 'k',
                 label='spectra k^(-5/3)', linewidth=2)
        ax1.legend()

        fig, ax2 = self.output.figure_axe()
        ax2.set_xlabel('$k_v$')
        ax2.set_ylabel('vertical EK spectra')
        ax2.set_title('1D spectra, solver '+self.output.name_solver +
                      ', nh = {0:5d}'.format(self.nx))
        ax2.hold(True)
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        #coef_norm = kv**(coef_compensate)
        EK_ux_ky = (dset_spectrum1Dky_EK_ux[imin_plot:imax_plot+1]).mean(0)
        EK_uy_ky = (dset_spectrum1Dky_EK_uy[imin_plot:imax_plot+1]).mean(0)
        ax2.plot(kv, EK_ux_ky, 'b--', label='u_x', linewidth=3)
        ax2.plot(kv, EK_uy_ky, 'r--', label='u_y', linewidth=3)
        ax2.plot(kv[1:], 0.01*kv[1:]**(-3), 'k',
                 label='spectra k^(-3)', linewidth=2)
        ax2.legend()

        fig, ax3 = self.output.figure_axe()
        ax3.set_xlabel('$k_v$')
        ax3.set_ylabel('vertical EA spectra')
        ax3.set_title('1D spectra, solver '+self.output.name_solver +
                      ', nh = {0:5d}'.format(self.nx))
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        EA_ky = (dset_spectrum1Dky_EA[imin_plot:imax_plot+1]).mean(0)
        ax3.plot(kv, EA_ky, 'b--', linewidth=3)


        # coef_norm = kv**(coef_compensate)
        # if delta_t != 0.:
        #     for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
        #          EK = dset_spectrum1Dky_EK_uy[it]
        #          EK[EK < 10e-16] = 0.
        #          ax2.plot(kv, EK*coef_norm, 'k', linewidth=2)


        # coef_norm = 1
        # kh = kh[1:]
        # coef_norm = kh**(coef_compensate)
        # if delta_t != 0.:
        #     for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
        #          EK = (dset_spectrum1Dkx_EK[it]+ dset_spectrum1Dky_EK[it])
        #          EK[EK < 10e-16] = 0.
        #          ax1.plot(kh, EK[1:]*coef_norm, 'k', linewidth=2)
        # print 'it = ', it
        # print 'EK = ', EK
        # print 'kh = ', kh

        # EK = (dset_spectrum1Dkx_EK[imin_plot:imax_plot+1] +
        #       dset_spectrum1Dky_EK[imin_plot:imax_plot+1]).mean(0)
        # print 'EK = ', EK
        # ax1.plot(kh, EK[1:]*coef_norm, 'b--', linewidth=3)

        # ax1.plot(kh[1:], kh[1:]**(-3)*coef_norm, 'r', linewidth=1)
        # ax1.plot(kh, 0.01*kh**(-5/3)*coef_norm, 'k', linewidth=2)
        # ax1.plot(kh, kh**(-5/3), 'r', linewidth=2)

    def plot2d(self, tmin=0, tmax=1000, delta_t=2,
               coef_compensate=3):
        f = h5py.File(self.path_file2D, 'r')
        dset_times = f['times']
        # nb_spectra = dset_times.shape[0]
        times = dset_times.value
        # nt = len(times)

        kh = f['khE'].value

        dset_spectrum = f['spectrum2D_EK']

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
