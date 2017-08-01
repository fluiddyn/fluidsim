import h5py

import numpy as np

from fluidsim.base.output.spectra import Spectra


class SpectraSW1L(Spectra):
    """Save and plot spectra."""


    def __init__(self, output):
        params = output.sim.params
        self.c2 = params.c2
        self.f = params.f

        super(SpectraSW1L, self).__init__(output)



    def init_online_plot(self):
        fig, axe = self.output.figure_axe(numfig=1000000)
        self.axe = axe
        axe.set_xlabel('k_h')
        axe.set_ylabel('E(k_h)')
        title = ('spectra, solver '+self.output.name_solver+
                 ', nh = {0:5d}'.format(self.nx)+
                 ', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f)
                 )
        axe.set_title(title)
        axe.hold(True)


    def compute(self):
        """compute the values at one time."""
        # compute 'quantities_fft'
        energyK_fft, energyA_fft, energyKr_fft = (
            self.output.compute_energies_fft())
        ErtelPE_fft, CharneyPE_fft = self.output.compute_PE_fft()

        energy_glin_fft, energy_dlin_fft, energy_alin_fft = \
            self.output.compute_lin_energies_fft()



        # compute the spectra 1D
        spectrum1Dkx_EK, spectrum1Dky_EK = \
            self.spectra1D_from_fft(energyK_fft)
        spectrum1Dkx_EA, spectrum1Dky_EA = \
            self.spectra1D_from_fft(energyA_fft)
        spectrum1Dkx_EKr, spectrum1Dky_EKr = \
            self.spectra1D_from_fft(energyKr_fft)
        spectrum1Dkx_EPE, spectrum1Dky_EPE = \
            self.spectra1D_from_fft(ErtelPE_fft)
        spectrum1Dkx_CPE, spectrum1Dky_CPE = \
            self.spectra1D_from_fft(CharneyPE_fft)
        spectrum1Dkx_Eglin, spectrum1Dky_Eglin = \
            self.spectra1D_from_fft(energy_glin_fft)
        spectrum1Dkx_Edlin, spectrum1Dky_Edlin = \
            self.spectra1D_from_fft(energy_dlin_fft)
        spectrum1Dkx_Ealin, spectrum1Dky_Ealin = \
            self.spectra1D_from_fft(energy_alin_fft)

        dico_spectra1D = {
            'spectrum1Dkx_EK': spectrum1Dkx_EK,
            'spectrum1Dky_EK': spectrum1Dky_EK,
            'spectrum1Dkx_EA': spectrum1Dkx_EA,
            'spectrum1Dky_EA': spectrum1Dky_EA,
            'spectrum1Dkx_EKr': spectrum1Dkx_EKr,
            'spectrum1Dky_EKr': spectrum1Dky_EKr,
            'spectrum1Dkx_EPE': spectrum1Dkx_EPE,
            'spectrum1Dky_EPE': spectrum1Dky_EPE,
            'spectrum1Dkx_CPE': spectrum1Dkx_CPE,
            'spectrum1Dky_CPE': spectrum1Dky_CPE,
            'spectrum1Dkx_Eglin': spectrum1Dkx_Eglin,
            'spectrum1Dky_Eglin': spectrum1Dky_Eglin,
            'spectrum1Dkx_Edlin': spectrum1Dkx_Edlin,
            'spectrum1Dky_Edlin': spectrum1Dky_Edlin,
            'spectrum1Dkx_Ealin': spectrum1Dkx_Ealin,
            'spectrum1Dky_Ealin': spectrum1Dky_Ealin}

        # compute the spectra 2D
        spectrum2D_EK = self.spectrum2D_from_fft(energyK_fft)
        spectrum2D_EA = self.spectrum2D_from_fft(energyA_fft)
        spectrum2D_EKr = self.spectrum2D_from_fft(energyKr_fft)
        spectrum2D_EPE = self.spectrum2D_from_fft(ErtelPE_fft)
        spectrum2D_CPE = self.spectrum2D_from_fft(CharneyPE_fft)
        spectrum2D_Eglin = self.spectrum2D_from_fft(energy_glin_fft)
        spectrum2D_Edlin = self.spectrum2D_from_fft(energy_dlin_fft)
        spectrum2D_Ealin = self.spectrum2D_from_fft(energy_alin_fft)

        dico_spectra2D = {
            'spectrum2D_EK': spectrum2D_EK,
            'spectrum2D_EA': spectrum2D_EA,
            'spectrum2D_EKr': spectrum2D_EKr,
            'spectrum2D_EPE': spectrum2D_EPE,
            'spectrum2D_CPE': spectrum2D_CPE,
            'spectrum2D_Eglin': spectrum2D_Eglin,
            'spectrum2D_Edlin': spectrum2D_Edlin,
            'spectrum2D_Ealin': spectrum2D_Ealin}

        return dico_spectra1D, dico_spectra2D



    def _online_plot(self, dico_spectra1D, dico_spectra2D):
        if (self.params.oper.nx==self.params.oper.ny
                and self.params.oper.Lx==self.params.oper.Ly):
            spectrum2D_EK = dico_spectra2D['spectrum2D_EK']
            spectrum2D_EA = dico_spectra2D['spectrum2D_EA']
            spectrum2D_EKr = dico_spectra2D['spectrum2D_EKr']
            spectrum2D_E = spectrum2D_EK + spectrum2D_EA
            spectrum2D_EKd = spectrum2D_EK - spectrum2D_EKr
            khE = self.oper.khE
            coef_norm = khE**(3.)
            self.axe.loglog(khE, spectrum2D_E*coef_norm, 'k')
            self.axe.loglog(khE, spectrum2D_EK*coef_norm, 'r')
            self.axe.loglog(khE, spectrum2D_EA*coef_norm, 'b')
            self.axe.loglog(khE, spectrum2D_EKr*coef_norm, 'r--')
            self.axe.loglog(khE, spectrum2D_EKd*coef_norm, 'r:')
            lin_inf, lin_sup = self.axe.get_ylim()
            if lin_inf<10e-6:
                lin_inf=10e-6
            self.axe.set_ylim([lin_inf, lin_sup])
        else:
            print('you need to implement the ploting '
                  'of the spectra for this case')

    def plot1d(self, tmin=0, tmax=1000, delta_t=2,
               coef_compensate=3):

        f = h5py.File(self.path_file1D, 'r')
        dset_times = f['times']
        times = dset_times[...]
        # nb_spectra = times.shape[0]

        dset_kxE = f['kxE']
        # dset_kyE = f['kyE']

        kh = dset_kxE[...]

        dset_spectrum1Dkx_EK = f['spectrum1Dkx_EK']
        dset_spectrum1Dky_EK = f['spectrum1Dky_EK']
        dset_spectrum1Dkx_EA = f['spectrum1Dkx_EA']
        dset_spectrum1Dky_EA = f['spectrum1Dky_EA']

        dset_spectrum1Dkx_EKr = f['spectrum1Dkx_EKr']
        dset_spectrum1Dky_EKr = f['spectrum1Dky_EKr']

        nt = len(times)

        delta_t_save = np.mean(times[1:]-times[0:-1])
        delta_i_plot = int(np.round(delta_t/delta_t_save))
        if delta_i_plot == 0 and delta_t != 0.:
            delta_i_plot=1
        delta_t = delta_i_plot*delta_t_save

        imin_plot = np.argmin(abs(times-tmin))
        imax_plot = np.argmin(abs(times-tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]

        to_print = (
'plot1d(tmin={0}, tmax={1}, delta_t={2:.2f},'.format(tmin, tmax, delta_t)
+' coef_compensate={0:.3f})'.format(coef_compensate)
)
        print(to_print)

        to_print = '''plot 1D spectra
tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}'''.format(
tmin_plot, tmax_plot, delta_t,
imin_plot, imax_plot, delta_i_plot)
        print(to_print)

        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel('$k_h$')
        ax1.set_ylabel('1D spectra')
        title = ('1D spectra, solver '+self.output.name_solver+
', nh = {0:5d}'.format(self.nx)+
', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f)
)
        ax1.set_title(title)
        ax1.hold(True)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        coef_norm = kh**(coef_compensate)

        min_to_plot = 1e-16

        if delta_t != 0.:
            for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
                E_K = (dset_spectrum1Dkx_EK[it]+dset_spectrum1Dky_EK[it])
                # E_K[E_K<min_to_plot] = 0.
                E_A = (dset_spectrum1Dkx_EA[it]+dset_spectrum1Dky_EA[it])
                # E_A[E_A<min_to_plot] = 0.
                E_tot = E_K + E_A

                E_Kr = (dset_spectrum1Dkx_EKr[it]+dset_spectrum1Dky_EKr[it])
                # E_Kr[E_Kr<min_to_plot] = 0.
                E_Kd = E_K - E_Kr

                ax1.plot(kh, E_tot*coef_norm, 'k', linewidth=2)
                ax1.plot(kh, E_K*coef_norm, 'r', linewidth=1)
                ax1.plot(kh, E_A*coef_norm, 'b', linewidth=1)
                # ax1.plot(kh, E_Kr*coef_norm, 'r--', linewidth=1)
                # ax1.plot(kh, E_Kd*coef_norm, 'r:', linewidth=1)


        E_K = (dset_spectrum1Dkx_EK[imin_plot:imax_plot+1]
               +dset_spectrum1Dky_EK[imin_plot:imax_plot+1]).mean(0)

        E_A = (dset_spectrum1Dkx_EA[imin_plot:imax_plot+1]
               +dset_spectrum1Dky_EA[imin_plot:imax_plot+1]).mean(0)

        ax1.plot(kh, E_K*coef_norm, 'r', linewidth=2)
        ax1.plot(kh, E_A*coef_norm, 'b', linewidth=2)

        ax1.plot(kh, kh**(-3)*coef_norm, 'k', linewidth=1)
        ax1.plot(kh, 0.01*kh**(-5./3)*coef_norm, 'k--', linewidth=1)


    def plot2d(self, tmin=0, tmax=1000, delta_t=2,
               coef_compensate=3):

        f = h5py.File(self.path_file2D, 'r')
        dset_times = f['times']
        nb_spectra = dset_times.shape[0]
        times = dset_times[...]
        nt = len(times)

        dset_khE = f['khE']
        kh = dset_khE[...]

        dset_spectrumEK = f['spectrum2D_EK']
        dset_spectrumEA = f['spectrum2D_EA']
        dset_spectrumEKr = f['spectrum2D_EKr']

        delta_t_save = np.mean(times[1:]-times[0:-1])
        delta_i_plot = int(np.round(delta_t/delta_t_save))
        if delta_i_plot == 0 and delta_t != 0.:
            delta_i_plot=1
        delta_t = delta_i_plot*delta_t_save

        imin_plot = np.argmin(abs(times-tmin))
        imax_plot = np.argmin(abs(times-tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]

        to_print = (
'plot2d(tmin={0}, tmax={1}, delta_t={2:.2f},'.format(tmin, tmax, delta_t)
+' coef_compensate={0:.3f})'.format(coef_compensate)
)
        print(to_print)

        to_print = '''plot 2D spectra
tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}'''.format(
tmin_plot, tmax_plot, delta_t,
imin_plot, imax_plot, delta_i_plot)
        print(to_print)

        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel('$k_h$')
        ax1.set_ylabel('2D spectra')
        title = ('2D spectra, solver '+self.output.name_solver+
', nh = {0:5d}'.format(self.nx)+
', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f)
)
        ax1.set_title(title)
        ax1.hold(True)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        coef_norm = kh**coef_compensate

        if delta_t != 0.:
            for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
                EK = dset_spectrumEK[it]
                EA = dset_spectrumEA[it]
                EKr = dset_spectrumEKr[it]

                EK[EK<10e-16] = 0.
                EA[EA<10e-16] = 0.
                EKr[EKr<10e-16] = 0.

                E_tot = EK + EA
                EKd = EK - EKr

                ax1.plot(kh, E_tot*coef_norm, 'k', linewidth=1)
                ax1.plot(kh, EK*coef_norm, 'r', linewidth=1)
                ax1.plot(kh, EA*coef_norm, 'b', linewidth=1)
                ax1.plot(kh, EKr*coef_norm, 'r--', linewidth=1)
                ax1.plot(kh, EKd*coef_norm, 'r:', linewidth=1)

        EK = dset_spectrumEK[imin_plot:imax_plot+1].mean(0)
        EA = dset_spectrumEA[imin_plot:imax_plot+1].mean(0)
        EKr = dset_spectrumEKr[imin_plot:imax_plot+1].mean(0)


        EK[abs(EK)<10e-16] = 0.
        EA[abs(EA)<10e-16] = 0.
        EKr[abs(EKr)<10e-16] = 0.

        E_tot = EK + EA
        EKd = EK - EKr


        ax1.plot(kh, E_tot*coef_norm, 'k', linewidth=4)
        ax1.plot(kh, EK*coef_norm, 'r', linewidth=2)
        ax1.plot(kh, EA*coef_norm, 'b', linewidth=2)
        ax1.plot(kh, EKr*coef_norm, 'r--', linewidth=2)
        ax1.plot(kh, EKd*coef_norm, 'r:', linewidth=2)

        ax1.plot(kh, -EK*coef_norm, 'm', linewidth=2)
        ax1.plot(kh, -EKd*coef_norm, 'm:', linewidth=2)





        if self.sim.info.solver.short_name.startswith('SW1L'):
            dset_spectrumEdlin = f['spectrum2D_Edlin']
            Edlin = dset_spectrumEdlin[imin_plot:imax_plot+1].mean(0)
            ax1.plot(kh, Edlin*coef_norm, 'y:', linewidth=1)

        if self.params.f != 0:
            dset_spectrumEglin = f['spectrum2D_Eglin']
            Eglin = dset_spectrumEglin[imin_plot:imax_plot+1].mean(0)
            ax1.plot(kh, Eglin*coef_norm, 'c', linewidth=1)

            dset_spectrumEalin = f['spectrum2D_Ealin']
            Ealin = dset_spectrumEalin[imin_plot:imax_plot+1].mean(0)
            ax1.plot(kh, Ealin*coef_norm, 'y', linewidth=1)




        ax1.plot(kh, kh**(-3)*coef_norm, 'k--', linewidth=1)
        ax1.plot(kh, 0.01*kh**(-5./3)*coef_norm, 'k-.', linewidth=1)

    def _ani_get_field(self, time):
        f = h5py.File(self.path_file2D, 'r')
        dset_times = f['times']
        times = dset_times[...]

        it = np.argmin(abs(times-time))
        y = self._select_field(h5file=f, key_field=self._ani_key, it=it)
        y[abs(y) < 10e-16] = 0
        
        return y, self._ani_key
    
    def _select_field(self, h5file=None, key_field=None, it=None):
        if key_field is 'EK' or key_field is None:
            self._ani_key = 'EK'
            y = h5file['spectrum2D_EK'][it]
        elif key_field is 'EA':
            y = h5file['spectrum2D_EA'][it]
        elif key_field is 'EKr':
            y = h5file['spectrum2D_EKr'][it]
        elif key_field is 'EKd':
            y = h5file['spectrum2D_EK'][it] - h5file['spectrum2D_EKr'][it]
        else:
            raise KeyError('Unknown key ',key_field)
        
        return y
    
