from __future__ import division
from __future__ import print_function
from builtins import range
import h5py

import numpy as np

from fluiddyn.util import mpi
from fluidsim.base.output.spectra import Spectra
from .normal_mode import NormalModeBase


class SpectraSW1L(Spectra):
    """Save and plot spectra."""

    def __init__(self, output):
        params = output.sim.params
        self.c2 = params.c2
        self.f = params.f

        super(SpectraSW1L, self).__init__(output)

    def _init_online_plot(self):
        super(SpectraSW1L, self)._init_online_plot()
        if mpi.rank == 0:
            title = ('spectra, solver '+self.output.name_solver+
                     ', nh = {0:5d}'.format(self.nx)+
                     ', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f)
                     )
            self.axe.set_title(title)

    def compute(self):
        """compute the values at one time."""
        # compute 'quantities_fft'
        energyK_fft, energyA_fft, energyKr_fft = (
            self.output.compute_energies_fft())
        ErtelPE_fft, CharneyPE_fft = self.output.compute_PE_fft()

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
            'spectrum1Dky_CPE': spectrum1Dky_CPE}

        # compute the spectra 2D
        spectrum2D_EK = self.spectrum2D_from_fft(energyK_fft)
        spectrum2D_EA = self.spectrum2D_from_fft(energyA_fft)
        spectrum2D_EKr = self.spectrum2D_from_fft(energyKr_fft)
        spectrum2D_EPE = self.spectrum2D_from_fft(ErtelPE_fft)
        spectrum2D_CPE = self.spectrum2D_from_fft(CharneyPE_fft)

        dico_spectra2D = {
            'spectrum2D_EK': spectrum2D_EK,
            'spectrum2D_EA': spectrum2D_EA,
            'spectrum2D_EKr': spectrum2D_EKr,
            'spectrum2D_EPE': spectrum2D_EPE,
            'spectrum2D_CPE': spectrum2D_CPE}

        dico_lin_spectra1D, dico_lin_spectra2D = self.compute_lin_spectra()
        dico_spectra1D.update(dico_lin_spectra1D)
        dico_spectra2D.update(dico_lin_spectra2D)

        return dico_spectra1D, dico_spectra2D

    def compute_lin_spectra(self):
        energy_glin_fft, energy_dlin_fft, energy_alin_fft = \
            self.output.compute_lin_energies_fft()

        spectrum1Dkx_Eglin, spectrum1Dky_Eglin = \
            self.spectra1D_from_fft(energy_glin_fft)
        spectrum1Dkx_Edlin, spectrum1Dky_Edlin = \
            self.spectra1D_from_fft(energy_dlin_fft)
        spectrum1Dkx_Ealin, spectrum1Dky_Ealin = \
            self.spectra1D_from_fft(energy_alin_fft)

        dico_spectra1D = {
            'spectrum1Dkx_Eglin': spectrum1Dkx_Eglin,
            'spectrum1Dky_Eglin': spectrum1Dky_Eglin,
            'spectrum1Dkx_Edlin': spectrum1Dkx_Edlin,
            'spectrum1Dky_Edlin': spectrum1Dky_Edlin,
            'spectrum1Dkx_Ealin': spectrum1Dkx_Ealin,
            'spectrum1Dky_Ealin': spectrum1Dky_Ealin}

        spectrum2D_Eglin = self.spectrum2D_from_fft(energy_glin_fft)
        spectrum2D_Edlin = self.spectrum2D_from_fft(energy_dlin_fft)
        spectrum2D_Ealin = self.spectrum2D_from_fft(energy_alin_fft)

        dico_spectra2D = {
            'spectrum2D_Eglin': spectrum2D_Eglin,
            'spectrum2D_Edlin': spectrum2D_Edlin,
            'spectrum2D_Ealin': spectrum2D_Ealin}

        return dico_spectra1D, dico_spectra2D

    def _online_plot_saving(self, dico_spectra1D, dico_spectra2D):
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
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        coef_norm = kh**(coef_compensate)

        min_to_plot = 1e-16

        if delta_t != 0.:
            for it in range(imin_plot, imax_plot+1, delta_i_plot):
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

        kh_pos = kh[kh > 0]
        coef_norm = coef_norm[kh > 0]
        ax1.plot(kh_pos, kh_pos ** (-3) * coef_norm, 'k--', linewidth=1)
        ax1.plot(kh_pos, kh_pos ** (-5./3) * coef_norm, 'k-.', linewidth=1)


    def plot2d(self, tmin=0, tmax=1000, delta_t=2,
               coef_compensate=0, keys=['Etot', 'EK', 'EA', 'EKr', 'EKd'],
               colors=['k', 'r', 'b', 'r--', 'r:']):

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
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        coef_norm = kh**coef_compensate

        machine_zero = 1e-15
        if delta_t != 0.:
            for it in range(imin_plot, imax_plot+1, delta_i_plot):
                for k, c in zip(keys, colors):
                    dset = self._select_field(it, k, f)
                    dset[dset < 10e-16] = machine_zero
                    ax1.plot(kh, dset * coef_norm, c, linewidth=1)

        EK = dset_spectrumEK[imin_plot:imax_plot+1].mean(0)
        EA = dset_spectrumEA[imin_plot:imax_plot+1].mean(0)
        EKr = dset_spectrumEKr[imin_plot:imax_plot+1].mean(0)


        EK[abs(EK)<10e-16] = machine_zero
        EA[abs(EA)<10e-16] = machine_zero
        EKr[abs(EKr)<10e-16] = machine_zero

        E_tot = EK + EA
        EKd = EK - EKr + machine_zero

        if 'Etot' in keys:
            ax1.plot(kh, E_tot * coef_norm, 'k', linewidth=3, label='$E_{tot}$')

        if 'EK' in keys:
            ax1.plot(kh, EK * coef_norm, 'r', linewidth=2, label='$E_{K}$')
            ax1.plot(kh, -EK * coef_norm, 'k-', linewidth=2)

        if 'EA' in keys:
            ax1.plot(kh, EA * coef_norm, 'b', linewidth=2, label='$E_{A}$')

        if 'EKr' in keys:
            ax1.plot(kh, EKr * coef_norm, 'r--', linewidth=2, label='$E_{Kr}$')

        if 'EKd' in keys:
            ax1.plot(kh, EKd * coef_norm, 'r:', linewidth=2, label='$E_{Kd}$')
            ax1.plot(kh, -EKd * coef_norm, 'k:', linewidth=2)

        self._plot2d_lin_spectra(f, ax1, imin_plot, imax_plot, kh, coef_norm, keys)

        kh_pos = kh[kh > 0]
        coef_norm = coef_norm[kh > 0]
        ax1.plot(kh_pos, kh_pos ** (-2) * coef_norm, 'k-', linewidth=1)
        ax1.plot(kh_pos, kh_pos ** (-3) * coef_norm, 'k--', linewidth=1)
        ax1.plot(kh_pos, kh_pos ** (-5./3) * coef_norm, 'k-.', linewidth=1)

        font = {'family': 'serif',
                'weight': 'normal',
                'size': 16
               }
        postxt = kh.max()
        ax1.text(postxt, postxt**(-2 + coef_compensate), r'$k^{-2}$', fontdict=font)
        ax1.text(postxt, postxt**(-3 + coef_compensate), r'$k^{-3}$', fontdict=font)
        ax1.text(postxt, postxt**(-5./3 + coef_compensate), r'$k^{-5/3}$', fontdict=font)
        ax1.legend()

    def _plot2d_lin_spectra(self, f, ax1, imin_plot, imax_plot, kh, coef_norm):
        machine_zero = 1e-15
        if self.sim.info.solver.short_name.startswith('SW1L'):
            dset_spectrumEdlin = f['spectrum2D_Edlin']
            Edlin = dset_spectrumEdlin[imin_plot:imax_plot + 1].mean(0) + machine_zero
            ax1.plot(kh, Edlin * coef_norm, 'c', linewidth=1, label='$E_{D}$')

        if self.params.f != 0:
            dset_spectrumEglin = f['spectrum2D_Eglin']
            Eglin = dset_spectrumEglin[imin_plot:imax_plot + 1].mean(0) + machine_zero
            ax1.plot(kh, Eglin * coef_norm, 'g', linewidth=1, label='$E_{G}$')

            dset_spectrumEalin = f['spectrum2D_Ealin']
            Ealin = dset_spectrumEalin[imin_plot:imax_plot + 1].mean(0) + machine_zero
            ax1.plot(kh, Ealin * coef_norm, 'y', linewidth=1, label='$E_{A}$')

    def _select_field(self, idx, key_field=None, f=None):
        if key_field is None:
            key_field = self._ani_key

        def select(idx, key_field, f):
            if key_field is 'Etot' or key_field is None:
                self._ani_key = 'Etot'
                y = f['spectrum2D_EK'][idx] + f['spectrum2D_EA'][idx]
            elif key_field is 'EKd':
                y = f['spectrum2D_EK'][idx] - f['spectrum2D_EKr'][idx]
            else:
                try:
                    key_field = 'spectrum2D_' + key_field
                    y = f[key_field][idx]
                except:
                    raise KeyError('Unknown key ', key_field)

            return y

        if f is None:
            with h5py.File(self.path_file2D) as f:
                return select(idx, key_field, f)
        else:
            return select(idx, key_field, f)


class SpectraSW1LNormalMode(SpectraSW1L):
    def __init__(self, output):
        self.norm_mode = NormalModeBase(output)
        super(SpectraSW1LNormalMode, self).__init__(output)

    def compute_lin_spectra(self):
        energy_glin_fft, energy_aplin_fft, energy_amlin_fft = \
            self.norm_mode.compute_qapam_energies_fft()

        energy_alin_fft = energy_aplin_fft + energy_amlin_fft
        spectrum1Dkx_Eglin, spectrum1Dky_Eglin = \
            self.spectra1D_from_fft(energy_glin_fft)
        spectrum1Dkx_Ealin, spectrum1Dky_Ealin = \
            self.spectra1D_from_fft(energy_alin_fft)

        dico_spectra1D = {
            'spectrum1Dkx_Eglin': spectrum1Dkx_Eglin,
            'spectrum1Dky_Eglin': spectrum1Dky_Eglin,
            'spectrum1Dkx_Ealin': spectrum1Dkx_Ealin,
            'spectrum1Dky_Ealin': spectrum1Dky_Ealin}

        spectrum2D_Eglin = self.spectrum2D_from_fft(energy_glin_fft)
        spectrum2D_Ealin = self.spectrum2D_from_fft(energy_alin_fft)

        dico_spectra2D = {
            'spectrum2D_Eglin': spectrum2D_Eglin,
            'spectrum2D_Ealin': spectrum2D_Ealin}

        return dico_spectra1D, dico_spectra2D

    def plot2d(self, tmin=0, tmax=1000, delta_t=2,
               coef_compensate=0, keys=['Etot', 'EK', 'Eglin', 'Ealin'],
               colors=['k', 'r', 'g', 'y']):

        super(SpectraSW1LNormalMode, self).plot2d(tmin, tmax, delta_t, coef_compensate, keys, colors)

    def _plot2d_lin_spectra(self, f, ax1, imin_plot, imax_plot, kh, coef_norm, keys):
        machine_zero = 1e-15
        if self.sim.info.solver.short_name.startswith('SW1L'):
            if 'spectrum2D_Edlin' in f.keys():
                dset_spectrumEalin = f['spectrum2D_Edlin']  # TODO: To be removed. Kept for compatibility
            else:
                dset_spectrumEalin = f['spectrum2D_Ealin']

            if 'Ealin' in keys:
                Ealin = dset_spectrumEalin[imin_plot:imax_plot + 1].mean(0) + machine_zero
                ax1.plot(kh, Ealin * coef_norm, 'y', linewidth=1, label='$E_{AGEO}$')

            if 'Eglin' in keys:
                dset_spectrumEglin = f['spectrum2D_Eglin']
                Eglin = dset_spectrumEglin[imin_plot:imax_plot + 1].mean(0) + machine_zero
                ax1.plot(kh, Eglin * coef_norm, 'g', linewidth=1, label='$E_{GEO}$')
