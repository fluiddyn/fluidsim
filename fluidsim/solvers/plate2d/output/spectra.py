"""Spectra (:mod:`fluidsim.solvers.plate2d.output.spectra`)
=================================================================


Provides:

.. autoclass:: SpectraPlate2D
   :members:
   :private-members:

"""

import h5py

import numpy as np

from fluidsim.base.output.spectra import Spectra


class SpectraPlate2D(Spectra):
    """Compute, save, load and plot spectra."""

    def compute(self):
        """compute the values at one time."""
        EK_fft, EL_fft, EE_fft = self.output.compute_energies_fft()
        # compute the spectra 1D
        spectrum1Dkx_EK, spectrum1Dky_EK = self.spectra1D_from_fft(EK_fft)
        spectrum1Dkx_EL, spectrum1Dky_EL = self.spectra1D_from_fft(EL_fft)
        spectrum1Dkx_EE, spectrum1Dky_EE = self.spectra1D_from_fft(EE_fft)

        dict_spectra1D = {
            "spectrum1Dkx_EK": spectrum1Dkx_EK,
            "spectrum1Dky_EK": spectrum1Dky_EK,
            "spectrum1Dkx_EL": spectrum1Dkx_EL,
            "spectrum1Dky_EL": spectrum1Dky_EL,
            "spectrum1Dkx_EE": spectrum1Dkx_EE,
            "spectrum1Dky_EE": spectrum1Dky_EE,
        }

        # compute the spectra 2D
        spectrum2D_EK = self.spectrum2D_from_fft(EK_fft)
        spectrum2D_EL = self.spectrum2D_from_fft(EL_fft)
        spectrum2D_EE = self.spectrum2D_from_fft(EE_fft)
        dict_spectra2D = {
            "spectrum2D_EK": spectrum2D_EK,
            "spectrum2D_EL": spectrum2D_EL,
            "spectrum2D_EE": spectrum2D_EE,
        }
        return dict_spectra1D, dict_spectra2D

    def _online_plot_saving(self, dict_spectra1D, dict_spectra2D):
        if (
            self.nx == self.params.oper.ny
            and self.params.oper.Lx == self.params.oper.Ly
        ):
            spectrum2D_EK = dict_spectra2D["spectrum2D_EK"]
            spectrum2D_EL = dict_spectra2D["spectrum2D_EL"]
            spectrum2D_EE = dict_spectra2D["spectrum2D_EE"]
            spectrum2D_Etot = spectrum2D_EK + spectrum2D_EL + spectrum2D_EE
            khE = self.oper.khE
            coef_norm = khE ** (3.0)
            self.axe.loglog(khE, spectrum2D_Etot * coef_norm, "k", linewidth=2)
            self.axe.loglog(khE, spectrum2D_EK * coef_norm, "r--")
            self.axe.loglog(khE, spectrum2D_EL * coef_norm, "b--")
            self.axe.loglog(khE, spectrum2D_EE * coef_norm, "y--")
        # lin_inf, lin_sup = self.axe.get_ylim()
        # if lin_inf < 10e-6:
        #     lin_inf = 10e-6
        # self.axe.set_ylim([lin_inf, lin_sup])
        else:
            print(
                "you need to implement the ploting "
                "of the spectra for this case"
            )

    def plot1d(self, tmin=0, tmax=1000, delta_t=2, coef_compensate=3):

        with h5py.File(self.path_file1D, "r") as h5file:
            dset_times = h5file["times"]

            dset_kxE = h5file["kxE"]
            # dset_kyE = h5file['kyE']
            kh = dset_kxE[...]

            kh_not0 = kh.copy()
            kh_not0[kh_not0 == 0] = 1e-15

            dset_spectrum1Dkx_EK = h5file["spectrum1Dkx_EK"]
            dset_spectrum1Dky_EK = h5file["spectrum1Dky_EK"]
            # dset_spectrum1Dkx_EL = h5file['spectrum1Dkx_EL']
            # dset_spectrum1Dky_EL = h5file['spectrum1Dky_EL']
            # dset_spectrum1Dkx_EE = h5file['spectrum1Dkx_EE']
            # dset_spectrum1Dky_EE = h5file['spectrum1Dky_EE']

            times = dset_times[...]

            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))
            delta_t = delta_t_save * delta_i_plot
            if delta_i_plot == 0:
                delta_i_plot = 1

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            print(
                "plot1d(tmin={}, tmax={}, delta_t={:.2f},".format(
                    tmin, tmax, delta_t
                )
                + f" coef_compensate={coef_compensate:.3f})"
            )

            print(
                (
                    "plot 1D spectra\n"
                    "tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}\n"
                    "imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}"
                ).format(
                    tmin_plot,
                    tmax_plot,
                    delta_t,
                    imin_plot,
                    imax_plot,
                    delta_i_plot,
                )
            )

            fig, ax1 = self.output.figure_axe()
            ax1.set_xlabel("$k_h$")
            ax1.set_ylabel("spectra")
            ax1.set_title("1D spectra\n" + self.output.summary_simul)
            ax1.set_xscale("log")
            ax1.set_yscale("log")

            coef_norm = kh_not0 ** (coef_compensate)
            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot + 1, delta_i_plot):
                    EK = dset_spectrum1Dkx_EK[it] + dset_spectrum1Dky_EK[it]
                    EK[EK < 10e-16] = 0.0
                    ax1.plot(kh, EK * coef_norm, "r", linewidth=1)

            EK = (
                dset_spectrum1Dkx_EK[imin_plot : imax_plot + 1]
                + dset_spectrum1Dky_EK[imin_plot : imax_plot + 1]
            ).mean(0)

        ax1.plot(kh, kh_not0 ** (-3) * coef_norm, "k", linewidth=1)
        ax1.plot(kh, 0.01 * kh_not0 ** (-5 / 3) * coef_norm, "k--", linewidth=1)

    def plot2d(self, tmin=0, tmax=1000, delta_t=2, coef_compensate=3):
        with h5py.File(self.path_file2D, "r") as h5file:
            dset_times = h5file["times"]
            nt = dset_times.shape[0]
            if nt == 0:
                raise ValueError("No spectra are saved in this file.")

            times = dset_times[...]

            kh = h5file["khE"][...]

            kh_not0 = kh.copy()
            kh_not0[kh_not0 == 0] = 1e-15

            dset_spectrum_EK = h5file["spectrum2D_EK"]
            dset_spectrum_EL = h5file["spectrum2D_EL"]
            dset_spectrum_EE = h5file["spectrum2D_EE"]

            if nt == 1:
                imin_plot = imax_plot = 0
            else:
                delta_t_save = np.mean(times[1:] - times[0:-1])
                delta_i_plot = int(np.round(delta_t / delta_t_save))
                if delta_i_plot == 0 and delta_t != 0.0:
                    delta_i_plot = 1
                delta_t = delta_i_plot * delta_t_save

                imin_plot = np.argmin(abs(times - tmin))
                imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            print(
                "plot2d(tmin={}, tmax={}, delta_t={:.2f},".format(
                    tmin, tmax, delta_t
                )
                + f" coef_compensate={coef_compensate:.3f})"
            )

            print(
                (
                    "plot 2D spectra\n"
                    "tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}\n"
                    "imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}"
                ).format(
                    tmin_plot,
                    tmax_plot,
                    delta_t,
                    imin_plot,
                    imax_plot,
                    delta_i_plot,
                )
            )

            fig, ax1 = self.output.figure_axe()
            ax1.set_xlabel("$k_h$")
            ax1.set_ylabel("2D spectra")
            ax1.set_title("2D spectra\n" + self.output.summary_simul)
            ax1.set_xscale("log")
            ax1.set_yscale("log")

            coef_norm = kh_not0**coef_compensate

            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot + 1, delta_i_plot):
                    EK = dset_spectrum_EK[it]
                    EK[EK < 10e-16] = 0.0
                    EL = dset_spectrum_EL[it]
                    EL[EL < 10e-16] = 0.0
                    EE = dset_spectrum_EE[it]
                    EE[EE < 10e-16] = 0.0
                    Etot = EK + EL + EE

                    # print(Etot)

                    ax1.plot(kh, Etot * coef_norm, "k--", linewidth=2)
                    ax1.plot(kh, EK * coef_norm, "r--", linewidth=1)
                    ax1.plot(kh, EL * coef_norm, "b--", linewidth=1)
                    ax1.plot(kh, EE * coef_norm, "y--", linewidth=1)

            EK = dset_spectrum_EK[imin_plot : imax_plot + 1].mean(0)
            EK[EK < 10e-16] = 0.0
            EL = dset_spectrum_EL[imin_plot : imax_plot + 1].mean(0)
            EL[EK < 10e-16] = 0.0
            EE = dset_spectrum_EE[imin_plot : imax_plot + 1].mean(0)
            EE[EK < 10e-16] = 0.0

        ax1.plot(kh, EK * coef_norm, "r-", linewidth=2)
        ax1.plot(kh, EL * coef_norm, "b-", linewidth=2)
        ax1.plot(kh, EE * coef_norm, "y-", linewidth=2)

        ax1.plot(kh, kh_not0 ** (-3) * coef_norm, "k:", linewidth=1)
        ax1.plot(kh, 0.01 * kh_not0 ** (-5.0 / 3) * coef_norm, "k-.", linewidth=1)
