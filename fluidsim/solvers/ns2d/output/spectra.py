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
        dict_spectra1D = {
            "spectrum1Dkx_E": spectrum1Dkx_E,
            "spectrum1Dky_E": spectrum1Dky_E,
        }
        # compute the spectra 2D
        spectrum2D_E = self.spectrum2D_from_fft(energy_fft)
        dict_spectra2D = {"spectrum2D_E": spectrum2D_E}
        return dict_spectra1D, dict_spectra2D

    def _online_plot_saving(self, dict_spectra1D, dict_spectra2D):
        if (
            self.nx == self.params.oper.ny
            and self.params.oper.Lx == self.params.oper.Ly
        ):
            spectrum2D = dict_spectra2D["spectrum2D_E"]
            khE = self.oper.khE
            coef_norm = khE ** (3.0)
            self.axe.loglog(khE, spectrum2D * coef_norm, "k")
            lin_inf, lin_sup = self.axe.get_ylim()
            if lin_inf < 10e-6:
                lin_inf = 10e-6
            self.axe.set_ylim([lin_inf, lin_sup])
        else:
            print(
                "you need to implement the ploting "
                "of the spectra for this case"
            )

    def plot1d(self, tmin=0, tmax=1000, delta_t=2, coef_compensate=3):

        with h5py.File(self.path_file1D, "r") as h5file:
            dset_times = h5file["times"]

            dset_kxE = h5file["kxE"]
            kh = dset_kxE[...]
            kh2 = kh[:]
            kh2[kh == 0] = 1e-15

            dset_spectrum1Dkx = h5file["spectrum1Dkx_E"]
            dset_spectrum1Dky = h5file["spectrum1Dky_E"]
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
                """plot 1D spectra
    tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
    imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
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

            EKx = dset_spectrum1Dkx[0]
            EKy = dset_spectrum1Dky[0]

            is_asym = len(EKx) == len(EKy)

            coef_norm = kh2 ** (coef_compensate)
            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot + 1, delta_i_plot):
                    EK = dset_spectrum1Dkx[it]
                    if is_asym:
                        EK += dset_spectrum1Dky[it]

                    EK[EK < 10e-16] = 0.0
                    ax1.plot(kh, EK * coef_norm, "k", linewidth=2)

            EK = dset_spectrum1Dkx[imin_plot : imax_plot + 1]
            if is_asym:
                EK += dset_spectrum1Dky[imin_plot : imax_plot + 1]

        EK = EK.mean(0)

        ax1.plot(kh, kh2 ** (-3) * coef_norm, "k", linewidth=1)
        ax1.plot(kh, 0.01 * kh2 ** (-5 / 3) * coef_norm, "k--", linewidth=1)

    def plot2d(self, tmin=0, tmax=1000, delta_t=2, coef_compensate=3):
        with h5py.File(self.path_file2D, "r") as h5file:
            dset_times = h5file["times"]
            # nb_spectra = dset_times.shape[0]
            times = dset_times[...]
            # nt = len(times)

            kh = h5file["khE"][...]
            kh2 = kh[:]
            kh2[kh == 0] = 1e-15

            dset_spectrum = h5file["spectrum2D_E"]

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
                "plot2s(tmin={}, tmax={}, delta_t={:.2f},".format(
                    tmin, tmax, delta_t
                )
                + f" coef_compensate={coef_compensate:.3f})"
            )

            print(
                """plot 2D spectra
    tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
    imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
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

            coef_norm = kh2**coef_compensate

            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot + 1, delta_i_plot):
                    EK = dset_spectrum[it]
                    EK[EK < 10e-16] = 0.0
                    ax1.plot(kh, EK * coef_norm, "k", linewidth=1)

            EK = dset_spectrum[imin_plot : imax_plot + 1].mean(0)
            EK[EK < 10e-16] = 0.0
            ax1.plot(kh, EK * coef_norm, "k", linewidth=2)

            ax1.plot(kh, kh2 ** (-3) * coef_norm, "k--", linewidth=1)
            ax1.plot(kh, 0.01 * kh2 ** (-5.0 / 3) * coef_norm, "k-.", linewidth=1)
