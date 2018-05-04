"""Spectra output (:mod:`fluidsim.solvers.ns2d.strat.output.spectra`)
=====================================================================

.. autoclass:: SpectraNS2DStrat
   :members:
   :private-members:

"""
from __future__ import print_function

import h5py

import numpy as np

from fluidsim.base.output.spectra import Spectra


class SpectraNS2DStrat(Spectra):
    """Save and plot spectra."""

    def compute(self):
        """compute the values at one time."""
        # energy_fft = self.output.compute_energy_fft()
        energyK_fft, energyA_fft = self.output.compute_energies_fft()
        energy_fft = energyK_fft + energyA_fft
        energyK_ux_fft, energyK_uy_fft = self.output.compute_energies2_fft()
        energyK, energyA, energyK_ux = self.output.compute_energies()

        # Compute the kinetic energy spectra 1D for the two velocity components
        # and two directions
        spectrum1Dkx_EK_ux, spectrum1Dky_EK_ux = self.spectra1D_from_fft(
            energyK_ux_fft
        )
        spectrum1Dkx_EK_uy, spectrum1Dky_EK_uy = self.spectra1D_from_fft(
            energyK_uy_fft
        )
        spectrum1Dkx_EK, spectrum1Dky_EK = self.spectra1D_from_fft(energyK_fft)

        # Compute the potential energy spectra 1D two directions
        spectrum1Dkx_EA, spectrum1Dky_EA = self.spectra1D_from_fft(energyA_fft)

        # Compute the total energy spectra 1D
        spectrum1Dkx_E, spectrum1Dky_E = self.spectra1D_from_fft(energy_fft)
        # Dictionary with the 1D kinetic energy spectra
        dict_spectra1D = {
            "spectrum1Dkx_EK_ux": spectrum1Dkx_EK_ux,
            "spectrum1Dky_EK_ux": spectrum1Dky_EK_ux,
            "spectrum1Dkx_EK_uy": spectrum1Dkx_EK_uy,
            "spectrum1Dky_EK_uy": spectrum1Dky_EK_uy,
            "spectrum1Dkx_EK": spectrum1Dkx_EK,
            "spectrum1Dky_EK": spectrum1Dky_EK,
            "spectrum1Dkx_EA": spectrum1Dkx_EA,
            "spectrum1Dky_EA": spectrum1Dky_EA,
            "spectrum1Dkx_E": spectrum1Dkx_E,
            "spectrum1Dky_E": spectrum1Dky_E,
        }

        # compute the kinetic energy spectra 2D
        spectrum2D_E = self.spectrum2D_from_fft(energy_fft)
        spectrum2D_EK = self.spectrum2D_from_fft(energyK_fft)
        spectrum2D_EK_ux = self.spectrum2D_from_fft(energyK_ux_fft)
        spectrum2D_EK_uy = self.spectrum2D_from_fft(energyK_uy_fft)
        spectrum2D_EA = self.spectrum2D_from_fft(energyA_fft)
        dict_spectra2D = {
            "spectrum2D_EK_ux": spectrum2D_EK_ux,
            "spectrum2D_EK_uy": spectrum2D_EK_uy,
            "spectrum2D_EK": spectrum2D_EK,
            "spectrum2D_EA": spectrum2D_EA,
            "spectrum2D_E": spectrum2D_E,
        }

        return dict_spectra1D, dict_spectra2D

    def _online_plot_saving(self, dict_spectra1D, dict_spectra2D):
        if (
            self.nx == self.params.oper.ny
            and self.params.oper.Lx == self.params.oper.Ly
        ):
            spectrum2D_EK = dict_spectra2D["spectrum2D_EK"]
            spectrum2D_EA = dict_spectra2D["spectrum2D_EA"]
            spectrum2D = spectrum2D_EK + spectrum2D_EA
            khE = self.oper.khE
            coef_norm = khE ** (3.)
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

    def plot1d(
        self,
        tmin=0,
        tmax=1000,
        delta_t=2,
        coef_compensate_kx=5. / 3,
        coef_compensate_kz=3.,
    ):
        """Plot spectra one-dimensional."""

        f = h5py.File(self.path_file1D, "r")

        # Open data from file
        dset_times = f["times"]
        dset_kxE = f["kxE"]
        dset_kyE = f["kyE"]
        times = dset_times.value
        kx = dset_kxE.value
        ky = dset_kyE.value

        # Open data set 1D spectra
        dset_spectrum1Dkx_EA = f["spectrum1Dkx_EA"]
        dset_spectrum1Dky_EA = f["spectrum1Dky_EA"]
        dset_spectrum1Dkx_EK = f["spectrum1Dkx_EK"]
        dset_spectrum1Dky_EK = f["spectrum1Dky_EK"]
        dset_spectrum1Dkx_E = f["spectrum1Dkx_E"]
        dset_spectrum1Dky_E = f["spectrum1Dky_E"]

        # Compute average from tmin and tmax for plot
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
            "plot1d(tmin={0}, tmax={1}, delta_t={2:.2f},".format(
                tmin, tmax, delta_t
            )
            + " coef_compensate_kx={0:.3f})".format(coef_compensate_kx)
        )

        print(
            """plot 1D spectra
        tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
        imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}""".format(
                tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
            )
        )

        # Parameters figure E(k_x)
        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel("$k_x$")
        ax1.set_ylabel(r"$E(k_x)k_x^{{{}}}$".format(round(coef_compensate_kx, 2)))
        ax1.set_title(
            "1D spectra, solver "
            + self.output.name_solver
            + ", nh = {0:5d}".format(self.nx)
        )
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        # ax1.set_ylim(ymin=1e-6, ymax=1e3)

        # Average in time between tmin and tmax
        E_kx = (dset_spectrum1Dkx_E[imin_plot:imax_plot + 1]).mean(0)
        EK_kx = (dset_spectrum1Dkx_EK[imin_plot:imax_plot + 1]).mean(0)
        EA_kx = (dset_spectrum1Dkx_EA[imin_plot:imax_plot + 1]).mean(0)

        id_kx_dealiasing = np.argmin(abs(kx - self.sim.oper.kxmax_dealiasing)) - 1
        id_ky_dealiasing = np.argmin(abs(ky - self.sim.oper.kymax_dealiasing)) - 1

        # Remove modes dealiased.
        E_kx_plot = E_kx[:id_kx_dealiasing]
        EK_kx_plot = EK_kx[:id_kx_dealiasing]
        EA_kx_plot = EA_kx[:id_kx_dealiasing]
        kx_plot = kx[:id_kx_dealiasing]

        # Remove shear modes if there is no energy on them.
        if self.sim.params.oper.NO_SHEAR_MODES:
            E_kx_plot = E_kx_plot[1:]
            EK_kx_plot = EK_kx_plot[1:]
            EA_kx_plot = EA_kx_plot[1:]
            kx_plot = kx_plot[1:]

        ax1.plot(kx_plot, E_kx_plot * kx_plot ** coef_compensate_kx, label="E")
        ax1.plot(kx_plot, EK_kx_plot * kx_plot ** coef_compensate_kx, label="EK")
        ax1.plot(kx_plot, EA_kx_plot * kx_plot ** coef_compensate_kx, label="EA")

        # Parameters figure E(k_y)
        fig, ax2 = self.output.figure_axe()
        ax2.set_xlabel(r"$k_z$")
        ax2.set_ylabel(r"$E(k_z)k_z^{{{}}}$".format(round(coef_compensate_kz, 2)))
        ax2.set_title(
            "1D spectra, solver "
            + self.output.name_solver
            + ", nh = {0:5d}".format(self.nx)
        )

        ax2.set_xscale("log")
        ax2.set_yscale("log")

        E_ky = (dset_spectrum1Dky_E[imin_plot:imax_plot + 1]).mean(0)
        EK_ky = (dset_spectrum1Dky_EK[imin_plot:imax_plot + 1]).mean(0)
        EA_ky = (dset_spectrum1Dky_EA[imin_plot:imax_plot + 1]).mean(0)

        # Remove modes dealiased.
        E_ky_plot = E_ky[:id_ky_dealiasing]
        EK_ky_plot = EK_ky[:id_ky_dealiasing]
        EA_ky_plot = EA_ky[:id_ky_dealiasing]
        ky_plot = ky[:id_ky_dealiasing]

        # Remove shear modes if there is no energy on them.
        if self.sim.params.oper.NO_SHEAR_MODES:
            E_ky_plot = E_ky_plot[1:]
            EK_ky_plot = EK_ky_plot[1:]
            EA_ky_plot = EA_ky_plot[1:]
            ky_plot = ky_plot[1:]

        ax2.plot(ky_plot, E_ky_plot * ky_plot ** coef_compensate_kz, label="E")
        ax2.plot(ky_plot, EK_ky_plot * ky_plot ** coef_compensate_kz, label="EK")
        ax2.plot(ky_plot, EA_ky_plot * ky_plot ** coef_compensate_kz, label="EA")

        # Plot forcing wave-number k_f
        nkmax = self.sim.params.forcing.nkmax_forcing
        nkmin = self.sim.params.forcing.nkmin_forcing
        k_f = ((nkmax + nkmin) / 2) * self.sim.oper.deltak
        try:
            angle = self.sim.forcing.forcing_maker.angle
        except AttributeError:
            pass
        else:
            k_fx = np.sin(angle) * k_f
            k_fy = np.cos(angle) * k_f
            ax1.axvline(x=k_fx, color="k", linestyle=":", label="$k_{f,x}$")
            ax2.axvline(x=k_fy, color="k", linestyle=":", label="$k_{f,z}$")

        ax1.legend()
        ax2.legend()

    def plot2d(self, tmin=0, tmax=1000, delta_t=2, coef_compensate=3):
        """Plot 2D spectra."""

        # Load data from file
        f = h5py.File(self.path_file2D, "r")
        dset_times = f["times"]
        kh = f["khE"].value
        times = dset_times.value

        dset_spectrum_E = f["spectrum2D_E"]
        dset_spectrum_EK = f["spectrum2D_EK"]
        dset_spectrum_EA = f["spectrum2D_EA"]

        # Compute average from tmin and tmax for plot
        delta_t_save = np.mean(times[1:] - times[0:-1])
        delta_i_plot = int(np.round(delta_t / delta_t_save))
        if delta_i_plot == 0 and delta_t != 0.:
            delta_i_plot = 1
        delta_t = delta_i_plot * delta_t_save

        imin_plot = np.argmin(abs(times - tmin))
        imax_plot = np.argmin(abs(times - tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]

        print(
            "plot2s(tmin={0}, tmax={1}, delta_t={2:.2f},".format(
                tmin, tmax, delta_t
            )
            + " coef_compensate={0:.3f})".format(coef_compensate)
        )

        print(
            """plot 2D spectra
        tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
        imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}""".format(
                tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
            )
        )

        # Parameters figure
        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel(r"$k$")
        ax1.set_ylabel(r"$E(k)$")
        ax1.set_title(
            "2D spectra, solver "
            + self.output.name_solver
            + ", nh = {0:5d}".format(self.nx)
        )

        ax1.set_xscale("log")
        ax1.set_yscale("log")

        E = dset_spectrum_E[imin_plot:imax_plot + 1].mean(0)
        EK = dset_spectrum_EK[imin_plot:imax_plot + 1].mean(0)
        EA = dset_spectrum_EA[imin_plot:imax_plot + 1].mean(0)

        ax1.plot(kh, E, label="E")
        ax1.plot(kh, EK, label="EK")
        ax1.plot(kh, EA, label="EA")

        ax1.legend()
