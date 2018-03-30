"""
SpatioTempSpectra (:mod:`fluidsim.solvers.ns2d.strat.spatio_temporal_spectra`)
===============================================================================


Provides:

.. autoclass:: SpatioTempSpectra
   :members:
   :private-members:

#TODO: key_quantity == linear eigenmode a

"""

from __future__ import division, print_function

from builtins import range

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from fluiddyn.util import mpi
from fluiddyn.calcul.easypyfft import FFTW1DReal2Complex, \
    FFTW2DReal2Complex

from fluidsim.base.output.base import SpecificOutput


class SpatioTempSpectra(SpecificOutput):
    """
    Compute, save, load and plot the spatio temporal spectra.

    """

    _tag = 'spatio_temporal_spectra'
    _name_file = _tag + '.h5'

    @staticmethod
    def _complete_params_with_default(params):
        tag = 'spatio_temporal_spectra'

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag,
                                 attribs={
                                     'HAS_TO_PLOT_SAVED': False,
                                     'it_start': 10,
                                     'nb_times_compute': 100,
                                     'coef_decimate': 10,
                                     'key_quantity': 'ux'})

    def __init__(self, output):
        params = output.sim.params
        pspatiotemp_spectra = params.output.spatio_temporal_spectra
        super(SpatioTempSpectra, self).__init__(
            output,
            period_save=params.output.periods_save.spatio_temporal_spectra,
            has_to_plot_saved=pspatiotemp_spectra.HAS_TO_PLOT_SAVED)

        # Parameters
        self.nb_times_compute = pspatiotemp_spectra.nb_times_compute
        self.coef_decimate = pspatiotemp_spectra.coef_decimate
        self.key_quantity = pspatiotemp_spectra.key_quantity
        self.it_last_run = pspatiotemp_spectra.it_start

        n0 = len(list(range(0, output.sim.oper.shapeX_loc[0],
                            self.coef_decimate)))
        n1 = len(list(range(0, output.sim.oper.shapeX_loc[1],
                            self.coef_decimate)))
        print('n0', n0)
        print('n1', n1)
        # 3D array (time, x, y) and init FFTW object
        # self.spatio_temp = np.empty([self.nb_times_compute, n0, n1])
        self.spatio_temp = np.empty([self.nb_times_compute, n0, n1 // 2 + 1])
        self.oper_fft2 = FFTW2DReal2Complex(n0, n1)
        self.oper_fft1 = FFTW1DReal2Complex(self.spatio_temp.shape, axis=0)
        self.nb_omegas = self.oper_fft1.shapeK[0]
        self.hamming = np.hanning(self.nb_times_compute)

        # Compute kxs and kys with the decimate values
        deltakx = 2 * np.pi / self.sim.oper.Lx
        self.kxs_decimate = np.arange(0, deltakx * (n0/2) + deltakx, deltakx)

        self.nb_times_in_spatio_temp = 0

        if os.path.exists(self.path_file):
            with h5py.File(self.path_file, 'r') as f:
                link_spatio_temp_spectra = f['spatio_temp_spectra']
                self.spatio_temp_spectra = link_spatio_temp_spectra[-1]
                self.periods_fill = f['periods_fill'][...]
                if self.sim.time_stepping.deltat != f['deltat'][...]:
                    raise ValueError()
        else:

            self.periods_fill = \
                            params.output.periods_save.spatio_temporal_spectra

        if self.periods_fill > 0:
            # self.periods_fill = self.periods_fill - 1 
            dt_output = self.periods_fill * output.sim.time_stepping.deltat
            print('dt_output = ', dt_output)
            duration = self.nb_times_compute * dt_output
            self.delta_omega = 2 * np.pi / duration
            print('duration = ', duration)
            print('delta_omega = ', self.delta_omega)
            self.omegas = self.delta_omega * np.arange(self.nb_omegas)

            self.omega_Nyquist = np.pi / dt_output

            self.omega_dealiasing = (
                self.params.oper.coef_dealiasing * np.pi *
                self.params.oper.nx / self.params.oper.Lx)**2

            if self.omega_dealiasing > self.omega_Nyquist:
                print('Warning: omega_dealiasing > omega_Nyquist')

    def _init_files(self, dico_arrays_1time=None):
        # we can not do anything when this function is called.
        pass

    def _init_files2(self, spatio_temp_spectra):
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
            self.path_file, spatio_temp_spectra, dico_arrays_1time)

        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Computes and saves the values at one time."""
        itsim = int(self.sim.time_stepping.t / self.sim.time_stepping.deltat)
        periods_save = \
                self.sim.params.output.periods_save.spatio_temporal_spectra

        # print('it_sim = ', itsim)
        # print('it_sim - it_last_run = ', itsim - self.it_last_run)
        # print('periods_save = ', periods_save - 1)

        if (itsim - self.it_last_run >= periods_save - 1):
            # print('#save_period')
            self.it_last_run = itsim
            field = self.sim.state.state_phys.get_var(self.key_quantity)
            field_decimate = field[::self.coef_decimate, ::self.coef_decimate]
            field_fft = self.oper_fft2.fft2d(field_decimate)
            self.spatio_temp[self.nb_times_in_spatio_temp, :, :] = field_fft
            self.nb_times_in_spatio_temp += 1

            if self.nb_times_in_spatio_temp == self.nb_times_compute:
                # print('#####save_spatio_temporal...')
                self.nb_times_in_spatio_temp = 0
                self.t_last_save = self.sim.time_stepping.t

                # It is not the best way to apply hanning.
                for i, value in enumerate(self.hamming):
                    self.spatio_temp[i, :, :] = value * \
                                                self.spatio_temp[i, :, :]
                # self.spatio_fft = self.oper_fft1.fft(
                #     self.hamming * self.spatio_temp)
                self.spatio_fft = self.oper_fft1.fft(self.spatio_temp)

                if mpi.rank == 0:
                    spatio_temp_spectra = {'spatio_temp_spectra': \
                                           self.spatio_fft}

                    if not os.path.exists(self.path_file):
                            self._init_files2(spatio_temp_spectra)
                    else:
                        self.add_dico_arrays_to_file(self.path_file,
                                                     spatio_temp_spectra)

    def load(self):
        """Loads spatio temporal fft from file. """
        with h5py.File(self.path_file, 'r') as f:
            spatio_temporal_fft = f['spatio_temp_spectra'].value
        return spatio_temporal_fft

    def _compute_energy_from_spatio_temporal_fft(self, spatio_temporal_fft):
        """
        Compute energy at each fourier mode from spatio temporal fft.
        """
        return (1/2.) * np.abs(spatio_temporal_fft)**2

    def compute_spatio_temporal_spectra(self, spatio_temporal_fft):
        """Compute the spectra (kx, omega) and (ky, omega)"""
        energy_fft = self._compute_energy_from_spatio_temporal_fft(
            spatio_temporal_fft)

        # Axes of the spatio temporal fft.
        omega_axis = 0
        ky_axis = 1
        kx_axis = 2

        delta_kx = self.oper.deltakx
        delta_ky = self.oper.deltaky

        # Compute energy spectra (kx, omega).
        # We multiply by 2 and 2 because there are only omega>=0 and kx>=0.
        # We divide by two because the energy at the zero modes shoud be
        # counted only once.
        E_kx_omega = 2. * 2. * energy_fft.sum(ky_axis) / (
            delta_kx * self.delta_omega)

        E_kx_omega[0, :] = E_kx_omega[0, :] / 2.
        E_kx_omega[:, 0] = E_kx_omega[:, 0] / 2.

        return E_kx_omega

    def plot(self):
        """ Plot the spatio temporal spectra. """
        # Load data from file.
        # Spatio_temporal_fft is an array with all spatio_temporal_fft.
        # We'll need to average all of them.
        spatio_temporal_fft = self.load()

        # Average all spatio_temporal_fft. Axis = 0
        spatio_temporal_fft = spatio_temporal_fft.mean(axis=0)

        # Compute the spatio_temporal_spectra
        E_kx_omega = self.compute_spatio_temporal_spectra(spatio_temporal_fft)
        # print('E_kx_omega = ', E_kx_omega)

        # Data grid
        omegas = self.omegas

        kx_grid, omegas_grid = np.meshgrid(self.kxs_decimate, omegas)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel('kx')
        ax.set_ylabel('omega')
        ax.set_title('E_kx_omega')

        # ax.pcolor(
        #     kx_grid, omegas_grid, E_kx_omega, \
        #     vmin=E_kx_omega.min(), vmax=E_kx_omega.max())

        ax.pcolor(
            kx_grid, omegas_grid, E_kx_omega, \
            vmin=1e-5, vmax=E_kx_omega.max())

        plt.show()
