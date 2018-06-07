"""
SpatioTempSpectra (:mod:`fluidsim.solvers.ns2d.strat.spatio_temporal_spectra`)
===============================================================================


Provides:

.. autoclass:: SpatioTempSpectra
   :members:
   :private-members:

"""

from __future__ import division, print_function

from builtins import range

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from math import pi
from fluiddyn.util import mpi
from fluiddyn.calcul.easypyfft import FFTW1DReal2Complex, FFTW2DReal2Complex

from fluidsim.base.output.base import SpecificOutput


class SpatioTempSpectra(SpecificOutput):
    """
    Compute, save, load and plot the spatio temporal spectra.

    """

    _tag = "spatio_temporal_spectra"
    _name_file = _tag + ".h5"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "spatio_temporal_spectra"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(
            tag,
            attribs={
                "HAS_TO_PLOT_SAVED": False,
                "it_start": 10,
                "nb_times_compute": 100,
                "coef_decimate": 10,
                "overlap_arrays": 0.5,
            },
        )

    def __init__(self, output):
        params = output.sim.params
        pspatiotemp_spectra = params.output.spatio_temporal_spectra
        super(SpatioTempSpectra, self).__init__(
            output,
            period_save=params.output.periods_save.spatio_temporal_spectra,
            has_to_plot_saved=pspatiotemp_spectra.HAS_TO_PLOT_SAVED,
        )

        # Parameters
        self.nb_times_compute = pspatiotemp_spectra.nb_times_compute
        self.coef_decimate = pspatiotemp_spectra.coef_decimate
        self.it_last_run = pspatiotemp_spectra.it_start
        self.overlap_arrays = pspatiotemp_spectra.overlap_arrays

        n0 = len(
            list(range(0, output.sim.oper.shapeX_loc[0], self.coef_decimate))
        )
        n1 = len(
            list(range(0, output.sim.oper.shapeX_loc[1], self.coef_decimate))
        )

        # Initialization operators FFT and hanning windows.
        self.oper_fft2 = FFTW2DReal2Complex(n0, n1)
        self.oper_fft1 = FFTW1DReal2Complex(
            (self.nb_times_compute, n0, n1 // 2 + 1), axis=0)
        self.hamming = np.hanning(self.nb_times_compute)

        # 3D array
        self.spatio_temp = np.empty(
            [self.nb_times_compute, n0, n1 // 2 + 1], dtype=complex)
        self.spatio_temp_olap = np.empty(
            [self.nb_times_compute, n0, n1 // 2 + 1], dtype=complex)

        # Array omegas
        deltat = self.sim.time_stepping.deltat
        nt = self.nb_times_compute
        print("deltat", deltat)
        time_tot = deltat * nt
        self.delta_omega = 2 * pi / time_tot
        self.omegas = np.arange(
            0, self.delta_omega * (nt // 2 + 1), self.delta_omega)

        self.nb_times_in_spatio_temp = 0
        self.nb_times_in_spatio_temp_olap = - int(
            self.overlap_arrays * self.nb_times_compute)

    def _init_files(self, dict_arrays_1time=None):
        # we can not do anything when this function is called.
        pass

    def _init_files2(self, spatio_temp_spectra):
        """ Initialize a file to save. """

        dict_arrays_1time = {
            "omegas": self.omegas,
            "deltat": self.sim.time_stepping.deltat,
            "nb_times_compute": self.nb_times_compute}

        self._create_file_from_dict_arrays(
            self.path_file, spatio_temp_spectra, dict_arrays_1time
        )

        self.t_last_save = self.sim.time_stepping.t

    def _write_array_to_file(self, array, name="spatio_temp"):
        """Writes array to file self.path_file"""
        array_to_file = {name: array}

        if not os.path.exists(self.path_file):
            self._init_files2(array_to_file)
        else:
            self._add_dict_arrays_to_file(
                self.path_file, array_to_file
            )

    def _online_save(self):
        """Computes and saves the values at one time."""
        itsim = int(self.sim.time_stepping.t / self.sim.time_stepping.deltat)
        periods_save = self.sim.params.output.periods_save.spatio_temporal_spectra
        nb_times_compute = self.nb_times_compute

        if itsim - self.it_last_run >= periods_save - 1:
            self.it_last_run = itsim
            field = self.sim.state.compute("ap_fft")
            field_decimate = field[::self.coef_decimate, ::self.coef_decimate]
            self.spatio_temp[self.nb_times_in_spatio_temp, :, :] = field_decimate

            # Check start overlapped array
            nb_times_olap = self.nb_times_in_spatio_temp_olap
            condition = nb_times_olap >= 0 and nb_times_olap < nb_times_compute
            if condition:
                field = self.sim.state.compute("ap_fft")
                field_decimate = field[::self.coef_decimate, ::self.coef_decimate]
                self.spatio_temp_olap[nb_times_olap, :, :] = field_decimate

            # Add index
            self.nb_times_in_spatio_temp += 1
            self.nb_times_in_spatio_temp_olap += 1

            # Check and write overlapped file
            if self.nb_times_in_spatio_temp_olap == nb_times_compute:
                self.nb_times_in_spatio_temp_olap = 0
                self._write_array_to_file(self.spatio_temp_olap)

            # Check and write file
            if self.nb_times_in_spatio_temp == nb_times_compute:
                self.nb_times_in_spatio_temp = 0
                self._write_array_to_file(self.spatio_temp)

    def load(self):
        """Loads spatio temporal fft from file. """
        with h5py.File(self.path_file, "r") as f:
            spatio_temp = f["spatio_temp"].value
        return spatio_temp

    # def _compute_energy_from_spatio_temporal_fft(self, spatio_temporal_fft):
    #     """
    #     Compute energy at each fourier mode from spatio temporal fft.
    #     """
    #     return (1 / 2.) * np.abs(spatio_temporal_fft) ** 2

    # def plot_frequency_spectra(self):
    #     """ Plots the frequency spectra F(\omega). """
    #     # Load data from file
    #     spatio_temporal_fft = self.load()

    #     # Average over all spatio_temporal_fft 3d arrays
    #     spatio_temporal_fft = spatio_temporal_fft.mean(axis=0)

    #     # Compute energy from spatio_temporal_fft
    #     energy_fft = self._compute_energy_from_spatio_temporal_fft(
    #         spatio_temporal_fft)

    #     omegas =  np.arange(
    #         0, self.delta_omega * (self.nb_times_compute// 2 + 1), self.delta_omega)

    #     E_omega = (1 / self.delta_omega) * energy_fft.sum(1).sum(1)

    #     # Plot
    #     fig, ax = plt.subplots()
    #     ax.set_xlabel("$\omega$")
    #     ax.set_ylabel(r"$F(\omega)$")
    #     ax.set_title(r"$E(\omega)$")
    #     ax.loglog(omegas, E_omega)

    #     plt.show()

    # def plot_omega_kx(self):
    #     """ Plots the frequency spectra F(\omega). """
    #     # Load data from file
    #     spatio_temporal_fft = self.load()

    #     # Average over all spatio_temporal_fft 3d arrays
    #     spatio_temporal_fft = spatio_temporal_fft.mean(axis=0)

    #     # Compute energy from spatio_temporal_fft
    #     energy_fft = self._compute_energy_from_spatio_temporal_fft(
    #         spatio_temporal_fft)

    #     delta_kx =  2 * pi / self.oper.Lx
    #     nx_decimate = len(
    #         list(range(0, self.sim.oper.shapeX_loc[1], self.coef_decimate)))

    #     kxs = np.arange(0, delta_kx * (nx_decimate // 2 + 1), delta_kx)
    #     omegas =  np.arange(
    #         0, self.delta_omega * (self.nb_times_compute// 2 + 1), self.delta_omega)

    #     E_omega_kx = (1 / self.delta_omega) * (1 / delta_kx) * energy_fft.sum(1)

    #     # Plot
    #     kx_grid, omega_grid = np.meshgrid(kxs, omegas)

    #     fig, ax = plt.subplots()
    #     ax.set_xlabel("$\omega$")
    #     ax.set_ylabel("$k_x$")
    #     ax.set_title(r"$E(\omega, k_x)$")
    #     ax.pcolor(omega_grid, kx_grid, E_omega_kx)

    #     plt.show()
