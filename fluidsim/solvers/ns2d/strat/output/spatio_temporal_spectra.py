"""
SpatioTempSpectra (:mod:`fluidsim.solvers.ns2d.strat.output.spatio_temporal_spectra`)
=====================================================================================


Provides:

.. autoclass:: SpatioTempSpectra
   :members:
   :private-members:

"""

import os
import sys
import time
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

from math import pi
from glob import glob
from scipy import signal

from fluiddyn.util import mpi
from fluidsim.base.output.base import SpecificOutput

# Notes
# -----
# Use scipy.signal.periodogram
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.periodogram.html

# Lindborg, Brethouwer 2007 "Stratified turbulence forced in rotational and divergent modes"


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
                "time_start": 1,
                "time_decimate": 2,
                "spatial_decimate": 2,
                "size_max_file": 0.1,
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
        self.time_start = pspatiotemp_spectra.time_start
        self.time_decimate = pspatiotemp_spectra.time_decimate
        self.spatial_decimate = pspatiotemp_spectra.spatial_decimate
        self.size_max_file = pspatiotemp_spectra.size_max_file

        self.periods_save = params.output.periods_save.spatio_temporal_spectra

        nK0, nK1 = self.sim.oper.shapeK_seq
        nK0_dec = len(list(range(0, nK0, self.spatial_decimate)))
        nK1_dec = len(list(range(0, nK1, self.spatial_decimate)))

        # Compute size in bytes of one array
        # self.size_max_file is given in Mbytes. 1 Mbyte == 1024 ** 2 bytes
        nb_bytes = np.empty([nK0_dec, nK1_dec], dtype=complex).nbytes
        self.nb_arr_in_file = int(self.size_max_file * (1024 ** 2) // nb_bytes)
        if mpi.rank == 0:
            print("nb_arr_in_file", self.nb_arr_in_file)

        # Check: duration file <= duration simulation
        self.duration_file = (
            self.nb_arr_in_file
            * self.params.time_stepping.deltat0
            * self.time_decimate
        )
        if (
            self.duration_file > self.params.time_stepping.t_end
            and self.periods_save > 0
        ):
            raise ValueError(
                "The duration of the simulation is not enough to fill a file."
            )

        # Check: self.nb_arr_in_file should be > 0
        if self.nb_arr_in_file <= 0 and self.periods_save > 0:
            raise ValueError("The size of the file should be larger.")

        else:
            # Array 4D (2 keys, times, n0, n1)
            self.spatio_temp_new = np.empty(
                [2, self.nb_arr_in_file, nK0_dec, nK1_dec], dtype=complex
            )
        # Convert time_start to it_start
        self.it_start = int(self.time_start / self.params.time_stepping.deltat0)

        # Create empty array with times
        self.times_arr = np.empty([self.nb_arr_in_file])

        if (
            params.time_stepping.USE_CFL
            and params.output.periods_save.spatio_temporal_spectra > 0
        ):
            raise ValueError(
                "To compute the spatio temporal: \n"
                + "USE_CFL = FALSE and periods_save.spatio_temporal_spectra > 0"
            )

        # Create directory to save files
        if mpi.rank == 0:
            dir_name = "spatio_temporal"
            self.path_dir = os.path.join(self.sim.output.path_run, dir_name)

            if not os.path.exists(self.path_dir):
                os.mkdir(self.path_dir)

        # Start loop in _online_save
        self.it_last_run = self.it_start
        self.nb_times_in_spatio_temp = 0

    def _init_files(self, dict_arrays_1time=None):
        # we can not do anything when this function is called.
        pass

    def _write_to_file(self, spatio_temp_arr, times_arr):
        """Writes a file with the spatio temporal data"""
        if mpi.rank == 0:
            # Name file
            it_start = int(times_arr[0] / self.sim.params.time_stepping.deltat0)
            name_file = "spatio_temp_it={}.h5".format(it_start)
            path_file = os.path.join(self.path_dir, name_file)

            # Dictionary arrays
            dict_arr = {
                "it_start": it_start,
                "times_arr": times_arr,
                "spatio_temp": spatio_temp_arr,
            }

            # Write dictionary to file
            with h5py.File(path_file, "w") as f:
                for k, v in list(dict_arr.items()):
                    f.create_dataset(k, data=v)

    def _online_save(self):
        """Computes and saves the values at one time."""
        oper = self.sim.oper
        if self.periods_save == 0:
            pass
        else:
            itsim = int(
                self.sim.time_stepping.t / self.sim.params.time_stepping.deltat0
            )

            if itsim - self.it_last_run >= self.time_decimate:
                self.it_last_run = itsim

                # Save the field to self.spatio_temp
                field_ap = self.sim.state.compute("ap_fft")
                field_am = self.sim.state.compute("am_fft")

                if mpi.nb_proc > 1:
                    # Create big array
                    if mpi.rank == 0:
                        field_ap_seq = oper.create_arrayK(shape="seq")
                        field_am_seq = oper.create_arrayK(shape="seq")
                    else:
                        field_ap_seq = None
                        field_am_seq = None

                    # Define size of each array
                    sendcounts = np.array(mpi.comm.gather(field_ap.size, root=0))

                    # Send array each process to root (process 0)
                    mpi.comm.Gatherv(
                        sendbuf=field_ap,
                        recvbuf=(field_ap_seq, sendcounts),
                        root=0,
                    )
                    mpi.comm.Gatherv(
                        sendbuf=field_am,
                        recvbuf=(field_am_seq, sendcounts),
                        root=0,
                    )

                else:
                    field_ap_seq = field_ap
                    field_am_seq = field_am

                # Decimate with process 0
                if mpi.rank == 0:
                    field_ap_decimate = field_ap_seq[
                        :: self.spatial_decimate, :: self.spatial_decimate
                    ]
                    field_am_decimate = field_am_seq[
                        :: self.spatial_decimate, :: self.spatial_decimate
                    ]

                    self.spatio_temp_new[
                        0, self.nb_times_in_spatio_temp, :, :
                    ] = field_ap_decimate
                    self.spatio_temp_new[
                        1, self.nb_times_in_spatio_temp, :, :
                    ] = field_am_decimate

                # Save the time to self.times_arr
                self.times_arr[self.nb_times_in_spatio_temp] = (
                    itsim * self.sim.params.time_stepping.deltat0
                )

                # Check if self.spatio_temp is filled. If yes, writes to a file.
                if self.nb_times_in_spatio_temp == self.nb_arr_in_file - 1:
                    if mpi.rank == 0:
                        print("Saving spatio_temporal data...")
                    self._write_to_file(self.spatio_temp_new, self.times_arr)

                    self.nb_times_in_spatio_temp = 0
                else:
                    self.nb_times_in_spatio_temp += 1

    def compute_frequency_spectra(self):
        """
        Computes and saves the frequency spectra of individual Fourier modes.
        """
        # Define list of path files
        list_files = glob(os.path.join(self.path_dir, "spatio_temp_it=*"))

        # Compute sampling frequency
        freq_sampling = 1.0 / (
            self.time_decimate * self.params.time_stepping.deltat0
        )

        for index, file_path in enumerate(list_files):

            # Generating counter
            print(
                "Computing frequency spectra = {}/{}".format(
                    index, len(list_files) - 1
                ),
                end="\r",
            )

            # Load data from file
            with h5py.File(file_path, "r") as f:
                spatio_temp = f["spatio_temp"].value
                times = f["times_arr"].value

            # Compute the temporal spectrum of a 3D array
            omegas, temp_spectrum = signal.periodogram(
                spatio_temp,
                fs=freq_sampling,
                window="hann",
                nfft=spatio_temp.shape[1],
                detrend="constant",
                return_onesided=False,
                scaling="spectrum",
                axis=1,
            )

            # Save array omegas and spectrum to file
            dict_arr = {"omegas": omegas, "temp_spectrum": temp_spectrum}

            with h5py.File(file_path, "r+") as f:
                for k, v in list(dict_arr.items()):
                    f.create_dataset(k, data=v)

            # Flush buffer and sleep time
            sys.stdout.flush()
            time.sleep(0.2)

    def print_info_frequency_spectra(self):
        """Print information frequency spectra. """

        print("""*** Info frequency spectra ***""")
        print("dt = ", self.params.time_stepping.deltat0)
        print("Decimation time = ", self.time_decimate)
        print("Number arrays in file = ", self.nb_arr_in_file)
        print("Duration file = ", self.duration_file)
        print(
            "Total number files simulation = ",
            int(
                (self.params.time_stepping.t_end - self.time_start)
                / self.duration_file
            ),
        )

    def plot_frequency_spectra_individual_mode(self, mode):
        """
        Plot frequency spectra from individual Fourier mode.

        It plots the frequency spectra of the last file.

        Parameters
        ----------
        mode : tuple
          Mode to plot (kx , kz)
        """
        # Define path file. It is the last file.
        path_file = glob(os.path.join(self.path_dir, "spatio_temp_it=*"))[0]
        print("File to plot is {} ...".format(os.path.basename(path_file)))
        print("kx_user = {:.3f} ; kz_user = {:.3f}".format(mode[0], mode[1]))

        # Load frequency spectra
        with h5py.File(path_file, "r") as f:
            if "temp_spectrum" in f.keys():
                temp_spectrum = f["temp_spectrum"].value
                omegas = f["omegas"].value
            else:
                raise ValueError(
                    "temp_spectrum not in f.keys(). " "It needs to be computed."
                )

        # Define index with spatial decimation
        idx_mode = np.argmin(
            abs(self.sim.oper.kx[:: self.spatial_decimate] - mode[0])
        )
        idz_mode = np.argmin(
            abs(self.sim.oper.ky[:: self.spatial_decimate] - mode[1])
        )
        print(
            "kx_plot = {:.3f} ; kz_plot = {:.3f}".format(
                self.sim.oper.kx[:: self.spatial_decimate][idx_mode],
                self.sim.oper.ky[:: self.spatial_decimate][idz_mode],
            )
        )
        print("ikx_mode = {} ; idz_mode = {}".format(idx_mode, idz_mode))

        # Compute omega dispersion relation mode
        kx_mode = self.sim.oper.kx[:: self.spatial_decimate][idx_mode]
        kz_mode = self.sim.oper.ky[:: self.spatial_decimate][idz_mode]
        f_iw = (
            (1 / (2 * pi))
            * self.sim.params.N
            * (kx_mode / np.sqrt(kx_mode ** 2 + kz_mode ** 2))
        )

        # Plot omega +
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel(r"$\omega / \omega_{i\omega}$")
        ax1.set_ylabel(r"$F(\omega)$")
        ax1.set_title(
            r"$\omega_+ ; (k_x, k_z) = ({:.2f}, {:.2f})$".format(
                self.sim.oper.kx[:: self.spatial_decimate][idx_mode],
                self.sim.oper.ky[:: self.spatial_decimate][idz_mode],
            )
        )

        ax1.loglog(
            omegas[0 : len(omegas) // 2] / f_iw,
            temp_spectrum[0, 0 : len(omegas) // 2, idz_mode, idx_mode],
        )
        ax1.axvline(x=f_iw / f_iw, color="k", linestyle="--")

        # Plot omega -
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel(r"$\omega / \omega_{i\omega}$")
        ax2.set_ylabel(r"$F(\omega)$")
        ax2.set_title(
            r"$\omega_- ; (k_x, k_z) = ({:.2f}, {:.2f})$".format(
                self.sim.oper.kx[:: self.spatial_decimate][idx_mode],
                self.sim.oper.ky[:: self.spatial_decimate][idz_mode],
            )
        )

        ax2.loglog(
            -1 * omegas[len(omegas) // 2 + 1 :] / f_iw,
            temp_spectrum[0, len(omegas) // 2 + 1 :, idz_mode, idx_mode],
        )
        ax2.axvline(x=f_iw / f_iw, color="k", linestyle="--")

    def _init_online_plot(self):
        if mpi.rank == 0:
            fig, axe = self.output.figure_axe(numfig=1000000)
            self.axe = axe
            axe.set_xlabel("$k_h$")
            axe.set_ylabel("$E(k_h)$")
            axe.set_title(
                "spectra, solver "
                + self.output.name_solver
                + ", nh = {0:5d}".format(self.params.oper.nx)
            )
