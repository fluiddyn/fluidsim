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
from pathlib import Path

from fluiddyn.util import mpi
from fluidsim import load_state_phys_file
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
            self.nb_arr_in_file * self.params.time_stepping.deltat0 * self.time_decimate
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
        # Convert time_start to it_start.
        if not self.sim.time_stepping.it:
            self.it_start = int(self.time_start / self.params.time_stepping.deltat0)
        else:
            # If simulation starts from an specific time...
            self.it_start = self.sim.time_stepping.it + int(
                (self.time_start - self.sim.time_stepping.t)
                / self.params.time_stepping.deltat0
            )

            if self.it_start < self.sim.time_stepping.it:
                self.it_start = self.sim.time_stepping.it

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
            itsim = self.sim.time_stepping.it
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
                        sendbuf=field_ap, recvbuf=(field_ap_seq, sendcounts), root=0
                    )
                    mpi.comm.Gatherv(
                        sendbuf=field_am, recvbuf=(field_am_seq, sendcounts), root=0
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

    def compute_frequency_spectra(
        self, tmin=None, tmax=None, nb_periods_N=None, overlap=0.5
    ):
        """Computes the temporal frequency spectrum.

        Keyword Arguments:
            tmin {float} -- Minimum time to perform temporal FT. (default: {None})
            tmax {float} -- Maximum time to perform temporal FT. [description] (default: {None})
            nb_periods_N {int} -- Number of periods of N to compute windows (default: {None})
            overlap {float} -- Overlap between windows. From 0 to 1. (default: {0.5})
        """

        # Get concatenated array of times and spatio temporal
        times, spatio_temp = self._get_concatenate_data(tmin, tmax)

        # Compute time_windows
        if not nb_periods_N:
            nb_periods_N = 10

        windows_size = int(
            2
            * pi
            / self.sim.params.N
            * nb_periods_N
            * (1 / self.sim.params.time_stepping.deltat0 / self.time_decimate)
        )

        if windows_size > len(times):
            windows_size = len(times)

        # Check argument overlap
        if overlap < 0 or overlap > 1:
            raise ValueError("Overlap should be between 0 and 1.")

        # Compute number of indexes to skip
        skip = int((1 - overlap) * windows_size)

        # Slice step cannot be zero
        if not skip or skip == 1:
            skip = 2

        # Array of times to start the FT
        times_start = times[:: skip - 1]
        times_start = times_start[:-1]

        # Create array to allocate all FT's
        spatio_temp_all = np.empty(
            [
                len(times_start),
                2,
                windows_size,
                spatio_temp.shape[2],
                spatio_temp.shape[3],
            ],
            dtype=complex,
        )

        # Compute sampling frequency in Hz
        freq_sampling = 1.0 / (self.time_decimate * self.params.time_stepping.deltat0)

        # Compute temporal FT
        for index, time_start in enumerate(times_start):

            # Compute index to start and to end FT
            it0 = np.argmin(abs(times - time_start))
            it1 = it0 + windows_size

            # Compute spectrum
            omegas, temp_spectrum = signal.periodogram(
                spatio_temp[:, it0:it1, :, :],
                fs=freq_sampling,
                window="hann",
                nfft=windows_size,
                detrend="constant",
                return_onesided=False,
                scaling="spectrum",
                axis=1,
            )

            # Allocate FT
            spatio_temp_all[index, :, :, :, :] = temp_spectrum

        # Build up a dictionary with the rsults
        dict_arr = {
            "omegas": omegas,
            "spectrum": np.mean(spatio_temp_all, axis=0),
            "windows_size": windows_size,
        }

        # Create name path file
        str_time = time.strftime("%Y_%m_%d-%H_%M_%S")
        name_file = f"spectrum_{str_time}.h5"

        # Create directory if does not exist
        path_dir = Path(self.path_dir) / "Spectrum"

        if not path_dir.exists():
            os.mkdir(path_dir)

        path_file = path_dir / name_file

        # Save file into directory spatio_temporal/Spectrum
        with h5py.File(path_file, "w") as f:
            for k, v in list(dict_arr.items()):
                f.create_dataset(k, data=v)

    def _get_concatenate_data(self, tmin, tmax):
        """Gives concatenated time and spatio_temp arrays/

        Arguments:
            tmin {float} -- Minimum time to perform temporal FT.
            tmax {float} -- Maximum time to perform temporal FT.

        Raises:
            ValueError -- itmin == itmax

        Returns:
            times {array} -- Concatenated time array
            spatio_temp {array} -- concatenated spatio temporal array
        """

        # Define list files
        list_files = sorted(Path(self.path_dir).glob("spatio_temp_it=*"))

        # Define concatenate arrays
        spatio_temp_conc = None
        times_conc = None

        # Load all data
        for index, path_file in enumerate(list_files):
            with h5py.File(path_file, "r") as f:
                spatio_temp = f["spatio_temp"].value
                times = f["times_arr"].value

            # Print concatenating info..
            print(f"Concatenating file = {index + 1}/{len(list_files)}..", end="\r")

            # Concatenate arrays
            if isinstance(spatio_temp_conc, np.ndarray):
                np.concatenate((spatio_temp_conc, spatio_temp), axis=1)
            elif isinstance(times_conc, np.ndarray):
                np.concatenate((times_conc, times), axis=0)
            else:
                spatio_temp_conc = spatio_temp
                times_conc = times

            sys.stdout.flush()
            time.sleep(0.2)

        # Default tmin and tmax
        if not tmin:
            tmin = np.min(times_conc)
        if not tmax:
            tmax = np.max(times_conc)

        # Find indexes itmin and itmax
        itmin = np.argmin(abs(times_conc - tmin))
        itmax = np.argmin(abs(times_conc - tmax)) + 1

        # Check..
        if itmin == itmax:
            raise ValueError(
                "itmin == itmax. \n"
                + f"Choose between tmin = {np.min(times_conc)} - "
                + f"tmax = {np.max(times_conc)}"
            )

        # Print info..
        to_print = f"tmin = {np.min(times_conc)} ; tmax = {np.max(times_conc)}"
        print(to_print)

        return times_conc[itmin:itmax], spatio_temp_conc[:, itmin:itmax, :, :]

    # def compute_frequency_spectra(
    # self, tmin=None, tmax=None, windows_size=None, overlap=None
    # ):
    # """
    # Computes and saves the frequency spectra of individual Fourier modes.
    # """
    # # Define list of path files
    # list_files = glob(os.path.join(self.path_dir, "spatio_temp_it=*"))

    # # Compute sampling frequency
    # freq_sampling = 1.0 / (self.time_decimate * self.params.time_stepping.deltat0)

    # for index, file_path in enumerate(list_files):

    # # Generating counter
    # print(
    # "Computing frequency spectra = {}/{}".format(
    # index, len(list_files) - 1
    # ),
    # end="\r",
    # )

    # # Load data from file
    # with h5py.File(file_path, "r") as f:
    # spatio_temp = f["spatio_temp"].value
    # times = f["times_arr"].value

    # # Compute the temporal spectrum of a 3D array
    # omegas, temp_spectrum = signal.periodogram(
    # spatio_temp,
    # fs=freq_sampling,
    # window="hann",
    # nfft=spatio_temp.shape[1],
    # detrend="constant",
    # return_onesided=False,
    # scaling="spectrum",
    # axis=1,
    # )

    # # Save array omegas and spectrum to file
    # dict_arr = {"omegas": omegas, "temp_spectrum": temp_spectrum}

    # with h5py.File(file_path, "r+") as f:
    # for k, v in list(dict_arr.items()):
    # f.create_dataset(k, data=v)

    # # Flush buffer and sleep time
    # sys.stdout.flush()
    # time.sleep(0.2)

    def plot_kx_omega_cross_section(
        self, path_file=None, field="ap_fft", ikz_plot=None, func_plot="pcolormesh"
    ):
        # Define path dir as posix path
        path_dir = Path(self.path_dir)

        # If path_file does not exist.
        if not path_file:
            path_file = sorted((path_dir / "Spectrum").glob("spectrum*"))[-1]

        # Load data
        with h5py.File(path_file, "r") as f:
            temp_spectrum_mean = f["spectrum"].value
            omegas = f["omegas"].value

        # Load simulation object
        sim = load_state_phys_file(path_dir.parent, merge_missing_params=True)

        # Compute kx_decimate and ky_decimate
        kx_decimate = sim.oper.kx[
            :: sim.params.output.spatio_temporal_spectra.spatial_decimate
        ]
        kz_decimate = sim.oper.ky[
            :: sim.params.output.spatio_temporal_spectra.spatial_decimate
        ]

        # Omegas
        omegas *= 2 * np.pi
        omegas *= 1 / (sim.params.N)

        #### PLOT OMEGA - KX
        kxmin_plot = 0
        kxmax_plot = 80

        ikxmin_plot = np.argmin(abs(kx_decimate - kxmin_plot))
        ikxmax_plot = np.argmin(abs(kx_decimate - kxmax_plot)) + 1

        omega_max_plot = 0
        omega_min_plot = -3

        omegas = np.append(omegas, 0)
        imin_omegas_plot = np.argmin(abs(omegas - omega_min_plot)) - 1

        kxs_grid, omegas_grid = np.meshgrid(
            kx_decimate[ikxmin_plot:ikxmax_plot], omegas[imin_omegas_plot:]
        )

        if not ikz_plot:
            ikz_plot = 1

        # Parameters figure
        fig, ax = plt.subplots()
        ax.set_xlim(left=kxmin_plot, right=kxmax_plot)
        ax.set_ylim(top=omega_max_plot, bottom=omega_min_plot)
        ax.set_xlabel(r"$k_x$", fontsize=16)
        ax.set_ylabel(r"$\omega / N$", fontsize=16)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_title(f"$log (k_x, {kz_decimate[ikz_plot]}, \omega)$")

        # Compute index array corresponding to field.
        if field == "ap_fft":
            i_field = 0
        elif field == "am_fft":
            i_field = 1
        else:
            raise ValueError(f"field = {field} not known.")

        # Compute data
        temp_spectrum_mean = temp_spectrum_mean[i_field]

        data = np.log10(
            temp_spectrum_mean[imin_omegas_plot:, ikz_plot, ikxmin_plot:ikxmax_plot]
        )
        new = np.empty(
            [
                temp_spectrum_mean.shape[0] + 1,
                temp_spectrum_mean.shape[1],
                temp_spectrum_mean.shape[2],
            ]
        )
        new[:-1] = temp_spectrum_mean
        new[-1] = temp_spectrum_mean[-1]
        data = np.log10(new[imin_omegas_plot:, ikz_plot, ikxmin_plot:ikxmax_plot])

        # Maximum and minimum values
        vmin = np.min(data[abs(data) != np.inf])
        vmax = np.max(data)

        # Plot with contourf
        import matplotlib.cm as cm

        if func_plot == "contourf":
            cf = ax.contourf(
                kxs_grid, omegas_grid, data, cmap=cm.viridis, vmin=vmin, vmax=vmax
            )
        elif func_plot == "pcolormesh":
            cf = ax.pcolormesh(
                kxs_grid, omegas_grid, data, cmap=cm.viridis, vmin=vmin, vmax=vmax
            )
        else:
            print(f"Function plot not known.")

        # Plot colorbar
        m = plt.cm.ScalarMappable(cmap=cm.viridis)
        m.set_array(data)
        m.set_clim(vmin, vmax)
        plt.colorbar(m)

        # Plot dispersion relation
        ax.plot(
            kx_decimate[ikxmin_plot:ikxmax_plot],
            -np.log10(
                sim.params.N
                * (
                    kx_decimate[ikxmin_plot:ikxmax_plot]
                    / np.sqrt(
                        kx_decimate[ikxmin_plot:ikxmax_plot] ** 2
                        + kz_decimate[ikz_plot] ** 2
                    )
                )
            ),
            color="k",
        )

        # Plot forcing region
        ax.axvline(
            sim.oper.deltaky
            * sim.params.forcing.nkmin_forcing
            * np.sin(sim.params.forcing.tcrandom_anisotropic.angle),
            color="red",
        )

        ax.axvline(
            sim.oper.deltaky
            * sim.params.forcing.nkmax_forcing
            * np.sin(sim.params.forcing.tcrandom_anisotropic.angle),
            color="red",
        )

        # Plot text shear modes
        ax.text(
            sim.oper.deltakx // 2, -1, r"$E(k_x=0, k_z) = 0$", rotation=90, fontsize=14
        )

        # Tight layout of the figure
        fig.tight_layout()

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
                (self.params.time_stepping.t_end - self.time_start) / self.duration_file
            ),
        )

    def plot_frequency_spectra_individual_mode(self, path_file=None, mode=None):
        """[summary]
        
        Keyword Arguments:
            path_file {[type]} -- [description] (default: {None})
            mode {[type]} -- [description] (default: {None})
        
        Raises:
            ValueError -- [description]
        """

        # Define path_dir as posix
        path_dir = Path(self.path_dir)

        # If path_file does not exists..
        if not path_file:
            path_file = sorted((path_dir / "Spectrum").glob("spectrum*"))[-1]

        # If mode does is not given..
        if not mode:
            kx = kz = 16
            mode = (kx, kz)

        # Load frequency spectra
        print("path_file", path_file)
        with h5py.File(path_file, "r") as f:
            spectrum = f["spectrum"].value
            omegas = f["omegas"].value

        # Define index with spatial decimation
        idx_mode = np.argmin(abs(self.sim.oper.kx[:: self.spatial_decimate] - mode[0]))
        idz_mode = np.argmin(abs(self.sim.oper.ky[:: self.spatial_decimate] - mode[1]))
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

        # Linear frequency. Used for compensation of the plots..
        f_l = self.sim.params.N / (2 * np.pi)

        # Plot omega +
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel(r"$\omega / N$")
        ax1.set_ylabel(r"$F(\omega)$")
        ax1.set_title(
            r"$\omega_+ ; (k_x, k_z) = ({:.2f}, {:.2f})$".format(
                self.sim.oper.kx[:: self.spatial_decimate][idx_mode],
                self.sim.oper.ky[:: self.spatial_decimate][idz_mode],
            )
        )

        ax1.loglog(
            omegas[0 : len(omegas) // 2] / f_l,
            spectrum[0, 0 : len(omegas) // 2, idz_mode, idx_mode],
        )
        ax1.axvline(x=f_l / f_l, color="k", linestyle="--")

        # Plot omega -
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel(r"$\omega / N$")
        ax2.set_ylabel(r"$F(\omega)$")
        ax2.set_title(
            r"$\omega_- ; (k_x, k_z) = ({:.2f}, {:.2f})$".format(
                self.sim.oper.kx[:: self.spatial_decimate][idx_mode],
                self.sim.oper.ky[:: self.spatial_decimate][idz_mode],
            )
        )

        ax2.loglog(
            -1 * omegas[len(omegas) // 2 + 1 :] / f_l,
            spectrum[0, len(omegas) // 2 + 1 :, idz_mode, idx_mode],
        )
        ax2.axvline(x=f_l / f_l, color="k", linestyle="--")

    def _init_online_plot(self):
        if mpi.rank == 0:
            fig, axe = self.output.figure_axe(numfig=1_000_000)
            self.axe = axe
            axe.set_xlabel("$k_h$")
            axe.set_ylabel("$E(k_h)$")
            axe.set_title(
                "spectra, solver "
                + self.output.name_solver
                + ", nh = {0:5d}".format(self.params.oper.nx)
            )
