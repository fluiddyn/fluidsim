"""
SpatioTempSpectra (:mod:`fluidsim.solvers.ns2d.strat.output.spatio_temporal_spectra`)
=====================================================================================


Provides:

.. autoclass:: SpatioTempSpectra
   :members:
   :private-members:

"""

import sys
import time
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

from math import pi
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

        params.output.periods_save._set_attrib(tag, False)
        params.output._set_child(
            tag,
            attribs={
                "HAS_TO_PLOT_SAVED": False,
                "time_start": 1,
                "time_decimate": 2,
                "spatial_decimate": 2,
                "size_max_file": 0.1,
                "kx_max": None,
                "kz_max": None,
            },
        )

    def __init__(self, output):
        params = output.sim.params
        pspatiotemp_spectra = params.output.spatio_temporal_spectra
        super().__init__(
            output,
            period_save=params.output.periods_save.spatio_temporal_spectra,
            has_to_plot_saved=pspatiotemp_spectra.HAS_TO_PLOT_SAVED,
        )

        # Parameters
        self.time_start = pspatiotemp_spectra.time_start
        self.time_decimate = pspatiotemp_spectra.time_decimate
        self.spatial_decimate = pspatiotemp_spectra.spatial_decimate
        self.size_max_file = pspatiotemp_spectra.size_max_file

        self.kx_max = pspatiotemp_spectra.kx_max
        self.kz_max = pspatiotemp_spectra.kz_max

        # By default: kxmax_dealiasing or kymax_dealiasing
        if self.kx_max is None:
            self.kx_max = self.sim.oper.kxmax_dealiasing

        if self.kz_max is None:
            self.kz_max = self.sim.oper.kymax_dealiasing

        self.has_to_save = bool(
            params.output.periods_save.spatio_temporal_spectra
        )

        # Check: In restart... time_start == time last state.
        path_dir = Path(self.sim.output.path_run)
        path_spatio_temp_files = path_dir / "spatio_temporal"

        if sorted(path_spatio_temp_files.glob("spatio_temp*")):
            time_last_file = float(
                sorted(path_dir.glob("state_phys*"))[-1].stem.split(
                    "state_phys_t"
                )[1]
            )

            if round(self.time_start, 3) != time_last_file:
                self.time_start = time_last_file

        # Compute arrays
        nK0, nK1 = self.sim.oper.shapeK_seq

        # The maximum value kx_max should be the dealiased value
        if self.kx_max > self.sim.oper.kxmax_dealiasing:
            self.kx_max = self.sim.oper.kxmax_dealiasing

        # The maximum value kz_max should be the dealiased value
        if self.kz_max > self.sim.oper.kymax_dealiasing:
            self.kz_max = self.sim.oper.kymax_dealiasing

        # Modified values to take into account kx_max and kz_max
        self.nK0 = np.argmin(abs(self.sim.oper.kxE - self.kx_max))
        self.nK1 = np.argmin(abs(self.sim.oper.kyE - self.kz_max))

        nK0_dec = len(list(range(0, self.nK0, self.spatial_decimate)))
        nK1_dec = len(list(range(0, self.nK1, self.spatial_decimate)))

        # Compute size in bytes of one array
        # self.size_max_file is given in Mbytes. 1 Mbyte == 1024 ** 2 bytes
        nb_bytes = np.empty([nK0_dec, nK1_dec], dtype=complex).nbytes
        self.nb_arr_in_file = int(self.size_max_file * (1024**2) // nb_bytes)
        mpi.printby0("nb_arr_in_file", self.nb_arr_in_file)

        # Check: duration file <= duration simulation
        self.duration_file = (
            self.nb_arr_in_file
            * self.params.time_stepping.deltat0
            * self.time_decimate
        )
        if (
            self.duration_file > self.params.time_stepping.t_end
            and self.has_to_save
        ):
            raise ValueError(
                "The duration of the simulation is not enough to fill a file."
            )

        # Check: self.nb_arr_in_file should be > 0
        if self.nb_arr_in_file <= 0 and self.has_to_save:
            raise ValueError("The size of the file should be larger.")

        else:
            # Create array 4D (2 keys, times, n0, n1)

            self.spatio_temp_new = np.empty(
                [2, self.nb_arr_in_file, nK0_dec, nK1_dec], dtype=complex
            )

        #  Convert time_start to it_start.
        if not self.sim.time_stepping.it:
            self.it_start = int(
                self.time_start / self.params.time_stepping.deltat0
            )
        else:
            # If simulation starts from a specific time...
            self.it_start = self.sim.time_stepping.it + int(
                (self.time_start - round(self.sim.time_stepping.t, 3))
                / self.params.time_stepping.deltat0
            )

            if self.it_start < self.sim.time_stepping.it:
                self.it_start = self.sim.time_stepping.it

        # Create empty array with times
        self.times_arr = np.empty([self.nb_arr_in_file])
        self.its_arr = np.empty([self.nb_arr_in_file])

        if params.time_stepping.USE_CFL and self.has_to_save:
            raise ValueError(
                "To compute the spatio temporal: \n"
                "USE_CFL = FALSE and periods_save.spatio_temporal_spectra > 0"
            )

        # Create directory to save files
        dir_name = "spatio_temporal"
        self.path_dir = Path(self.sim.output.path_run) / dir_name

        if self.has_to_save and mpi.rank == 0:
            self.path_dir.mkdir(exist_ok=True)

        # Start loop in _online_save
        self.it_last_run = self.it_start
        self.nb_times_in_spatio_temp = 0

    def _init_files(self, arrays_1st_time=None):
        # we can not do anything when this function is called.
        pass

    def _write_to_file(self, spatio_temp_arr, times_arr, its_arr):
        """Writes a file with the spatio temporal data"""
        if mpi.rank == 0:
            # Name file
            it_end = self.sim.time_stepping.it
            name_file = f"spatio_temp_it{it_end}.h5"
            path_file = self.path_dir / name_file

            # Dictionary arrays
            dict_arr = {
                "its_arr": its_arr,
                "times_arr": times_arr,
                "spatio_temp": spatio_temp_arr,
            }

            # Write dictionary to file
            with h5py.File(path_file, "w") as file:
                for k, v in list(dict_arr.items()):
                    file.create_dataset(k, data=v)

    def _online_save(self):
        """Computes and saves the values at one time."""
        oper = self.sim.oper
        if not self.has_to_save:
            return

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

            # Cut off kx_max and kz_max
            if mpi.rank == 0:
                field_ap_seq = field_ap_seq[0 : self.nK0, 0 : self.nK1]
                field_am_seq = field_am_seq[0 : self.nK0, 0 : self.nK1]

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
            if not self.sim.time_stepping.it:
                self.times_arr[self.nb_times_in_spatio_temp] = (
                    itsim * self.sim.params.time_stepping.deltat0
                )
            else:
                self.times_arr[
                    self.nb_times_in_spatio_temp
                ] = self.sim.time_stepping.t

            self.its_arr[self.nb_times_in_spatio_temp] = self.sim.time_stepping.it

            # Check if self.spatio_temp is filled. If yes, writes to a file.
            if (
                self.nb_times_in_spatio_temp == self.nb_arr_in_file - 1
                or self.sim.time_stepping.is_simul_completed() == True
            ):
                if mpi.rank == 0:
                    print("Saving spatio_temporal data...")

                # If the file is not completed AND the simulation is finished, it saves the file too.
                # Check if the last time is good.
                if self.sim.time_stepping.is_simul_completed() and mpi.rank == 0:
                    self.spatio_temp_new = self.spatio_temp_new[
                        :, : self.nb_times_in_spatio_temp + 1, :, :
                    ]
                    self.times_arr = self.times_arr[
                        : self.nb_times_in_spatio_temp + 1
                    ]
                    self.its_arr = self.its_arr[
                        : self.nb_times_in_spatio_temp + 1
                    ]

                self._write_to_file(
                    self.spatio_temp_new, self.times_arr, self.its_arr
                )

                self.nb_times_in_spatio_temp = 0
            else:
                self.nb_times_in_spatio_temp += 1

    def compute_frequency_spectra(
        self, tmin=None, tmax=None, time_windows=None, overlap=0.5
    ):
        """Computes the temporal frequency spectrum.

        Keyword Arguments:
            tmin {float} -- Minimum time to perform temporal FT. (default: {None})
            tmax {float} -- Maximum time to perform temporal FT. (default: {None})
            time_windows {float} -- Time windows to compute temporal FT. (default: {None})
            overlap {float} -- Overlap between windows. From 0 to 1. (default: {0.5})
        """

        # Get concatenated array of times and spatio temporal
        times, spatio_temp = self._get_concatenate_data(tmin, tmax)

        # Compute windows_size
        if time_windows is None:
            nb_periods_N = 10
            windows_size = int(
                2
                * pi
                / self.sim.params.N
                * nb_periods_N
                * (1 / self.sim.params.time_stepping.deltat0 / self.time_decimate)
            )
        else:
            windows_size = int(
                time_windows
                * (1 / self.sim.params.time_stepping.deltat0 / self.time_decimate)
            )

        # # Compute time_windows
        # if not nb_periods_N:
        # nb_periods_N = 10

        # windows_size = int(
        # 2
        # * pi
        # / self.sim.params.N
        # * nb_periods_N
        # * (1 / self.sim.params.time_stepping.deltat0 / self.time_decimate)
        # )

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
        freq_sampling = 1.0 / (
            self.time_decimate * self.params.time_stepping.deltat0
        )

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
        path_dir.mkdir(exist_ok=True)

        path_file = path_dir / name_file

        # Save file into directory spatio_temporal/Spectrum
        with h5py.File(path_file, "w") as file:
            for k, v in list(dict_arr.items()):
                file.create_dataset(k, data=v)

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
        list_files = sorted(self.path_dir.glob("spatio_temp_it*"))

        # Sort files by number of iteration.
        list_its = []

        for path_file in list_files:
            list_its.append(int(path_file.name.split("_it")[1].split(".h5")[0]))

        list_its = sorted(list_its)
        print(list_its)
        list_files_new = []
        for it in list_its:
            list_files_new.append(
                self.path_dir / ("spatio_temp_it" + str(it) + ".h5")
            )

        list_files = list_files_new

        # List of tuples (ifile, time)
        ifile_time = []

        for index, path_file in enumerate(list_files):
            with h5py.File(path_file, "r") as file:
                times = file["times_arr"][...]
            for time_value in times:
                ifile_time.append((index, time_value))

        ifile_min = None
        ifile_max = None

        for ii, value in enumerate(ifile_time):
            if ifile_min is None and tmin == round(value[1], 0):
                ifile_min = value[0]
            if ifile_max is None and tmax == round(value[1], 0):
                ifile_max = value[0]

        if ifile_min is None:
            ifile_min = 0

        if ifile_max is None:
            ifile_max = len(list_files)

        # Define concatenate arrays
        spatio_temp_conc = None
        times_conc = None

        # Load all data
        for index, path_file in enumerate(list_files[ifile_min:ifile_max]):
            with h5py.File(path_file, "r") as file:
                spatio_temp = file["spatio_temp"][...]
                times = file["times_arr"][...]

            # Print concatenating info..
            print(
                f"Concatenating file = {index + 1}/{len(list_files[ifile_min:ifile_max])}..",
                end="\r",
            )

            # Concatenate arrays
            if isinstance(spatio_temp_conc, np.ndarray):
                spatio_temp_conc = np.concatenate(
                    (spatio_temp_conc, spatio_temp), axis=1
                )
            if isinstance(times_conc, np.ndarray):
                times_conc = np.concatenate((times_conc, times), axis=0)
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

    def _add_inset_plot(self, fig, kxmax_inset=32, kzmax_inset=32):
        """
        Adds an inset plot to the object figure.
        It is the spectral space with the forcing region.
        """
        # Checks if self.sim.forcing.forcing_maker object exists.
        if self.sim.forcing.forcing_maker is None:
            ax_inset = None
        else:
            # Define position of the inset plot
            # There are percentages of the original axes.
            left, bottom, width, height = [0.56, 0.68, 0.22, 0.22]
            ax_inset = fig.add_axes([left, bottom, width, height])

            # Add labels
            ax_inset.set_xlabel("$k_x$", labelpad=0.05)
            ax_inset.set_ylabel("$k_z$", labelpad=0.05)

            # Set ticks each axis
            xticks = np.arange(0, kxmax_inset, self.sim.oper.deltakx)
            zticks = np.arange(0, kzmax_inset, self.sim.oper.deltaky)
            ax_inset.set_xticks(xticks[:: int(self.sim.oper.deltaky)])
            ax_inset.set_yticks(zticks)

            # Set axis limit & grid
            ax_inset.set_xlim(0, kxmax_inset)
            ax_inset.set_ylim(0, kzmax_inset)
            ax_inset.grid(linestyle="--", alpha=0.4)

            indices_forcing = np.argwhere(
                self.sim.forcing.forcing_maker.COND_NO_F == False
            )

            for i, index in enumerate(indices_forcing):
                ax_inset.plot(
                    self.sim.oper.KX[0, index[1]],
                    self.sim.oper.KY[index[0], 0],
                    color="red",
                    marker="o",
                    markersize=2,
                    label="Forced mode" if i == 0 else "",
                )

        return ax_inset

    def plot_kx_omega_cross_section(
        self,
        path_file=None,
        field="ap_fft",
        ikz_plot=None,
        kxmax_plot=None,
        func_plot="pcolormesh",
        INSET_PLOT=True,
    ):
        pspatio = self.params.output.spatio_temporal_spectra
        # Define path dir as posix path
        path_dir = self.path_dir

        # If path_file does not exist.
        if not path_file:
            path_file = sorted((path_dir / "Spectrum").glob("spectrum*"))[-1]

        print(f"File = {path_file}")
        # Load data
        with h5py.File(path_file, "r") as file:
            temp_spectrum_mean = file["spectrum"][...]
            omegas = file["omegas"][...]

        # Load simulation object
        sim = load_state_phys_file(path_dir.parent, merge_missing_params=True)

        # Cut_off indexes
        # ikx_top = np.argmin(abs(sim.oper.kx - sim.oper.kxmax_dealiasing))
        # ikz_top = np.argmin(abs(sim.oper.ky - sim.oper.kymax_dealiasing))

        kx_max = pspatio.kx_max
        kz_max = pspatio.kx_max

        if kx_max is None:
            kx_max = sim.oper.kxmax_dealiasing

        if kz_max is None:
            kz_max = sim.oper.kymax_dealiasing

        print("kx_max", kx_max)

        ikx_top = np.argmin(abs(sim.oper.kx - kx_max))
        ikz_top = np.argmin(abs(sim.oper.ky - kz_max))

        # Cut_off kx_max and ky_max
        kx_cut_off = sim.oper.kx[0:ikx_top]
        kz_cut_off = sim.oper.ky[0:ikz_top]

        # Compute kx_decimate and ky_decimate
        kx_decimate = kx_cut_off[
            :: sim.params.output.spatio_temporal_spectra.spatial_decimate
        ]
        kz_decimate = kz_cut_off[
            :: sim.params.output.spatio_temporal_spectra.spatial_decimate
        ]

        # Omegas
        omegas *= 2 * np.pi
        omegas *= 1 / (sim.params.N)

        #### PLOT OMEGA - KX

        # Define maximum and minimum values to plot.
        kxmin_plot = 0
        if kxmax_plot is None:
            kxmax_plot = sim.oper.kx[ikx_top]
        else:
            ikxmax_plot = np.argmin(abs(sim.oper.kx - kxmax_plot))
            kxmax_plot = sim.oper.kx[ikxmax_plot]

        ikxmin_plot = np.argmin(abs(kx_decimate - kxmin_plot))
        ikxmax_plot = np.argmin(abs(kx_decimate - kxmax_plot)) + 1

        omega_max_plot = 0
        omega_min_plot = -3

        omegas = np.append(omegas, 0)
        imin_omegas_plot = np.argmin(abs(omegas - omega_min_plot))

        kxs_grid, omegas_grid = np.meshgrid(
            kx_decimate[ikxmin_plot:ikxmax_plot], omegas[imin_omegas_plot:]
        )

        if ikz_plot is None:
            ikz_plot = 1

        # Parameters figure
        fig, ax = plt.subplots()
        ax.set_xlim(left=kxmin_plot, right=kxmax_plot)
        ax.set_ylim(top=omega_max_plot, bottom=omega_min_plot)
        ax.set_xlabel(r"$k_x$", fontsize=16)
        ax.set_ylabel(r"$\omega / N$", fontsize=16)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_title(rf"$log_{{10}} E(k_x, {kz_decimate[ikz_plot]}, \omega)$")

        # Compute index array corresponding to field.
        if field == "ap_fft":
            i_field = 0
        elif field == "am_fft":
            i_field = 1
        else:
            raise ValueError(f"field = {field} not known.")

        # Compute data
        temp_spectrum_mean = temp_spectrum_mean[i_field]

        # todo: fix the warnings that we get while testing
        # data = np.log10(
        # temp_spectrum_mean[imin_omegas_plot:, ikz_plot, ikxmin_plot:ikxmax_plot]
        # )

        data = np.log10(
            temp_spectrum_mean[
                imin_omegas_plot:, ikxmin_plot:ikxmax_plot, ikz_plot
            ]
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
        data = np.log10(new[imin_omegas_plot:, ikxmin_plot:ikxmax_plot, ikz_plot])
        # data = np.log10(new[imin_omegas_plot:, ikz_plot, ikxmin_plot:ikxmax_plot])

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
            # To fit in pcolormesh
            kxs_grid = kxs_grid - (sim.oper.deltakx / 2)
            omegas_grid = omegas_grid - (omegas[1] / 2)

            cf = ax.pcolormesh(
                kxs_grid,
                omegas_grid,
                data,
                shading="nearest",
                cmap=cm.viridis,
                vmin=vmin,
                vmax=vmax,
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
            kxs_grid[0],
            -(
                kxs_grid[0]
                / np.sqrt(kxs_grid[0] ** 2 + kz_decimate[ikz_plot] ** 2)
            ),
            color="k",
        )

        # Check if angle string or float instance.
        if (
            isinstance(sim.params.forcing.tcrandom_anisotropic.angle, str)
            and "°" in sim.params.forcing.tcrandom_anisotropic.angle
        ):
            angle = math.radians(
                float(sim.params.forcing.tcrandom_anisotropic.angle.split("°")[0])
            )
        else:
            angle = sim.params.forcing.tcrandom_anisotropic.angle

        # Plot forcing region
        ax.axvline(
            sim.oper.deltaky * sim.params.forcing.nkmin_forcing * np.sin(angle),
            color="red",
        )

        ax.axvline(
            sim.oper.deltaky * sim.params.forcing.nkmax_forcing * np.sin(angle),
            color="red",
        )

        # Plot text shear modes
        ax.text(
            sim.oper.deltakx // 2,
            -1,
            r"$E(k_x=0, k_z) = 0$",
            rotation=90,
            fontsize=14,
            color="white",
        )

        # Inset plot
        if INSET_PLOT:
            ax_inset = self._add_inset_plot(fig)
            if ax_inset is not None:
                ax_inset.axhline(y=self.sim.oper.ky[ikz_plot], color="k")

        # Tight layut of the figure
        fig.tight_layout()

    def plot_kz_omega_cross_section(
        self,
        path_file=None,
        field="ap_fft",
        ikx_plot=None,
        func_plot="pcolormesh",
        INSET_PLOT=True,
    ):
        # if ax_inset is not None:
        #  D    efine path dir as po   six path
        path_dir = self.path_dir
        # If path_file does not exist.
        if not path_file:
            path_file = sorted((path_dir / "Spectrum").glob("spectrum*"))[-1]

        # Print path_file
        print(f"Path = {path_file}")

        # Load data
        with h5py.File(path_file, "r") as file:
            temp_spectrum_mean = file["spectrum"][...]
            omegas = file["omegas"][...]

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

        #### PLOT OMEGA - KZ
        kzmin_plot = 0
        kzmax_plot = 80

        if kzmax_plot > sim.oper.kymax_dealiasing:
            kzmax_plot = sim.oper.kymax_dealiasing

        ikzmin_plot = np.argmin(abs(kz_decimate - kzmin_plot))
        ikzmax_plot = np.argmin(abs(kz_decimate - kzmax_plot))

        omega_max_plot = 0
        omega_min_plot = -3

        omegas = np.append(omegas, 0)
        imin_omegas_plot = np.argmin(abs(omegas - omega_min_plot)) - 1

        kzs_grid, omegas_grid = np.meshgrid(
            kz_decimate[ikzmin_plot:ikzmax_plot], omegas[imin_omegas_plot:]
        )

        if not ikx_plot:
            ikx_plot = 1

        # Parameters figure
        fig, ax = plt.subplots()
        ax.set_xlim(left=kzmin_plot, right=kzmax_plot - sim.oper.deltaky / 2)
        ax.set_ylim(top=omega_max_plot, bottom=omega_min_plot)
        ax.set_xlabel(r"$k_z$", fontsize=16)
        ax.set_ylabel(r"$\omega / N$", fontsize=16)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_title(rf"$log_{{10}} E({kx_decimate[ikx_plot]}, k_z, \omega)$")

        # Compute index array corresponding to field.
        if field == "ap_fft":
            i_field = 0
        elif field == "am_fft":
            i_field = 1
        else:
            raise ValueError(f"field = {field} not known.")

        # Compute data
        temp_spectrum_mean = temp_spectrum_mean[i_field]

        # todo: fix the warnings that we get while testing
        data = np.log10(
            temp_spectrum_mean[
                imin_omegas_plot:, ikx_plot, ikzmin_plot:ikzmax_plot
            ]
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
        data = np.log10(new[imin_omegas_plot:, ikx_plot, ikzmin_plot:ikzmax_plot])

        # Maximum and minimum values
        vmin = np.min(data[abs(data) != np.inf])
        vmin = -3
        vmax = np.max(data)
        vmax = 0

        # Plot with contourf
        import matplotlib.cm as cm

        if func_plot == "contourf":
            cf = ax.contourf(
                kzs_grid, omegas_grid, data, cmap=cm.viridis, vmin=vmin, vmax=vmax
            )
        elif func_plot == "pcolormesh":
            kzs_grid = kzs_grid - (sim.oper.deltaky / 2)
            omegas_grid = omegas_grid - (omegas[1] / 2)
            cf = ax.pcolormesh(
                kzs_grid,
                omegas_grid,
                data,
                shading="nearest",
                cmap=cm.viridis,
                vmin=vmin,
                vmax=vmax,
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
            kzs_grid[0][ikzmin_plot:ikzmax_plot],
            -(
                kx_decimate[ikx_plot]
                / np.sqrt(
                    kx_decimate[ikx_plot] ** 2
                    + kzs_grid[0][ikzmin_plot:ikzmax_plot] ** 2
                )
            ),
            color="k",
        )

        # Check if angle string or float instance.
        if (
            isinstance(sim.params.forcing.tcrandom_anisotropic.angle, str)
            and "°" in sim.params.forcing.tcrandom_anisotropic.angle
        ):
            angle = math.radians(
                float(sim.params.forcing.tcrandom_anisotropic.angle.split("°")[0])
            )
        else:
            angle = sim.params.forcing.tcrandom_anisotropic.angle

        # Plot forcing region
        ax.axvline(
            sim.oper.deltaky * sim.params.forcing.nkmin_forcing * np.sin(angle),
            color="red",
        )

        ax.axvline(
            sim.oper.deltaky * sim.params.forcing.nkmax_forcing * np.sin(angle),
            color="red",
        )

        # Inset plot
        if INSET_PLOT:
            ax_inset = self._add_inset_plot(fig)
            if ax_inset is not None:
                ax_inset.axvline(x=self.sim.oper.kx[ikx_plot], color="k")

        # Tight layout of the figure
        fig.tight_layout()

    def print_info_frequency_spectra(self):
        """Print information frequency spectra."""

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

    def plot_frequency_spectra_individual_mode(
        self, path_file=None, mode=None, INSET_PLOT=True
    ):
        """Plots the frequency spectra of an individual Fourier mode.

        Keyword Arguments:
            path_file {string} -- Path of the file.
            mode {tuple} -- Mode (kx, kz) to plot.

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
        print(f"path_file = {path_file}")
        with h5py.File(path_file, "r") as file:
            spectrum = file["spectrum"][...]
            omegas = file["omegas"][...]

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
        print(f"ikx_mode = {idx_mode} ; idz_mode = {idz_mode}")

        # Compute omega dispersion relation mode
        kx_mode = self.sim.oper.kx[:: self.spatial_decimate][idx_mode]
        kz_mode = self.sim.oper.ky[:: self.spatial_decimate][idz_mode]

        # Linear frequency. Used for compensation of the plots..
        N = self.sim.params.N
        f_l = N / (2 * np.pi)
        iw = (N / (2 * np.pi)) * kx_mode / (np.sqrt(kx_mode**2 + kz_mode**2))

        # Plot omega +
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel(r"$\omega / N$")
        ax1.set_ylabel(r"$F(\omega)$")
        ax1.set_ylim(bottom=1e-7)
        ax1.set_title(
            r"$\omega_+ ; (k_x, k_z) = ({:.2f}, {:.2f})$".format(
                self.sim.oper.kx[:: self.spatial_decimate][idx_mode],
                self.sim.oper.ky[:: self.spatial_decimate][idz_mode],
            )
        )

        ax1.loglog(
            omegas[0 : len(omegas) // 2] / f_l,
            spectrum[0, 0 : len(omegas) // 2, idx_mode, idz_mode],
            "k",
        )
        ax1.axvline(x=f_l / f_l, color="k", linestyle="--")
        ax1.axvline(x=iw / f_l, color="r", linestyle="--")

        # Plot omega -
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel(r"$\omega / N$")
        ax2.set_ylabel(r"$F(\omega)$")
        ax2.set_ylim(bottom=1e-7)
        ax2.set_title(
            r"$\omega_- ; (k_x, k_z) = ({:.2f}, {:.2f})$".format(
                self.sim.oper.kx[:: self.spatial_decimate][idx_mode],
                self.sim.oper.ky[:: self.spatial_decimate][idz_mode],
            )
        )

        ax2.loglog(
            -1 * omegas[len(omegas) // 2 + 1 :] / f_l,
            spectrum[0, len(omegas) // 2 + 1 :, idx_mode, idz_mode],
            "k",
        )
        ax2.axvline(x=f_l / f_l, color="k", linestyle="--")
        ax2.axvline(x=iw / f_l, color="r", linestyle="--")

        # Inset plot
        if INSET_PLOT:
            ax_inset = self._add_inset_plot(fig2)
            if ax_inset is not None:
                ax_inset.plot([kx_mode], [kz_mode], color="b", marker="o")

    def _init_online_plot(self):
        if mpi.rank == 0:
            fig, ax = self.output.figure_axe(numfig=1_000_000)
            self.ax = ax
            ax.set_xlabel("$k_h$")
            ax.set_ylabel("$E(k_h)$")
            ax.set_title("spectra\n" + self.output.summary_simul)
