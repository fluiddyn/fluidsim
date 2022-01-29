"""
FrequencySpectra (:mod:`fluidsim.solvers.ns2d.strat.output.frequency_spectra`)
==============================================================================


Provides:

.. autoclass:: FrequencySpectra
   :members:
   :private-members:

"""

import sys
import time
import h5py
import numpy as np

from scipy import signal
from pathlib import Path

from fluiddyn.util import mpi
from fluidsim.base.output.base import SpecificOutput


class FrequencySpectra(SpecificOutput):
    """
    Computes the frequency spectra.
    """

    _tag = "frequency_spectra"
    _name_file = _tag + ".h5"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "frequency_spectra"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(
            tag,
            attribs={
                "HAS_TO_PLOT_SAVED": False,
                "time_start": 1,
                "time_decimate": 1,
                "spatial_decimate": 2,
                "size_max_file": 1e-2,
            },
        )

    def __init__(self, output):
        params = output.sim.params
        pfreq_spectra = params.output.frequency_spectra
        super().__init__(
            output,
            period_save=params.output.periods_save.frequency_spectra,
            has_to_plot_saved=pfreq_spectra.HAS_TO_PLOT_SAVED,
        )

        # Parameters
        self.time_start = pfreq_spectra.time_start
        self.time_decimate = pfreq_spectra.time_decimate
        self.spatial_decimate = pfreq_spectra.spatial_decimate
        self.size_max_file = pfreq_spectra.size_max_file
        self.periods_save = params.output.periods_save.frequency_spectra

        self.has_to_save = bool(params.output.periods_save.frequency_spectra)

        # Check: In restart... time_start == time last state.
        path_run = Path(self.sim.output.path_run)
        path_spatio_temp_files = path_run / "frequency_spectra"

        if sorted(path_spatio_temp_files.glob("temp_array_it*")):
            name = sorted(path_run.glob("state_phys*"))[-1].stem
            if "_it" in name:
                time_last_file = float(
                    name.split("state_phys_t")[-1].split("_it")[0]
                )
            else:
                time_last_file = float(name.split("state_phys_t")[1])

            if round(self.time_start, 3) != time_last_file:
                self.time_start = time_last_file

        # Number points each direction
        n0 = len(list(range(0, params.oper.ny, self.spatial_decimate)))
        n1 = len(list(range(0, params.oper.nx, self.spatial_decimate)))

        # Compute number array in file
        nb_bytes = np.empty([n0, n1], dtype=float).nbytes
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
            and self.periods_save > 0
        ):
            raise ValueError(
                "The duration of the simulation is not enough to fill a file."
            )

        # Check: self.nb_arr_in_file should be > 0
        if self.nb_arr_in_file <= 0 and self.periods_save > 0:
            raise ValueError("The size of the file should be larger.")

        else:
            self.temp_array = np.empty([self.nb_arr_in_file, n0, n1], dtype=float)

            # Array 4D (2 keys, times, n0, n1)
            self.temp_array_new = np.empty(
                [2, self.nb_arr_in_file, n0, n1], dtype=float
            )

        # Convert time_start to it_start.
        if not self.sim.time_stepping.it:
            self.it_start = int(
                self.time_start / self.params.time_stepping.deltat0
            )
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
            and params.output.periods_save.frequency_spectra > 0
        ):
            raise ValueError(
                "To compute the frequency spectra: \n"
                + "USE_CFL = FALSE and periods_save.frequency_spectra > 0"
            )

        # Create directory to save files
        if mpi.rank == 0:
            dir_name = self._tag
            self.path_dir = Path(self.sim.output.path_run) / dir_name

        if self.has_to_save and mpi.rank == 0:
            self.path_dir.mkdir(exist_ok=True)

        # Start loop in _online_save
        self.it_last_run = self.it_start
        self.nb_times_in_temp_array = 0

    def _init_files(self, arrays_1st_time=None):
        # we can not do anything when this function is called.
        pass

    def _write_to_file(self, temp_arr, times_arr):
        """Writes a file with the temporal data"""
        if mpi.rank == 0:
            # Name file
            it_start = int(times_arr[0] / self.sim.params.time_stepping.deltat0)
            name_file = f"temp_array_it{it_start}.h5"
            path_file = self.path_dir / name_file

            # Dictionary arrays
            dict_arr = {
                "it_start": it_start,
                "times_arr": times_arr,
                "temp_arr": temp_arr,
            }

            # Write dictionary to file
            with h5py.File(path_file, "w") as file:
                for k, v in list(dict_arr.items()):
                    file.create_dataset(k, data=v)

    def _online_save(self):
        """Computes and saves the values at one time."""
        oper = self.sim.oper
        if self.periods_save == 0:
            pass
        else:
            itsim = self.sim.time_stepping.it
            if itsim - self.it_last_run >= self.time_decimate:
                self.it_last_run = itsim

                # Compute linear mode in physical space
                field_ap = self.sim.state.compute("ap")
                field_am = self.sim.state.compute("am")

                if mpi.nb_proc > 1:
                    # Create big array
                    if mpi.rank == 0:
                        field_ap_seq = oper.create_arrayX(shape="seq")
                        field_am_seq = oper.create_arrayX(shape="seq")
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

                    self.temp_array_new[
                        0, self.nb_times_in_temp_array, :, :
                    ] = field_ap_decimate
                    self.temp_array_new[
                        1, self.nb_times_in_temp_array, :, :
                    ] = field_am_decimate

                # Save the time to self.times_arr
                self.times_arr[self.nb_times_in_temp_array] = (
                    itsim * self.sim.params.time_stepping.deltat0
                )

                # Check if self.temp_array_new is filled. If yes, writes to a file.
                if self.nb_times_in_temp_array == self.nb_arr_in_file - 1:
                    if mpi.rank == 0:
                        print("Saving temporal data...")
                    self._write_to_file(self.temp_array_new, self.times_arr)

                    self.nb_times_in_temp_array = 0
                else:
                    self.nb_times_in_temp_array += 1

    def _concatenate_data(self, tmin, tmax):
        """
        Concatenates temporal data.

        Argument
            tmin {float} -- Minimum time to perform temporal FT.
            tmax {float} -- Maximum time to perform temporal FT.

        Raises:
            ValueError -- itmin == itmax

        Returns:
            times_conc {array} -- Concatenated time array
            temp_arr_conc {array} -- concatenated temporal data array
        """

        path_files = sorted(self.path_dir.glob("temp_array_it*"))

        # Sort files by number of iteration.
        list_its = []

        for path_file in path_files:
            list_its.append(int(path_file.name.split("_it")[1].split(".h5")[0]))

        list_its = sorted(list_its)
        print(list_its)
        list_files_new = []
        for it in list_its:
            list_files_new.append(
                self.path_dir / ("temp_array_it" + str(it) + ".h5")
            )

        list_files = list_files_new
        for lf in list_files:
            print(lf.name)

        # Create concatenated arrays
        temp_arr_conc = None
        times_conc = None

        for path_file in list_files:
            with h5py.File(path_file, "r") as file:
                temp_arr = file["temp_arr"][...]
                times = file["times_arr"][...]

            # Concatenate arrays
            if isinstance(temp_arr_conc, np.ndarray):
                temp_arr_conc = np.concatenate((temp_arr_conc, temp_arr), axis=1)
            if isinstance(times_conc, np.ndarray):
                times_conc = np.concatenate((times_conc, times), axis=0)
            else:
                temp_arr_conc = temp_arr
                times_conc = times

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

        return times_conc[itmin:itmax], temp_arr_conc[:, itmin:itmax, :, :]

    def compute_frequency_spectra(self):
        """
        Computes and saves the frequency spectra.
        """
        # Define list of path files
        list_files = sorted(self.path_dir.glob("temp_array_iself.t*"))

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
            with h5py.File(file_path, "r") as file:
                temp_array = file["temp_arr"][...]
                times = file["times_arr"][...]

            # Compute the temporal spectrum of a 3D array
            omegas, freq_spectrum = signal.periodogram(
                temp_array,
                fs=freq_sampling,
                window="hann",
                nfft=temp_array.shape[1],
                detrend="constant",
                return_onesided=False,
                scaling="spectrum",
                axis=1,
            )

            # Save array omegas and spectrum to file
            dict_arr = {"omegas": omegas, "freq_spectrum": freq_spectrum}

            with h5py.File(file_path, "r+") as file:
                for k, v in list(dict_arr.items()):
                    file.create_dataset(k, data=v)

            # Flush buffer and sleep time
            sys.stdout.flush()
            time.sleep(0.2)

    def _init_online_plot(self):
        if mpi.rank == 0:
            fig, axe = self.output.figure_axe(numfig=1_000_000)
            self.axe = axe
            axe.set_xlabel("$k_h$")
            axe.set_ylabel("$E(k_h)$")
            axe.set_title("spectra\n" + self.output.summary_simul)
