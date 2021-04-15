"""
Spatiotemporal Spectra (:mod:`fluidsim.base.output.spatiotemporal_spectra`)
===========================================================================

Provides:

.. autoclass:: SpatioTemporalSpectra
   :members:
   :private-members:

"""

from pathlib import Path
from logging import warn
from math import pi

import numpy as np
from scipy import signal
import h5py
from rich.progress import Progress

from fluiddyn.util import mpi
from fluidsim.base.output.base import SpecificOutput

from transonic import boost, Array, Type

Uf32f64 = Type(np.float32, np.float64)
A = Array[Uf32f64, "1d"]


@boost
def find_index_first_geq(arr: A, value: Uf32f64):
    """find the first index such that `arr[index] >= value`"""
    for i, v in enumerate(arr):
        if v >= value:
            return i
    raise ValueError("No index such that `arr[index] >= value`")


@boost
def find_index_first_g(arr: A, value: Uf32f64):
    """find the first index such that `arr[index] > value`"""
    for i, v in enumerate(arr):
        if v > value:
            return i
    raise ValueError("No index such that `arr[index] > value`")


@boost
def find_index_first_l(arr: A, value: Uf32f64):
    """find the first index such that `arr[index] < value`"""
    for i, v in enumerate(arr):
        if v < value:
            return i
    raise ValueError("No index such that `arr[index] < value`")


def filter_tmins_paths(tmin, tmins, paths):
    if tmins.size == 1:
        return tmins, paths
    delta_tmin = np.diff(tmins).min()
    start = find_index_first_l(tmin - tmins, delta_tmin)
    return tmins[start:], paths[start:]


@boost
def get_arange_minmax(times: A, tmin: Uf32f64, tmax: Uf32f64):
    """get a range of index for which `tmin <= times[i] <= tmax`

    This assumes that `times` is sorted.

    """

    if tmin <= times[0]:
        start = 0
    else:
        start = find_index_first_geq(times, tmin)

    if tmax >= times[-1]:
        stop = len(times)
    else:
        stop = find_index_first_g(times, tmax)

    return np.arange(start, stop)


class SpatioTemporalSpectra(SpecificOutput):
    """
    Computes the spatiotemporal spectra.
    """

    _tag = "spatiotemporal_spectra"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "spatiotemporal_spectra"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(
            tag,
            attribs={
                "HAS_TO_PLOT_SAVED": False,
                "probes_region": None,
                "file_max_size": 10.0,  # MB
                "SAVE_AS_COMPLEX64": True,
            },
        )

        params.output.temporal_spectra._set_doc(
            """
            probes_region: int tuple (default:None)

                Boundaries of the region to record in the spectral domain.

                probes_region = (ikxmax, ikymax, ikzmax), in nondimensional units (mode indices).

                The resulting ranges for each wavevectors are :

                    ikx in [0 , ikxmax]

                    iky in [-ikymax+1 , ikymax]

                    ikz in [-ikzmax+1 , ikzmax]

                If None, set all ikmax = 4.

            file_max_size: float (default: 10.0)

                Maximum size of one time series file, in megabytes.

            SAVE_AS_COMPLEX64: bool (default: True)

                If set to False, probes data is saved as complex128.

                Warning : saving as complex128 reduces digital noise at high frequency, but doubles the size of the output!

            """
        )

    def __init__(self, output):
        params = output.sim.params
        try:
            params_st_spec = params.output.spatiotemporal_spectra
        except AttributeError:
            warn(
                "Cannot initialize spatiotemporal spectra output because "
                "`params` does not contain parameters for this class."
            )
            return

        super().__init__(
            output,
            period_save=params.output.periods_save.spatiotemporal_spectra,
            has_to_plot_saved=params_st_spec.HAS_TO_PLOT_SAVED,
        )

        oper = self.sim.oper

        # Parameters
        self.period_save = params.output.periods_save.spatiotemporal_spectra

        self.path_dir = Path(self.sim.output.path_run) / "spatiotemporal"
        self.keys_fields = self.sim.info_solver.classes.State.keys_state_phys

        if not output._has_to_save:
            self.period_save = 0.0
        if self.period_save == 0.0:
            return

        if params_st_spec.probes_region is not None:
            ikxmax, ikymax, ikzmax = params_st_spec.probes_region
            ikxmax = min(ikxmax, params.oper.nx // 2)
            ikymax = min(ikymax, params.oper.ny // 2)
            ikzmax = min(ikzmax, params.oper.nz // 2)
            self.probes_region = ikxmax, ikymax, ikzmax
        else:
            ikxmax = ikymax = ikzmax = 4
            ikxmax = min(ikxmax, params.oper.nx // 2)
            ikymax = min(ikymax, params.oper.ny // 2)
            ikzmax = min(ikzmax, params.oper.nz // 2)
            self.probes_region = ikxmax, ikymax, ikzmax

        self.file_max_size = params_st_spec.file_max_size
        self.SAVE_AS_COMPLEX64 = params_st_spec.SAVE_AS_COMPLEX64

        # region must be int tuple
        ikxmax = int(ikxmax)
        ikymax = int(ikymax)
        ikzmax = int(ikzmax)

        # dimensions order in Fourier space
        self.dims_order = np.array(oper.oper_fft.get_dimX_K())

        # data directory
        if mpi.rank == 0:
            self.path_dir.mkdir(exist_ok=True)
        if mpi.nb_proc > 1:
            mpi.comm.barrier()

        # data type and size
        if self.SAVE_AS_COMPLEX64:
            self.datatype = np.complex64
            size_1_number = 8e-6
        else:
            self.datatype = np.complex128
            size_1_number = 16e-6

        # check for existing files
        paths = sorted(self.path_dir.glob("rank*.h5"))
        if paths:
            # check values in files
            with h5py.File(paths[0], "r") as file:
                if file.attrs["nb_proc"] != mpi.nb_proc:
                    raise ValueError("process number is different from files")
                if (file.attrs["dims_order"] != self.dims_order).any():
                    raise ValueError("dimensions order is different from files")
                if (file.attrs["probes_region"] != self.probes_region).any():
                    raise ValueError("probes region is different from files")
            # init from files
            INIT_FROM_PARAMS = False
            paths = [p for p in paths if p.name.startswith(f"rank{mpi.rank:05}")]
            if paths:
                self.path_file = paths[-1]
                with h5py.File(self.path_file, "r") as file:
                    self.index_file = file.attrs["index_file"]
                    self.probes_k0adim_loc = file["probes_k0adim_loc"][:]
                    self.probes_k1adim_loc = file["probes_k1adim_loc"][:]
                    self.probes_k2adim_loc = file["probes_k2adim_loc"][:]
                    self.probes_ik0_loc = file["probes_ik0_loc"][:]
                    self.probes_ik1_loc = file["probes_ik1_loc"][:]
                    self.probes_ik2_loc = file["probes_ik2_loc"][:]
                    self.probes_nb_loc = self.probes_ik0_loc.size
                    self.number_times_in_file = file["times"].size
                    self.t_last_save = file["times"][-1]
            else:
                # no probes in proc
                self.path_file = None
                self.index_file = 0
                self.number_times_in_file = 0
                self.probes_nb_loc = 0
                self.probes_ik0_loc = []
                self.probes_ik1_loc = []
                self.probes_ik2_loc = []

        else:
            # no files were found : initialize from params
            INIT_FROM_PARAMS = True
            # pair kx,ky,kz with k0,k1,k2
            iksmax = np.array([ikzmax, ikymax, ikxmax])
            iksmin = np.array([1 - ikzmax, 1 - ikymax, 0])
            ik0max, ik1max, ik2max = iksmax[self.dims_order]
            ik0min, ik1min, ik2min = iksmin[self.dims_order]

            # local probes indices
            k0_adim_loc, k1_adim_loc, k2_adim_loc = oper.oper_fft.get_k_adim_loc()
            K0_adim, K1_adim, K2_adim = np.meshgrid(
                k0_adim_loc, k1_adim_loc, k2_adim_loc, indexing="ij"
            )
            cond_region = (
                (K0_adim >= ik0min)
                & (K0_adim <= ik0max)
                & (K1_adim >= ik1min)
                & (K1_adim <= ik1max)
                & (K2_adim >= ik2min)
                & (K2_adim <= ik2max)
            )
            (
                self.probes_ik0_loc,
                self.probes_ik1_loc,
                self.probes_ik2_loc,
            ) = np.where(cond_region)

            self.probes_nb_loc = self.probes_ik0_loc.size

            # local probes wavenumbers (nondimensional)
            self.probes_k0adim_loc = K0_adim[
                self.probes_ik0_loc,
                self.probes_ik1_loc,
                self.probes_ik2_loc,
            ]
            self.probes_k1adim_loc = K1_adim[
                self.probes_ik0_loc,
                self.probes_ik1_loc,
                self.probes_ik2_loc,
            ]
            self.probes_k2adim_loc = K2_adim[
                self.probes_ik0_loc,
                self.probes_ik1_loc,
                self.probes_ik2_loc,
            ]

            # initialize files info
            self.index_file = 0
            self.number_times_in_file = 0
            self.t_last_save = -self.period_save

        # size of a single write: nb_fields * probes_nb_loc + time
        probes_write_size = (
            len(self.keys_fields) * self.probes_nb_loc + 1
        ) * size_1_number
        self.max_number_times_in_file = int(
            self.file_max_size / probes_write_size
        )

        # initialize files
        if INIT_FROM_PARAMS and self.probes_nb_loc > 0:
            self._init_new_file(tmin_file=self.sim.time_stepping.t)

    def _init_files(self, arrays_1st_time=None):
        # we don't want to do anything when this function is called.
        pass

    def _init_new_file(self, tmin_file=None):
        """Initializes a new file"""
        if tmin_file is not None:
            # max number of digits = int(log10(t_end)) + 1
            # add .3f precision = 4 additional characters
            # +2 by anticipation of potential restarts
            str_width = int(np.log10(self.sim.params.time_stepping.t_end)) + 7
            ind_str = f"tmin{tmin_file:0{str_width}.3f}"
        else:
            ind_str = f"file{self.index_file:04}"
        self.path_file = self.path_dir / f"rank{mpi.rank:05}_{ind_str}.h5"
        with h5py.File(self.path_file, "w") as file:
            file.attrs["nb_proc"] = mpi.nb_proc
            file.attrs["dims_order"] = self.dims_order
            file.attrs["index_file"] = self.index_file
            file.attrs["probes_region"] = self.probes_region
            file.attrs["period_save"] = self.period_save
            file.attrs["max_number_times_in_file"] = self.max_number_times_in_file
            create_ds = file.create_dataset
            create_ds("probes_k0adim_loc", data=self.probes_k0adim_loc)
            create_ds("probes_k1adim_loc", data=self.probes_k1adim_loc)
            create_ds("probes_k2adim_loc", data=self.probes_k2adim_loc)
            create_ds("probes_ik0_loc", data=self.probes_ik0_loc)
            create_ds("probes_ik1_loc", data=self.probes_ik1_loc)
            create_ds("probes_ik2_loc", data=self.probes_ik2_loc)
            for key in self.keys_fields:
                create_ds(
                    key + "_Fourier_loc",
                    (self.probes_nb_loc, 1),
                    maxshape=(self.probes_nb_loc, None),
                    dtype=self.datatype,
                )
            create_ds("times", (1,), maxshape=(None,))

    def _write_to_file(self, data):
        """Writes a file with the temporal data"""
        with h5py.File(self.path_file, "a") as file:
            for k, v in data.items():
                dset = file[k]
                if k.startswith("times"):
                    dset.resize((self.number_times_in_file,))
                    dset[-1] = v
                else:
                    dset.resize((self.probes_nb_loc, self.number_times_in_file))
                    dset[:, -1] = v

    def _add_probes_data_to_dict(self, data, key):
        """Probes fields in Fourier space and append data to a dict object"""
        data[key + "_Fourier_loc"] = self.sim.state.get_var(f"{key}_fft")[
            self.probes_ik0_loc, self.probes_ik1_loc, self.probes_ik2_loc
        ]

    def _online_save(self):
        """Prepares data and writes to file"""
        if self.probes_nb_loc == 0:
            return
        tsim = self.sim.time_stepping.t
        if (
            tsim + 1e-15
        ) // self.period_save > self.t_last_save // self.period_save:
            # if max write number is reached, init new file
            if self.number_times_in_file >= self.max_number_times_in_file:
                self.index_file += 1
                self.number_times_in_file = 0
                self._init_new_file(tmin_file=self.sim.time_stepping.t)
            # get data from probes
            data = {"times": self.sim.time_stepping.t}
            data["times"] = self.sim.time_stepping.t
            for key in self.keys_fields:
                self._add_probes_data_to_dict(data, key)
            # write to file
            self.number_times_in_file += 1
            self._write_to_file(data)
            self.t_last_save = tsim

    def load_time_series(self, keys=None, tmin=0, tmax=None, dtype=None):
        """load time series from files"""

        if mpi.nb_proc > 1:
            raise RuntimeError(
                "This postprocessing function should not be called with MPI."
            )

        if keys is None:
            keys = self.keys_fields
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        # get ranks
        paths = sorted(self.path_dir.glob("rank*.h5"))
        ranks = sorted({int(p.name[4:9]) for p in paths})

        # get times and dimensions order from the files of first rank
        print(f"load times series...")
        paths_1st_rank = [
            p for p in paths if p.name.startswith(f"rank{ranks[0]:05}")
        ]

        with h5py.File(paths_1st_rank[0], "r") as file:
            order = file.attrs["dims_order"]
            region = file.attrs["probes_region"]
            if dtype is None:
                dtype = file[keys[0] + "_Fourier_loc"].dtype

        # get list of useful files, from tmin
        tmins_files = np.array([float(p.name[14:-3]) for p in paths_1st_rank])
        tmins_files, paths_1st_rank = filter_tmins_paths(
            tmin, tmins_files, paths_1st_rank
        )

        with Progress() as progress:
            npaths = len(paths_1st_rank)
            task_files = progress.add_task(
                "Getting times from rank 0...", total=npaths
            )

            times = []
            for ip, path in enumerate(paths_1st_rank):
                with h5py.File(path, "r") as file:
                    if tmins_files[ip] > tmax:
                        progress.update(task_files, completed=npaths)
                        break
                    times_file = file["times"][:]
                    cond_times = (times_file >= tmin) & (times_file <= tmax)
                    times.append(times_file[cond_times])
                    progress.update(task_files, advance=1)

        times = np.concatenate(times)

        tmin = times.min()
        tmax = times.max()
        print(f"tmin={tmin:8.6g}, tmax={tmax:8.6g}, nit={times.size}")

        # get sequential shape of Fourier space
        ikxmax, ikymax, ikzmax = region
        iksmax = np.array([ikzmax, ikymax, ikxmax])
        iksmin = np.array([1 - ikzmax, 1 - ikymax, 0])
        ik0max, ik1max, ik2max = iksmax[order]
        ik0min, ik1min, ik2min = iksmin[order]
        shape_series = (
            ik0max + 1 - ik0min,
            ik1max + 1 - ik1min,
            ik2max + 1 - ik2min,
            times.size,
        )

        # load series, rebuild as state_spect arrays + time
        series = {
            f"{k}_Fourier": np.empty(shape_series, dtype=dtype) for k in keys
        }
        with Progress() as progress:
            task_ranks = progress.add_task("Rearranging...", total=len(ranks))
            task_files = progress.add_task("Rank 00000...", total=npaths)
            # loop on all files, rank by rank
            for rank in ranks:
                paths_rank = [
                    p for p in paths if p.name.startswith(f"rank{rank:05}")
                ]

                # get list of useful files, from tmin
                tmins_files = np.array([float(p.name[14:-3]) for p in paths_rank])
                tmins_files, paths_rank = filter_tmins_paths(
                    tmin, tmins_files, paths_rank
                )

                npaths = len(paths_rank)
                progress.update(
                    task_files,
                    description=f"Rank {rank:05}...",
                    total=npaths,
                    completed=0,
                )

                # for a given rank, paths are sorted by time
                for ip, path_file in enumerate(paths_rank):
                    # break after the last useful file
                    if tmins_files[ip] > tmax:
                        progress.update(task_files, completed=npaths)
                        break

                    with h5py.File(path_file, "r") as file:
                        # time indices
                        times_file = file["times"][:]
                        its_file = get_arange_minmax(times_file, tmin, tmax)
                        tmin_keep = times_file[its_file[0]]
                        tmax_keep = times_file[its_file[-1]]
                        its = get_arange_minmax(times, tmin_keep, tmax_keep)

                        # k_adim_loc = global probes indices!
                        ik0 = file["probes_k0adim_loc"][:]
                        ik1 = file["probes_k1adim_loc"][:]
                        ik2 = file["probes_k2adim_loc"][:]

                        # load data at desired times for all keys_fields
                        for key in keys:
                            skey = key + "_Fourier"
                            data = file[skey + "_loc"][:, its_file]
                            for i, it in enumerate(its):
                                series[skey][ik0, ik1, ik2, it] = data[:, i]

                    # update rich task
                    progress.update(task_files, advance=1)

                # update rich task
                progress.update(task_ranks, advance=1)

        # add Ki_adim arrays, times and dims order
        k0_adim = np.r_[0 : ik0max + 1, ik0min:0]
        k1_adim = np.r_[0 : ik1max + 1, ik1min:0]
        k2_adim = np.r_[0 : ik2max + 1, ik2min:0]
        K0_adim, K1_adim, K2_adim = np.meshgrid(
            k0_adim, k1_adim, k2_adim, indexing="ij"
        )
        series.update(
            {
                "K0_adim": K0_adim,
                "K1_adim": K1_adim,
                "K2_adim": K2_adim,
                "times": times,
                "dims_order": order,
            }
        )

        return series

    def _compute_spectrum(self, data):
        if not hasattr(self, "f_sample"):
            paths = sorted(self.path_dir.glob("rank*.h5"))
            with h5py.File(paths[0], "r") as file:
                self.f_sample = 1.0 / file.attrs["period_save"]
            self.domega = 2 * pi * self.f_sample / data.shape[-1]

        # TODO: I'm not sure if detrend=False is good in prod, but it's much
        # better for testing
        freq, spectrum = signal.periodogram(
            data,
            fs=self.f_sample,
            scaling="spectrum",
            detrend=False,
            return_onesided=False,
        )
        return freq, spectrum / self.domega

    def compute_spectra(self, tmin=0, tmax=None, dtype=None):
        """compute spatiotemporal spectra from files"""
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        # load time series as state_spect arrays + times
        series = self.load_time_series(tmin=tmin, tmax=tmax, dtype=dtype)

        # compute spectra
        print("performing time fft...")

        spectra = {k: v for k, v in series.items() if k.startswith("K")}

        for key, data in series.items():
            if "_Fourier" not in key:
                continue
            key_spectrum = "spectrum_" + key.split("_Fourier")[0]
            freq, spectrum = self._compute_spectrum(data)
            spectra[key_spectrum] = spectrum

        spectra["omegas"] = 2 * pi * freq
        spectra["dims_order"] = series["dims_order"]

        return spectra
