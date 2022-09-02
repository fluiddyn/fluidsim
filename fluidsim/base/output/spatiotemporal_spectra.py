"""
Spatiotemporal Spectra
======================

Provides:

.. autoclass:: SpatioTemporalSpectra3D
   :members:
   :private-members:
   :undoc-members:

.. autoclass:: SpatioTemporalSpectra2D
   :members:
   :private-members:
   :undoc-members:

.. autoclass:: SpatioTemporalSpectraNS
   :members:
   :private-members:
   :undoc-members:
"""

from pathlib import Path
from logging import warn
from math import pi

import numpy as np
from scipy import signal
import h5py
from rich.progress import Progress
from fluidsim.util import ensure_radians

from fluiddyn.util import mpi
from fluidsim.util import open_patient
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
    print("arr", arr)
    raise ValueError(f"No index such that `arr[index] >= value (={value:.8g})`")


@boost
def find_index_first_g(arr: A, value: Uf32f64):
    """find the first index such that `arr[index] > value`"""
    for i, v in enumerate(arr):
        if v > value:
            return i
    print("arr", arr)
    raise ValueError(f"No index such that `arr[index] >= value (={value:.8g})`")


@boost
def find_index_first_l(arr: A, value: Uf32f64):
    """find the first index such that `arr[index] < value`"""
    for i, v in enumerate(arr):
        if v < value:
            return i
    print("arr", arr)
    raise ValueError(f"No index such that `arr[index] >= value (={value:.8g})`")


def filter_tmins_paths(tmin, tmins, paths):
    if tmins.size == 1:
        return tmins, paths
    delta_tmin = np.diff(tmins).min()
    start = find_index_first_l(tmin - tmins, delta_tmin)
    return tmins[start:], paths[start:]


def sort_files_tmin(paths, tmins=None):
    if not isinstance(paths, list):
        paths = list(paths)
    if tmins is None:
        tmins = np.array([float(p.name[14:-3]) for p in paths])
    return [
        path for (path, _) in sorted(zip(paths, tmins), key=lambda pair: pair[1])
    ]


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


class SpatioTemporalSpectra3D(SpecificOutput):
    """
    Computes the spatiotemporal spectra.
    """

    _tag = "spatiotemporal_spectra"
    nb_dim = 3

    @classmethod
    def _complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)
        params.output._set_child(
            cls._tag,
            attribs={
                "probes_region": None,
                "file_max_size": 10.0,  # MB
                "SAVE_AS_COMPLEX64": True,
            },
        )

        params.output.spatiotemporal_spectra._set_doc(
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
            if self.nb_dim == 3:
                ikxmax, ikymax, ikzmax = params_st_spec.probes_region
            else:
                ikxmax, ikymax = params_st_spec.probes_region
        else:
            ikxmax = ikymax = 4
            if self.nb_dim == 3:
                ikzmax = 4

        ikxmax = min(int(round(ikxmax)), params.oper.nx // 2)
        ikymax = min(int(round(ikymax)), params.oper.ny // 2)

        if self.nb_dim == 3:
            ikzmax = min(int(round(ikzmax)), params.oper.nz // 2)
            self.probes_region = ikxmax, ikymax, ikzmax
        else:
            self.probes_region = ikxmax, ikymax

        self.file_max_size = params_st_spec.file_max_size
        self.SAVE_AS_COMPLEX64 = params_st_spec.SAVE_AS_COMPLEX64

        # region must be int tuple
        ikxmax = int(ikxmax)
        ikymax = int(ikymax)
        if self.nb_dim == 3:
            ikzmax = int(ikzmax)

        # dimensions order in Fourier space
        if self.nb_dim == 3:
            self.dims_order = np.array(oper.oper_fft.get_dimX_K())
        else:
            self.dims_order = np.arange(2)
            if oper.oper_fft.get_is_transposed():
                self.dims_order = self.dims_order[::-1]

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
        paths = sort_files_tmin(self.path_dir.glob("rank*.h5"))
        if paths:
            # check values in files
            with open_patient(paths[0], "r") as file:
                if file.attrs["nb_proc"] != mpi.nb_proc:
                    raise ValueError(
                        f"process number ({mpi.nb_proc}) is different from "
                        f"process number in file ({file.attrs['nb_proc']})"
                    )
                if (file.attrs["dims_order"] != self.dims_order).any():
                    raise ValueError("dimensions order is different from files")
                if (file.attrs["probes_region"] != self.probes_region).any():
                    raise ValueError("probes region is different from files")
            # init from files
            INIT_FROM_PARAMS = False
            paths_rank = [
                p for p in paths if p.name.startswith(f"rank{mpi.rank:05}")
            ]
            if paths_rank:
                self.path_file = paths_rank[-1]
                with open_patient(self.path_file, "r") as file:
                    self.index_file = file.attrs["index_file"]
                    self.probes_k0adim_loc = file["probes_k0adim_loc"][:]
                    self.probes_ik0_loc = file["probes_ik0_loc"][:]
                    self.probes_k1adim_loc = file["probes_k1adim_loc"][:]
                    self.probes_ik1_loc = file["probes_ik1_loc"][:]

                    if self.nb_dim == 3:
                        self.probes_k2adim_loc = file["probes_k2adim_loc"][:]
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

                with open_patient(paths[-1], "r") as file:
                    self.t_last_save = file["times"][-1]

        else:
            # no files were found : initialize from params
            INIT_FROM_PARAMS = True

            if self.nb_dim == 3:
                # pair kx,ky,kz with k0,k1,k2
                iksmax = np.array([ikzmax, ikymax, ikxmax])
                iksmin = np.array([1 - ikzmax, 1 - ikymax, 0])
                ik0max, ik1max, ik2max = iksmax[self.dims_order]
                ik0min, ik1min, ik2min = iksmin[self.dims_order]

                # local probes indices
                (
                    k0_adim_loc,
                    k1_adim_loc,
                    k2_adim_loc,
                ) = oper.oper_fft.get_k_adim_loc()
                K0_adim, K1_adim, K2_adim = np.meshgrid(
                    k0_adim_loc, k1_adim_loc, k2_adim_loc, indexing="ij"
                )
            else:
                iksmax = np.array([ikymax, ikxmax])
                iksmin = np.array([1 - ikymax, 0])
                ik0max, ik1max = iksmax[self.dims_order]
                ik0min, ik1min = iksmin[self.dims_order]

                kx_adim_loc = np.array(
                    (oper.kx_loc / oper.deltakx).round(), dtype=int
                )
                ky_adim_loc = np.array(
                    (oper.ky_loc / oper.deltaky).round(), dtype=int
                )
                if oper.oper_fft.get_is_transposed():
                    k0_adim_loc = kx_adim_loc
                    k1_adim_loc = ky_adim_loc
                else:
                    k0_adim_loc = ky_adim_loc
                    k1_adim_loc = kx_adim_loc

                K0_adim, K1_adim = np.meshgrid(
                    k0_adim_loc, k1_adim_loc, indexing="ij"
                )

            cond_region = (
                (K0_adim >= ik0min)
                & (K0_adim <= ik0max)
                & (K1_adim >= ik1min)
                & (K1_adim <= ik1max)
            )

            if self.nb_dim == 3:
                cond_region = (
                    cond_region & (K2_adim >= ik2min) & (K2_adim <= ik2max)
                )

            if self.nb_dim == 3:
                (
                    self.probes_ik0_loc,
                    self.probes_ik1_loc,
                    self.probes_ik2_loc,
                ) = np.where(cond_region)
            else:
                (
                    self.probes_ik0_loc,
                    self.probes_ik1_loc,
                ) = np.where(cond_region)

            self.probes_nb_loc = self.probes_ik0_loc.size

            # local probes wavenumbers (nondimensional)
            self.probes_k0adim_loc = self._get_data_probe_from_field(K0_adim)
            self.probes_k1adim_loc = self._get_data_probe_from_field(K1_adim)

            if self.nb_dim == 3:
                self.probes_k2adim_loc = self._get_data_probe_from_field(K2_adim)

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
        with open_patient(self.path_file, "w") as file:
            file.attrs["nb_proc"] = mpi.nb_proc
            file.attrs["dims_order"] = self.dims_order
            file.attrs["index_file"] = self.index_file
            file.attrs["probes_region"] = self.probes_region
            file.attrs["period_save"] = self.period_save
            file.attrs["max_number_times_in_file"] = self.max_number_times_in_file
            create_ds = file.create_dataset
            create_ds("probes_k0adim_loc", data=self.probes_k0adim_loc)
            create_ds("probes_ik0_loc", data=self.probes_ik0_loc)
            create_ds("probes_k1adim_loc", data=self.probes_k1adim_loc)
            create_ds("probes_ik1_loc", data=self.probes_ik1_loc)
            if self.nb_dim == 3:
                create_ds("probes_k2adim_loc", data=self.probes_k2adim_loc)
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
        with open_patient(self.path_file, "a") as file:
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
        data[key + "_Fourier_loc"] = self._get_data_probe_from_field(
            self.sim.state.get_var(f"{key}_fft")
        )

    def _online_save(self):
        """Prepares data and writes to file"""
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
            for key in self.keys_fields:
                self._add_probes_data_to_dict(data, key)
            # write to file
            self.number_times_in_file += 1
            if self.probes_nb_loc > 0:
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

        # get ranks
        paths = sort_files_tmin(self.path_dir.glob("rank*.h5"))
        ranks = sorted({int(p.name[4:9]) for p in paths})

        # get times and dimensions order from the files of first rank
        print("load times series...")
        paths_1st_rank = [
            p for p in paths if p.name.startswith(f"rank{ranks[0]:05}")
        ]

        with open_patient(paths_1st_rank[0], "r") as file:
            dims_order = file.attrs["dims_order"]
            region = file.attrs["probes_region"]
            if dtype is None:
                dtype = file[keys[0] + "_Fourier_loc"].dtype

        # get list of useful files, from tmin
        tmins_files = np.array([float(p.name[14:-3]) for p in paths_1st_rank])
        tmins_files, paths_1st_rank = filter_tmins_paths(
            tmin, tmins_files, paths_1st_rank
        )

        paths_1st_rank = sort_files_tmin(paths_1st_rank, tmins_files)
        tmins_files = sorted(tmins_files)

        if tmax is None:
            with open_patient(paths_1st_rank[-1], "r") as file:
                tmax = file["/times"][-1]

        with Progress() as progress:
            npaths = len(paths_1st_rank)
            task_files = progress.add_task(
                "Getting times from rank 0...", total=npaths
            )

            times = []
            for ip, path in enumerate(paths_1st_rank):
                with open_patient(path, "r") as file:
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
        if self.nb_dim == 3:
            ikxmax, ikymax, ikzmax = region
            iksmax = np.array([ikzmax, ikymax, ikxmax])
            iksmin = np.array([1 - ikzmax, 1 - ikymax, 0])
            ik0max, ik1max, ik2max = iksmax[dims_order]
            ik0min, ik1min, ik2min = iksmin[dims_order]
            shape_series = (
                ik0max + 1 - ik0min,
                ik1max + 1 - ik1min,
                ik2max + 1 - ik2min,
                times.size,
            )
        else:
            ikxmax, ikymax = region
            iksmax = np.array([ikymax, ikxmax])
            iksmin = np.array([1 - ikymax, 0])
            ik0max, ik1max = iksmax[dims_order]
            ik0min, ik1min = iksmin[dims_order]
            shape_series = (
                ik0max + 1 - ik0min,
                ik1max + 1 - ik1min,
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

                    with open_patient(path_file, "r") as file:
                        # time indices
                        times_file = file["times"][:]
                        if times_file[-1] < tmin:
                            progress.update(task_files, advance=1)
                            continue
                        its_file = get_arange_minmax(times_file, tmin, tmax)
                        tmin_keep = times_file[its_file[0]]
                        tmax_keep = times_file[its_file[-1]]
                        its = get_arange_minmax(times, tmin_keep, tmax_keep)

                        # k_adim_loc = global probes indices!
                        ik0 = file["probes_k0adim_loc"][:]
                        ik1 = file["probes_k1adim_loc"][:]
                        if self.nb_dim == 3:
                            ik2 = file["probes_k2adim_loc"][:]

                        # load data at desired times for all keys_fields
                        for key in keys:
                            skey = key + "_Fourier"
                            data = file[skey + "_loc"][:, its_file]

                            if self.nb_dim == 3:
                                for i, it in enumerate(its):
                                    series[skey][ik0, ik1, ik2, it] = data[:, i]
                            else:
                                for i, it in enumerate(its):
                                    series[skey][ik0, ik1, it] = data[:, i]

                    # update rich task
                    progress.update(task_files, advance=1)

                # update rich task
                progress.update(task_ranks, advance=1)

        # add Ki_adim arrays, times and dims order
        k0_adim = np.r_[0 : ik0max + 1, ik0min:0]
        k1_adim = np.r_[0 : ik1max + 1, ik1min:0]

        if self.nb_dim == 3:
            k2_adim = np.r_[0 : ik2max + 1, ik2min:0]
            K0_adim, K1_adim, K2_adim = np.meshgrid(
                k0_adim, k1_adim, k2_adim, indexing="ij"
            )
        else:
            K0_adim, K1_adim = np.meshgrid(k0_adim, k1_adim, indexing="ij")

        series.update(
            {
                "K0_adim": K0_adim,
                "K1_adim": K1_adim,
                "times": times,
                "dims_order": dims_order,
            }
        )
        if self.nb_dim == 3:
            series["K2_adim"] = K2_adim

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

    def _get_data_probe_from_field(self, field):
        return field[
            self.probes_ik0_loc,
            self.probes_ik1_loc,
            self.probes_ik2_loc,
        ]


class SpatioTemporalSpectra2D(SpatioTemporalSpectra3D):
    nb_dim = 2

    def _get_data_probe_from_field(self, field):
        return field[
            self.probes_ik0_loc,
            self.probes_ik1_loc,
        ]


def _complete_name(name, dtype, save_urud):
    if dtype is not None:
        name += f"_{dtype}"
    if save_urud:
        name += "_urud"
    return name + ".h5"


class SpatioTemporalSpectraNS:
    def _get_path_saved_spectra(self, tmin, tmax, dtype, save_urud):
        if tmax is None:
            tmax = self._get_default_tmax()

        # we first check if a file corresponds to tmin and tmax
        # but we don't know how tmin/tmax are formatted
        for_glob = _complete_name("periodogram_*_*", dtype, save_urud)
        for path in self.path_dir.glob(for_glob):
            if "_temporal" in path.name:
                continue
            if (tmin, tmax) == tuple(float(s) for s in path.stem.split("_")[1:3]):
                return path

        name = _complete_name(
            f"periodogram_{float(tmin)}_{float(tmax)}", dtype, save_urud
        )
        return self.path_dir / name

    def _get_path_saved_tspectra(self, tmin, tmax, dtype, save_urud):
        if tmax is None:
            tmax = self._get_default_tmax()

        # we first check if a file corresponds to tmin and tmax
        # but we don't know how tmin/tmax are formatted
        for_glob = _complete_name("periodogram_temporal_*_*", dtype, save_urud)
        for path in self.path_dir.glob(for_glob):
            if (tmin, tmax) == tuple(float(s) for s in path.stem.split("_")[2:4]):
                return path

        name = _complete_name(
            f"periodogram_temporal_{float(tmin)}_{float(tmax)}", dtype, save_urud
        )
        return self.path_dir / name

    def save_spectra_kzkhomega(
        self, tmin=0, tmax=None, dtype=None, save_urud=False
    ):
        """
        save:
            - the spatiotemporal spectra, with a cylindrical average in k-space
            - the temporal spectra, with an average on the whole k-space
        """
        if tmax is None:
            tmax = self._get_default_tmax()

        # compute spectra
        print("Computing spectra...")
        spectra = self.compute_spectra(tmin=tmin, tmax=tmax, dtype=dtype)

        # get kz, kh
        params_oper = self.sim.params.oper
        deltakx = 2 * pi / params_oper.Lx
        order = spectra["dims_order"]
        KX = deltakx * spectra[f"K{order[-1]}_adim"]
        kx_max = self.sim.params.oper.nx // 2 * deltakx

        if self.nb_dim == 3:
            deltaky = 2 * pi / params_oper.Ly
            deltakz = 2 * pi / params_oper.Lz
            _deltakhs = [deltakx, deltaky]
            _ideltakh = np.argmax(_deltakhs)
            deltakh = _deltakhs[_ideltakh]
            KY = deltaky * spectra[f"K{order[1]}_adim"]
            KH = np.sqrt(KX**2 + KY**2)
            khmax_spectra = [KX, KY][_ideltakh].max()
            del KY
        else:
            # in 2d, vertical (here "z") is y
            deltakz = 2 * pi / params_oper.Ly
            KH = abs(KX)
            deltakh = deltakx
            khmax_spectra = KX.max()

        KZ = deltakz * spectra[f"K{order[0]}_adim"]

        kz_spectra = np.arange(0, KZ.max() + 1e-15, deltakz)

        nkh_spectra = max(2, int(khmax_spectra / deltakh))
        kh_spectra = deltakh * np.arange(nkh_spectra)

        # get one-sided frequencies
        omegas = spectra["omegas"]
        nomegas = (omegas.size + 1) // 2
        omegas_onesided = abs(omegas[:nomegas])

        # kzkhomega : perform cylindrical average
        # temporal spectra : average on Fourier space
        spectra_kzkhomega = {
            "kz_spectra": kz_spectra,
            "kh_spectra": kh_spectra,
            "omegas": omegas_onesided,
        }
        tspectra = {"omegas": omegas_onesided}
        for key, data in spectra.items():
            if not key.startswith("spectrum_"):
                continue
            spectra_kzkhomega[key] = self.compute_spectrum_kzkhomega(
                np.ascontiguousarray(data), kh_spectra, kz_spectra, KX, KZ, KH
            )
            tspectrum = self._sum_wavenumber(data, KX, kx_max)
            # one-sided frequencies
            tspectrum_onesided = np.zeros(nomegas)
            tspectrum_onesided[0] = tspectrum[0]
            tspectrum_onesided[1:] = (
                tspectrum[1:nomegas] + tspectrum[-1:-nomegas:-1]
            )
            tspectra[key] = tspectrum_onesided

        del spectra

        # total kinetic energy
        if self.nb_dim == 3:
            spectra_kzkhomega["spectrum_K"] = 0.5 * (
                spectra_kzkhomega["spectrum_vx"]
                + spectra_kzkhomega["spectrum_vy"]
                + spectra_kzkhomega["spectrum_vz"]
            )
            tspectra["spectrum_K"] = 0.5 * (
                tspectra["spectrum_vx"]
                + tspectra["spectrum_vy"]
                + tspectra["spectrum_vz"]
            )
        else:
            spectra_kzkhomega["spectrum_K"] = 0.5 * (
                spectra_kzkhomega["spectrum_ux"]
                + spectra_kzkhomega["spectrum_uy"]
            )
            tspectra["spectrum_K"] = 0.5 * (
                tspectra["spectrum_ux"] + tspectra["spectrum_uy"]
            )

        # potential energy
        try:
            N = self.sim.params.N
            spectra_kzkhomega["spectrum_A"] = (
                0.5 / N**2 * spectra_kzkhomega["spectrum_b"]
            )
            tspectra["spectrum_A"] = 0.5 / N**2 * tspectra["spectrum_b"]
        except AttributeError:
            pass

        # save to files
        path_file = self._get_path_saved_spectra(tmin, tmax, dtype, save_urud)
        with h5py.File(path_file, "w") as file:
            file.attrs["tmin"] = tmin
            file.attrs["tmax"] = tmax
            for key, val in spectra_kzkhomega.items():
                file.create_dataset(key, data=val)

        path_file = self._get_path_saved_tspectra(tmin, tmax, dtype, save_urud)
        with h5py.File(path_file, "w") as file:
            file.attrs["tmin"] = tmin
            file.attrs["tmax"] = tmax
            for key, val in tspectra.items():
                file.create_dataset(key, data=val)

        # toroidal/poloidal decomposition
        if save_urud:
            print("Computing ur, ud spectra...")
            spectra_urud_kzkhomega = {}
            tspectra_urud = {}
            spectra = self.compute_spectra_urud(tmin=tmin, tmax=tmax, dtype=dtype)

            for key, data in spectra.items():
                if not key.startswith("spectrum_"):
                    continue
                spectra_urud_kzkhomega[key] = self.compute_spectrum_kzkhomega(
                    np.ascontiguousarray(data), kh_spectra, kz_spectra, KX, KZ, KH
                )
                spectra_kzkhomega[key] = spectra_urud_kzkhomega[key]
                tspectrum = self._sum_wavenumber(data, KX, kx_max)
                # one-sided frequencies
                tspectrum_onesided = np.zeros(nomegas)
                tspectrum_onesided[0] = tspectrum[0]
                tspectrum_onesided[1:] = (
                    tspectrum[1:nomegas] + tspectrum[-1:-nomegas:-1]
                )
                tspectra_urud[key] = tspectrum_onesided
                tspectra[key] = tspectra_urud[key]

            path_file = self._get_path_saved_spectra(tmin, tmax, dtype, save_urud)
            with h5py.File(path_file, "a") as file:
                for key, val in spectra_urud_kzkhomega.items():
                    file.create_dataset(key, data=val)

            path_file = self._get_path_saved_tspectra(
                tmin, tmax, dtype, save_urud
            )
            with h5py.File(path_file, "a") as file:
                for key, val in tspectra_urud.items():
                    file.create_dataset(key, data=val)

        return spectra_kzkhomega, tspectra

    def load_spectra_kzkhomega(
        self, tmin=0, tmax=None, dtype=None, save_urud=False
    ):
        """load kzkhomega spectra from file"""
        spectra = {}

        path_file = self._get_path_saved_spectra(tmin, tmax, dtype, save_urud)
        with h5py.File(path_file, "r") as file:
            for key in file.keys():
                spectra[key] = file[key][...]

        return spectra

    def compute_omega_emp_vs_kzkh(
        self,
        spectra_kzkhomega,
        key_spect="spectrum_b",
    ):
        r"""Compute empirical frequency and fluctuation from the spatiotemporal spectra:

        .. math::

          \omega_{emp}(k_h, k_z) =
            \frac{\int ~ \omega ~ S(k_h, k_z, \omega)
            ~ \mathrm{d}\omega}{\int ~ S(k_h, k_z, \omega) ~ \mathrm{d}\omega},

          \delta \omega_{emp}(k_h, k_z) =
            \sqrt{\frac{\int ~ (\omega - \omega_{emp})^2 ~ S(k_h, k_z, \omega)
            ~ \mathrm{d}\omega}{\int ~ S(k_h, k_z, \omega) ~ \mathrm{d}\omega}},

        where :math:`\omega_{emp}` is the empirical frequency and :math:`\delta
        \omega_{emp}` is the empirical frequency fluctuation. :math:`S(k_h, k_z, \omega)` is the spectra
        of `key_spect`.
        """

        spectrum = spectra_kzkhomega[key_spect]
        kh_spectra = spectra_kzkhomega["kh_spectra"]
        kz_spectra = spectra_kzkhomega["kz_spectra"]
        omegas = spectra_kzkhomega["omegas"]

        # khv, kzv = np.meshgrid(kh_spectra, kz_spectra)
        omega_emp = np.zeros((len(kz_spectra), len(kh_spectra)))
        delta_omega_emp = np.zeros((len(kz_spectra), len(kh_spectra)))
        omega_norm = np.zeros((len(kz_spectra), len(kh_spectra)))

        # we compute omega_emp first
        for io in range(len(omegas)):
            omega_emp += omegas[io] * spectrum[:, :, io]
            omega_norm += spectrum[:, :, io]
        omega_emp = omega_emp / omega_norm

        # then we conpute delta_omega_emp
        for io in range(len(omegas)):
            delta_omega_emp += ((omegas[io] - omega_emp) ** 2) * spectrum[
                :, :, io
            ]
        delta_omega_emp = (np.divide(delta_omega_emp, omega_norm)) ** 0.5
        return omega_emp, delta_omega_emp

    def plot_kzkhomega(
        self,
        key_field="b",
        tmin=0,
        tmax=None,
        dtype=None,
        equation=None,
        xmax=None,
        ymax=None,
        cmap=None,
        vmin=None,
        vmax=None,
        plot_omega_emp=False,
        linscale=False,
    ):
        """plot the spatiotemporal spectra, with a cylindrical average in k-space

        equation must start with 'omega=', 'kh=', 'kz=', 'ikh=' or 'ikz='.

        For 3d solvers, `key_field` can be in `State.keys_state_phys + ["Khd", "Khr", "Kp"]`.

        """
        keys_plot = self.keys_fields.copy()
        if self.nb_dim == 3:
            keys_plot.extend(["Khd", "Khr", "Kp"])

        if key_field is None:
            key_field = keys_plot[0]
        if key_field not in keys_plot:
            raise KeyError(f"possible keys are {keys_plot}")
        if tmax is None:
            tmax = self._get_default_tmax()

        key_spect = "spectrum_" + key_field
        if key_field.startswith("Kh") or key_field.startswith("Kp"):
            save_urud = True
        else:
            save_urud = False

        path_file = self._get_path_saved_spectra(tmin, tmax, dtype, save_urud)
        path_urud = self._get_path_saved_spectra(tmin, tmax, dtype, True)
        if path_urud.exists() and not path_file.exists():
            path_file = path_urud

        # compute and save spectra if needed
        if not path_file.exists():
            if self.nb_dim == 3:
                self.save_spectra_kzkhomega(
                    tmin=tmin, tmax=tmax, dtype=dtype, save_urud=save_urud
                )
            else:
                self.save_spectra_kzkhomega(tmin=tmin, tmax=tmax, dtype=dtype)

        # load spectrum
        spectra_kzkhomega = {}
        with h5py.File(path_file, "r") as file:
            if key_spect.startswith("spectrum_Kp"):
                spectrum = file["spectrum_Khd"][:] + 0.5 * file["spectrum_vz"][:]
            else:
                spectrum = file[key_spect][...]
            if dtype == "complex64":
                float_dtype = "float32"
            elif dtype == "complex128":
                float_dtype = "float64"
            if dtype:
                spectrum = spectrum.astype(float_dtype)
            spectra_kzkhomega[key_spect] = spectrum
            spectra_kzkhomega["kh_spectra"] = file["kh_spectra"][...]
            spectra_kzkhomega["kz_spectra"] = file["kz_spectra"][...]
            spectra_kzkhomega["omegas"] = file["omegas"][...]

        # compute omega_emp if asked
        if plot_omega_emp:
            omega_emp, delta_omega_emp = self.compute_omega_emp_vs_kzkh(
                spectra_kzkhomega=spectra_kzkhomega, key_spect=key_spect
            )

        # slice along equation
        if equation is None:
            equation = f"omega=0"
        elif equation.startswith("kh="):
            kh = eval(equation[len("kh=") :])
            kh_spectra = spectra_kzkhomega["kh_spectra"]
            ikh = abs(kh_spectra - kh).argmin()
            equation = f"ikh={ikh}"
        elif equation.startswith("kz="):
            kz = eval(equation[len("kz=") :])
            kz_spectra = spectra_kzkhomega["kz_spectra"]
            ikz = abs(kz_spectra - kz).argmin()
            equation = f"ikz={ikz}"

        if equation.startswith("omega="):
            try:
                variables = dict(N=self.sim.params.N)
            except AttributeError:
                variables = dict()
            omega = eval(equation[len("omega=") :], variables)
            omegas = spectra_kzkhomega["omegas"]
            iomega = abs(omegas - omega).argmin()
            spect = spectra_kzkhomega[key_spect][:, :, iomega]
            xaxis = np.arange(spectra_kzkhomega["kh_spectra"].size)
            yaxis = np.arange(spectra_kzkhomega["kz_spectra"].size)
            xlabel = r"$k_h/\delta k_h$"
            ylabel = r"$k_z/\delta k_z$"
            omega = omegas[iomega]
            equation = r"$\omega=$" + f"{omega:.2g}"
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                equation = r"$\omega/N=$" + f"{omega/N:.2g}"
            except AttributeError:
                pass
        elif equation.startswith("ikh="):
            ikh = eval(equation[len("ikh=") :])
            kh_spectra = spectra_kzkhomega["kh_spectra"]
            spect = spectra_kzkhomega[key_spect][:, ikh, :].transpose()
            if plot_omega_emp:
                omega_emp = omega_emp[:, ikh]
                delta_omega_emp = delta_omega_emp[:, ikh]

            xaxis = np.arange(spectra_kzkhomega["kz_spectra"].size)
            yaxis = spectra_kzkhomega["omegas"]
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                yaxis /= N
            except AttributeError:
                pass

            xlabel = r"$k_z/\delta k_z$"
            ylabel = r"$\omega/N$"
            kh = kh_spectra[ikh]
            equation = f"$k_h = {ikh}\\delta k_h = {kh:.2g}$"
        elif equation.startswith("ikz="):
            ikz = eval(equation[len("ikz=") :])
            kz_spectra = spectra_kzkhomega["kz_spectra"]
            spect = spectra_kzkhomega[key_spect][ikz, :, :].transpose()
            if plot_omega_emp:
                omega_emp = omega_emp[ikz, :]
                delta_omega_emp = delta_omega_emp[ikz, :]

            xaxis = np.arange(spectra_kzkhomega["kh_spectra"].size)
            yaxis = spectra_kzkhomega["omegas"]
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                yaxis /= N
            except AttributeError:
                pass

            xlabel = r"$k_h/\delta k_h$"
            ylabel = r"$\omega/N$"
            kz = kz_spectra[ikz]
            equation = f"$k_z = {ikz}\\delta k_z = {kz:.2g}$"
        else:
            raise NotImplementedError(
                "equation must start with 'omega=', 'kh=', 'kz=', 'ikh=' or 'ikz='"
            )

        fig, ax = self.output.figure_axe()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # no log(0)
        spect += 1e-15

        if not linscale:
            spect_to_plot = np.log10(spect)
        else:
            spect_to_plot = spect

        im = ax.pcolormesh(
            xaxis,
            yaxis,
            spect_to_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
        )
        fig.colorbar(im)

        ax.set_title(
            f"{key_field} spatiotemporal spectra {equation}\n"
            f"tmin={tmin:.3f}, tmax={tmax:.3f}\n" + self.output.summary_simul
        )

        # add dispersion relation : omega = N * kh / sqrt(kh ** 2 + kz ** 2)
        try:
            N = self.sim.params.N
        except AttributeError:
            return
        dkz_over_dkh = (
            spectra_kzkhomega["kz_spectra"][1]
            / spectra_kzkhomega["kh_spectra"][1]
        )
        if equation.startswith(r"$\omega"):
            if omega > 0 and omega <= N:
                ikz_disp = np.sqrt(N**2 / omega**2 - 1) / dkz_over_dkh * xaxis
                ax.plot(xaxis, ikz_disp, "k-", linewidth=2)
        elif equation.startswith(r"$k_h"):
            omega_disp = ikh / np.sqrt(ikh**2 + dkz_over_dkh**2 * xaxis**2)
            ax.plot(xaxis, omega_disp, "k-", linewidth=2)
        elif equation.startswith(r"$k_z"):
            omega_disp = xaxis / np.sqrt(
                xaxis**2 + dkz_over_dkh**2 * ikz**2
            )
            ax.plot(xaxis, omega_disp, "k-", linewidth=2)
        else:
            raise ValueError("wrong equation for dispersion relation")

        # set axis limits after plotting dispersion relation
        if xmax is None:
            xmax = xaxis.max()
        if ymax is None:
            ymax = yaxis.max()
        ax.set_xlim((0, xmax))
        ax.set_ylim((0, ymax))

        # add empirical omega and broadening
        if plot_omega_emp:
            ax.plot(xaxis, omega_emp / N, "r-", linewidth=2)
            ax.plot(
                xaxis, (omega_emp + 0.5 * delta_omega_emp) / N, "r--", linewidth=1
            )
            ax.plot(
                xaxis, (omega_emp - 0.5 * delta_omega_emp) / N, "r--", linewidth=1
            )
            return omega_emp, delta_omega_emp, omega_disp

    def compute_spectra_urud(self, tmin=0, tmax=None, dtype=None):
        raise NotImplementedError

    def compute_temporal_spectra(
        self, tmin=0, tmax=None, dtype=None, compute_urud=False
    ):
        """compute the temporal spectra by averaging over Fourier space"""
        tspectra = {}

        # compute kxkykzomega spectra
        spectra = self.compute_spectra(tmin=tmin, tmax=tmax, dtype=dtype)
        if compute_urud:
            spectra.update(
                self.compute_spectra_urud(tmin=tmin, tmax=tmax, dtype=dtype)
            )

        # one-sided frequencies
        nomegas = (spectra["omegas"].size + 1) // 2
        tspectra["omegas"] = spectra["omegas"][:nomegas]

        order = spectra["dims_order"]
        KX = spectra[f"K{order[-1]}_adim"]
        deltakx = 2 * pi / self.sim.params.oper.Lx
        kx_max = self.sim.params.oper.nx // 2 * deltakx

        # average over Fourier space (kx,ky,kz)
        for key, spectrum in spectra.items():
            if not key.startswith("spectrum_"):
                continue
            tspectrum = self._sum_wavenumber(spectrum, KX, kx_max)
            # one-sided frequencies
            tspectrum_onesided = np.zeros(nomegas)
            tspectrum_onesided[0] = tspectrum[0]
            tspectrum_onesided[1:] = (
                tspectrum[1:nomegas] + tspectrum[-1:-nomegas:-1]
            )
            tspectra[key] = tspectrum_onesided

        # total kinetic energy
        if self.nb_dim == 3:
            tspectra["spectrum_K"] = 0.5 * (
                tspectra["spectrum_vx"]
                + tspectra["spectrum_vy"]
                + tspectra["spectrum_vz"]
            )
        else:
            tspectra["spectrum_K"] = 0.5 * (
                tspectra["spectrum_ux"] + tspectra["spectrum_uy"]
            )

        # potential energy
        try:
            N = self.sim.params.N
            tspectra["spectrum_A"] = 0.5 / N**2 * tspectra["spectrum_b"]
        except AttributeError:
            pass

        return tspectra

    def plot_temporal_spectra(
        self,
        key_field=None,
        tmin=0,
        tmax=None,
        xlim=None,
        ylim=None,
        dtype=None,
        xscale="log",
        coef_compensate=0,
        plot_resonant_modes=True,
    ):
        """plot the temporal spectra computed from the 4d spectra"""
        keys_plot = self.keys_fields.copy()
        if self.nb_dim == 3:
            keys_plot.extend(["Khd", "Khr", "Kp"])
        if key_field is None:
            key_field = keys_plot[0]
        if key_field not in keys_plot:
            raise KeyError(f"possible keys are {keys_plot}")
        if tmax is None:
            tmax = self._get_default_tmax()

        if self.nb_dim == 3:
            # much simpler for 3d
            save_urud = True
        else:
            save_urud = False
        path_file = self._get_path_saved_tspectra(tmin, tmax, dtype, save_urud)
        if path_file.exists():
            tspectra = self.load_temporal_spectra(
                tmin=tmin, tmax=tmax, dtype=dtype, save_urud=save_urud
            )
        else:
            tspectra = self.save_temporal_spectra(
                tmin=tmin, tmax=tmax, dtype=dtype, save_urud=save_urud
            )

        omegas = tspectra["omegas"]
        ylabel = "spectrum"

        if coef_compensate == 0:
            norm = 1.0
        else:
            omegas_no_0 = omegas.copy()
            omegas_no_0[0] = 1e-15
            norm = omegas_no_0**-coef_compensate
            norm[0] = np.nan
            ylabel = rf"$E(\omega) \omega^{{{coef_compensate}}}$"

        fig, ax = self.output.figure_axe()
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{key_field} temporal spectrum (tmin={tmin:.3f}, tmax={tmax:.3f})\n"
            + self.output.summary_simul
        )
        ax.set_xscale(xscale)
        ax.set_yscale("log")

        # specific to strat
        try:
            N = self.sim.params.N
        except AttributeError:
            ax.plot(
                omegas,
                tspectra["spectrum_" + key_field] / norm,
                "k",
                linewidth=2,
            )
        else:
            omegas = omegas / N
            if self.nb_dim == 3:
                # polo/toro/potential decomposition
                EKp = tspectra["spectrum_Khd"] + 0.5 * tspectra["spectrum_vz"]
                EKhr = tspectra["spectrum_Khr"]
                EK = EKhr + EKp
                ax.plot(omegas, EK / norm, "r", linewidth=2, label=r"$E_K$")
                try:
                    projection = self.sim.params.projection
                except AttributeError:
                    projection = None
                if projection != "poloidal":
                    ax.plot(
                        omegas,
                        EKhr / norm,
                        "r--",
                        linewidth=1,
                        label=r"$E_{K,toro}$",
                    )
                if projection != "toroidal":
                    ax.plot(
                        omegas,
                        EKp / norm,
                        "r-.",
                        linewidth=1,
                        label=r"$E_{K,polo}$",
                    )
            else:
                # kinetic energy
                EK = tspectra["spectrum_K"]
                ax.plot(omegas, EK / norm, "r", linewidth=2, label=r"$E_K$")
            EK_N = (EK / norm)[abs(omegas - 1).argmin()]  # value at N
            EA = tspectra["spectrum_A"]

            ax.plot(omegas, EA / norm, "b", linewidth=2, label=r"$E_A$")
            ax.set_title(
                f"kinetic/potential energy spectrum (tmin={tmin:.3f}, tmax={tmax:.3f})\n"
                + self.output.summary_simul
            )

            if plot_resonant_modes:
                if self.nb_dim == 3:
                    aspect_ratio = self.sim.oper.Lx / self.sim.oper.Lz
                else:
                    aspect_ratio = self.sim.oper.Lx / self.sim.oper.Ly

                def modes(nx, nz):
                    return np.sqrt(
                        nx**2 / (nx**2 + aspect_ratio**2 * nz**2)
                    )

                nxs = np.arange(1, 11)
                modes_nz1 = modes(nxs, 1)
                modes_nz2 = modes(nxs, 2)
                modes_y = np.full_like(modes_nz1, fill_value=10 * EK_N)

                ax.plot(modes_nz1, modes_y, "o", label="modes $n_z=1$")
                ax.plot(modes_nz2, modes_y * 3, "o", label="modes $n_z=2$")

            # omega^-2 scaling
            omegas_scaling = np.arange(0.4, 1 + 1e-15, 0.01)
            scaling_y = EK_N * omegas_scaling ** (-2 + coef_compensate)

            ax.plot(
                omegas_scaling, scaling_y, "k--", label=r"$\propto \omega^{-2}$"
            )

            # eye guide at N
            ax.axvline(1, linestyle="dotted")

            if xlim is not None:
                ax.set_xlim(xlim)

            if ylim is not None:
                ax.set_ylim(ylim)

            # eye guide at omega_f (specific to some forcing types)
            forcing_type = self.sim.params.forcing.type
            if forcing_type in ["watu_coriolis", "milestone"]:
                if forcing_type == "watu_coriolis":
                    omega_f = self.sim.params.forcing.watu_coriolis.omega_f
                elif forcing_type == "milestone":
                    period = self.sim.forcing.get_info()["period"]
                    omega_f = 2 * pi / period
                ax.axvline(omega_f / N, linestyle="dotted")

            elif forcing_type == "tcrandom_anisotropic":
                ymin, ymax = ax.get_ybound()
                factor = 2
                angle = ensure_radians(
                    self.params.forcing.tcrandom_anisotropic.angle
                )
                tmp = self.params.forcing.tcrandom_anisotropic
                try:
                    delta_angle = tmp.delta_angle
                except AttributeError:
                    # loading old simul with delta_angle
                    delta_angle = None

                if delta_angle is None:
                    omega_f = N * np.sin(angle)
                    ax.axvline(omega_f / N, linestyle="dotted")
                    omega_tmp = omega_f / N
                else:
                    delta_angle = ensure_radians(delta_angle)
                    omega_fmin = N * np.sin(angle - 0.5 * delta_angle)
                    omega_fmax = N * np.sin(angle + 0.5 * delta_angle)
                    omegas_f = N * np.logspace(-3, 3, 1000)
                    where = (omegas_f > omega_fmin) & (omegas_f < omega_fmax)
                    ax.fill_between(
                        omegas_f / N, ymin, ymax, where=where, alpha=0.5
                    )
                    omega_tmp = 0.5 * (omega_fmin + omega_fmax) / N

                ax.text(
                    omega_tmp,
                    factor * ymin,
                    r"$\omega_{f}/N$",
                    ha="center",
                    va="center",
                    size=10,
                )

            ax.set_xlabel(r"$\omega/N$")
            ax.legend()

    def load_temporal_spectra(
        self, tmin=0, tmax=None, dtype=None, save_urud=False
    ):
        """load temporal spectra from file"""
        tspectra = {}

        path_file = self._get_path_saved_tspectra(tmin, tmax, dtype, save_urud)
        with h5py.File(path_file, "r") as file:
            for key in file.keys():
                tspectra[key] = file[key][...]

        return tspectra

    def save_temporal_spectra(
        self, tmin=0, tmax=None, dtype=None, save_urud=False
    ):
        """compute temporal spectra from files"""
        if tmax is None:
            tmax = self._get_default_tmax()

        tspectra = self.compute_temporal_spectra(
            tmin=tmin, tmax=tmax, dtype=dtype, compute_urud=save_urud
        )

        path_file = self._get_path_saved_tspectra(tmin, tmax, dtype, save_urud)
        with h5py.File(path_file, "w") as file:
            file.attrs["tmin"] = tmin
            file.attrs["tmax"] = tmax
            for key, val in tspectra.items():
                file.create_dataset(key, data=val)

        return tspectra

    def _get_default_tmax(self):
        paths = list(self.path_dir.glob("rank*.h5"))
        if not paths:
            paths_periodo = list(self.path_dir.glob("periodogram_*.h5"))
            if paths_periodo:
                tmax = 0.0
                for path in paths_periodo:
                    with open_patient(path, "r") as file:
                        tmax = max(tmax, file.attrs["tmax"])
                return tmax
            return self.sim.params.time_stepping.t_end
        ranks = sorted({int(p.name[4:9]) for p in paths})
        paths_1st_rank = sort_files_tmin(
            p for p in paths if p.name.startswith(f"rank{ranks[0]:05}")
        )
        with open_patient(paths_1st_rank[-1], "r") as file:
            return file["/times"][-1]

    def get_spectra(self, tmin=0, tmax=None, dtype=None):
        save_urud = True
        path_file = self._get_path_saved_spectra(tmin, tmax, dtype, save_urud)
        if not path_file.exists():
            if self.nb_dim == 3:
                return self.save_spectra_kzkhomega(
                    tmin=tmin, tmax=tmax, dtype=dtype, save_urud=save_urud
                )
            else:
                return self.save_spectra_kzkhomega(
                    tmin=tmin, tmax=tmax, dtype=dtype
                )

        spectra_kzkhomega = self.load_spectra_kzkhomega(
            tmin, tmax, dtype, save_urud
        )
        tspectra = self.load_temporal_spectra(tmin, tmax, dtype, save_urud)

        return spectra_kzkhomega, tspectra
