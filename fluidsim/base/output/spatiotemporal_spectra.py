"""
SpatiotemporalSpectra (:mod:`fluidsim.solvers.ns3d.output.spatiotemporal_spectra`)
==============================================================================


Provides:

.. autoclass:: TemporalSpectra
   :members:
   :private-members:

"""

from pathlib import Path
from logging import warn

from math import pi
import numpy as np
from scipy import signal
import h5py
from rich.progress import track

from fluiddyn.util import mpi
from fluidsim.base.output.base import SpecificOutput


class SpatiotemporalSpectra(SpecificOutput):
    """
    Computes the spatiotemporal spectra.
    """

    _tag = "spatiotemporal_spectra"
    # _name_file = _tag + ".h5"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "spatiotemporal_spectra"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(
            tag,
            attribs={
                "HAS_TO_PLOT_SAVED": False,
                "probes_region": None,  # m
                "file_max_size": 10.0,  # MB
                "SAVE_AS_FLOAT32": False,
            },
        )

        params.output.temporal_spectra._set_doc(
            """
            probes_region: tuple (default:None)

                Boundaries of the region to record in the spectral domain.

                probes_region = (kxmax, kymax, kzmax), in units of params.oper.Lx^-1.

                The resulting ranges for each wavevectors are :

                    kx in [0 , kxmax]

                    ky in [-kymax+1 , kymax]

                    kz in [-kzmax+1 , kzmax]

                If None, set to kimax = deltaki.

            file_max_size: float (default: 10.0)

                Maximum size of one time series file, in megabytes.

            SAVE_AS_FLOAT32: bool (default: False)

                If set to true, probes data is saved as float32.

            """
        )

    def __init__(self, output):
        params = output.sim.params
        try:
            params_stspec = params.output.spatiotemporal_spectra
        except AttributeError:
            warn(
                "Cannot initialize spatiotemporal spectra output because "
                "`params` does not contain parameters for this class."
            )
            return

        super().__init__(
            output,
            period_save=params.output.periods_save.spatiotemporal_spectra,
            has_to_plot_saved=params_stspec.HAS_TO_PLOT_SAVED,
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

        deltakx = oper.deltakx
        deltaky = oper.deltaky
        deltakz = oper.deltakz

        if params_stspec.probes_region is not None:
            self.probes_region = params_stspec.probes_region
            kxmax, kymax, kzmax = self.probes_region
        else:
            kxmax = deltakx
            kymax = deltaky
            kzmax = deltakz
            self.probes_region = kxmax, kymax, kzmax

        self.file_max_size = params_stspec.file_max_size
        self.SAVE_AS_FLOAT32 = params_stspec.SAVE_AS_FLOAT32

        # make sure spectral region is not empty
        kxmax += 1e-15
        kymax += 1e-15
        kzmax += 1e-15

        # global probes wavenumbers
        kymin = 1 - kymax
        kzmin = 1 - kzmax
        self.probes_kx_seq = np.arange(0, kxmax, deltakx)
        self.probes_ky_seq = np.r_[0:kymax:deltaky, kymin:0:deltaky]
        self.probes_kz_seq = np.r_[0:kzmax:deltakz, kzmin:0:deltakz]

        # dimensions order in Fourier space
        self.dims_order = oper.oper_fft.get_dimX_K()

        # data directory
        if mpi.rank == 0:
            self.path_dir.mkdir(exist_ok=True)
        if mpi.nb_proc > 1:
            mpi.comm.barrier()

        # check for existing files
        paths = sorted(self.path_dir.glob("rank*.h5"))
        if paths:
            # check values in files
            with h5py.File(paths[0], "r") as file:
                if file.attrs["nb_proc"] != mpi.nb_proc:
                    raise ValueError("process number is different from files")
                if not np.allclose(file.attrs["dims_order"], self.dims_order):
                    raise ValueError("dimensions order is different from files")
                if not (
                    np.allclose(file["probes_kx_seq"][:], self.probes_kx_seq)
                    and np.allclose(file["probes_ky_seq"][:], self.probes_ky_seq)
                    and np.allclose(file["probes_kz_seq"][:], self.probes_kz_seq)
                ):
                    raise ValueError("probes position are different from files")
            # init from files
            paths = [p for p in paths if p.name.startswith(f"rank{mpi.rank:05}")]
            if paths:
                self.path_file = paths[-1]
                with h5py.File(self.path_file, "r") as file:
                    self.index_file = file.attrs["index_file"]
                    self.probes_kx_loc = file["probes_kx_loc"][:]
                    self.probes_ky_loc = file["probes_ky_loc"][:]
                    self.probes_kz_loc = file["probes_kz_loc"][:]
                    self.probes_ik0_loc = file["probes_ik0_loc"][:]
                    self.probes_ik1_loc = file["probes_ik1_loc"][:]
                    self.probes_ik2_loc = file["probes_ik2_loc"][:]
                    self.probes_nb_loc = self.probes_kx_loc.size
                    self.number_times_in_file = file["times"].size
                    self.t_last_save = file["times"][-1]
            else:
                # no probes in proc
                self.path_file = None
                self.index_file = 0
                self.number_times_in_file = 0
                self.probes_nb_loc = 0
                self.probes_kx_loc = []
                self.probes_ky_loc = []
                self.probes_kz_loc = []
                self.probes_ik0_loc = []
                self.probes_ik1_loc = []
                self.probes_ik2_loc = []

        else:
            # no files were found : initialize from params

            # local probes indices
            Kx = oper.Kx
            Ky = oper.Ky
            Kz = oper.Kz
            cond_region = (
                (Kx <= kxmax)
                & (Ky >= kymin)
                & (Ky <= kymax)
                & (Kz >= kzmin)
                & (Kz <= kzmax)
            )
            (
                self.probes_ik0_loc,
                self.probes_ik1_loc,
                self.probes_ik2_loc,
            ) = np.where(cond_region)

            # local probes wavenumbers
            self.probes_kx_loc = Kx[
                self.probes_ik0_loc, self.probes_ik1_loc, self.probes_ik2_loc
            ]
            self.probes_ky_loc = Ky[
                self.probes_ik0_loc, self.probes_ik1_loc, self.probes_ik2_loc
            ]
            self.probes_kz_loc = Kz[
                self.probes_ik0_loc, self.probes_ik1_loc, self.probes_ik2_loc
            ]

            self.probes_nb_loc = self.probes_kx_loc.size

            # initialize files
            self.index_file = 0
            self.number_times_in_file = 0
            self.t_last_save = -self.period_save
            if self.probes_nb_loc > 0:
                self._init_new_file()

        if self.SAVE_AS_FLOAT32:
            size_1_number = 4e-6
        else:
            # what is the size of a complex in python?
            size_1_number = 8e-6

        # size of a single write: nb_fields * probes_nb_loc + time
        probes_write_size = (
            len(self.keys_fields) * self.probes_nb_loc + 1
        ) * size_1_number
        self.max_number_times_in_file = int(
            self.file_max_size / probes_write_size
        )

    def _init_files(self, arrays_1st_time=None):
        # we don't want to do anything when this function is called.
        pass

    def _init_new_file(self):
        """Initializes a new file"""
        self.path_file = (
            self.path_dir / f"rank{mpi.rank:05}_file{self.index_file:04}.h5"
        )
        with h5py.File(self.path_file, "w") as file:
            file.attrs["nb_proc"] = mpi.nb_proc
            file.attrs["dims_order"] = self.dims_order
            file.attrs["index_file"] = self.index_file
            create_ds = file.create_dataset
            create_ds("probes_kx_seq", data=self.probes_kx_seq)
            create_ds("probes_ky_seq", data=self.probes_ky_seq)
            create_ds("probes_kz_seq", data=self.probes_kz_seq)
            create_ds("probes_kx_loc", data=self.probes_kx_loc)
            create_ds("probes_ky_loc", data=self.probes_ky_loc)
            create_ds("probes_kz_loc", data=self.probes_kz_loc)
            create_ds("probes_ik0_loc", data=self.probes_ik0_loc)
            create_ds("probes_ik1_loc", data=self.probes_ik1_loc)
            create_ds("probes_ik2_loc", data=self.probes_ik2_loc)
            for key in self.keys_fields:
                create_ds(
                    f"spect_{key}_loc",
                    (self.probes_nb_loc, 1),
                    maxshape=(self.probes_nb_loc, None),
                    dtype="complex",
                )
            create_ds("times", (1,), maxshape=(None,))

    def _write_to_file(self, data):
        """Writes a file with the temporal data"""
        with h5py.File(self.path_file, "a") as file:
            for k, v in data.items():
                dset = file[k]
                if k.startswith("times"):
                    dset.resize((self.number_times_in_file,))
                    if self.SAVE_AS_FLOAT32:
                        raise NotImplementedError
                    dset[-1] = v
                else:
                    dset.resize((self.probes_nb_loc, self.number_times_in_file))
                    if self.SAVE_AS_FLOAT32:
                        raise NotImplementedError
                    dset[:, -1] = v

    def _add_probes_data_to_dict(self, data_dict, key):
        """Probes fields in Fourier space and append data to a dict object"""
        data_dict[f"spect_{key}_loc"] = self.sim.state.get_var(f"{key}_fft")[
            self.probes_ik0_loc, self.probes_ik1_loc, self.probes_ik2_loc
        ]

    def _online_save(self):
        """Prepares data and writes to file"""
        if self.probes_nb_loc > 0:
            tsim = self.sim.time_stepping.t
            if (
                tsim + 1e-15
            ) // self.period_save > self.t_last_save // self.period_save:
                # if max write number is reached, init new file
                if self.number_times_in_file >= self.max_number_times_in_file:
                    self.index_file += 1
                    self.number_times_in_file = 0
                    self._init_new_file()
                # get data from probes
                data = {"times": self.sim.time_stepping.t}
                data["times"] = self.sim.time_stepping.t
                for key in self.keys_fields:
                    self._add_probes_data_to_dict(data, key)
                # write to file
                self.number_times_in_file += 1
                self._write_to_file(data)
                self.t_last_save = tsim

    def load_time_series(self, key=None, region=None, tmin=0, tmax=None):
        """load time series from files"""
        if key is None:
            key = self.keys_fields[0]
        key = f"spect_{key}_loc"
        if region is None:
            oper = self.sim.oper
            region = (oper.kxmax_spectra, oper.kymax_spectra, oper.kzmax_spectra)
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        kxmax, kymax, kzmax = region
        kymin = 1 - kymax
        kzmin = 1 - kzmax

        # get ranks
        paths = sorted(self.path_dir.glob("rank*.h5"))
        ranks = sorted({int(p.name[4:9]) for p in paths})

        # get times from the files of first rank
        times = []
        for path_file in paths:
            if not path_file.name.startswith(f"rank{ranks[0]:05}"):
                continue
            with h5py.File(path_file, "r") as file:
                times_file = file["times"][:]
                cond_times = (times_file >= tmin) & (times_file <= tmax)
                times.append(times_file[cond_times])
        times = np.concatenate(times)

        # load series
        series = []
        for rank in ranks:
            data = []
            for path_file in paths:
                if not path_file.name.startswith(f"rank{rank:05}"):
                    continue
                with h5py.File(path_file, "r") as file:
                    probes_kx = file["probes_kx_loc"][:]
                    probes_ky = file["probes_ky_loc"][:]
                    probes_kz = file["probes_kz_loc"][:]

                    cond_region = np.where(
                        (probes_kx <= kxmax)
                        & (probes_ky >= kymin)
                        & (probes_ky <= kymax)
                        & (probes_kz >= kzmin)
                        & (probes_kz <= kzmax)
                    )[0]

                    tmp = file[key][cond_region, :]

                    times_file = file["times"][:]
                    cond_times = (times_file >= tmin) & (times_file <= tmax)
                    data.append(tmp[:, cond_times])

            series.append(np.concatenate(data, axis=1))

        result = {key: series, "times": times}
        return result
